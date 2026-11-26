"""
Cyclic feedback graph for iterative specification generation and review.

Graph topology (all inside one Orchestrator):

    init ──[{}]──> input_task ──{"directories", "files"}──> read_code
                       │                                        │
                       │ {"request", ...}                 {"code": ...}
                       │                                        │
                       └──────────────> merge_inputs <──────────┘
                                              │
                                    {"code", "request", ...}
                                              v
                                         spec_writer
                                              │
                                    {"specification": ...}
                                              v
                                        reviewer_task
                                              │
        ┌─────────────────────────────────────┼─────────────────────────────┐
        │ ["retry"], prio 0                   │ ["edit"], prio 10           │ ["accept"]
        v                                     v                             v
    input_task                           spec_writer                     transit ──> run() output

Contract (activation policies):
- `input_task` must be constructed by the caller with `AnyPolicy()` so that it can
  be re-activated on `retry` (otherwise a `OncePolicy` would block the second run).
- `reviewer_task` must be constructed by the caller with `AnyPolicy()` so that it can
  be re-activated on every new specification coming from `spec_writer`.

`merge_inputs` is a transit (pass-through) node joining the two independent streams
that feed the writer: the request produced by `input_task` and the code produced by
`read_code`. It uses `AllPolicy()`, so it acts as a synchronization barrier and fires
only when BOTH incoming edges are fresh.

This join is required because the orchestrator commits task versions only after a
dispatch wave has been handled: `read_code` and `spec_writer` would otherwise be
activated within the same wave, and an `AnyPolicy()` writer would run on a snapshot
containing `request` but no `code` yet (and immediately fail). Merging the streams
first guarantees that every snapshot delivered to `spec_writer` from this branch is
complete.

`spec_writer` itself keeps `AnyPolicy()` because it has two mutually exclusive
feeders: the joined `merge_inputs` stream (initial run / retry) and the `edit` branch
coming straight from the reviewer, which already carries code and request.

The reviewer task is expected to emit, in its state, a mandatory `"__groups"` field
with one of the values: `"accept"`, `"edit"`, `"retry"` (and optionally
`"specification"`, `"request"` and `"code"` fields). The `__groups` value selects
which outgoing edge is considered fresh, so exactly one branch is taken per
reviewer run.
"""

from typing import Any, Dict, List, Optional, Set

from src.fab.task import Task, TaskId
from src.fab.orchestrator import Orchestrator
from src.fab.executor import InlineExecutor
from src.fab.policy import (
    ActivationPolicy, OncePolicy, AnyPolicy, AllPolicy,
    InputMergePolicy, SoftCartesianMergePolicy,
    FailurePolicy,
)
from src.tasks.ai_tasks import SpecWriterTask
from src.tasks.base_tasks import TransitTask, ConstantTask, ReadCodeFilesTask
from src.utils import AiClient


# Internal task ids (kept as constants so tests can reference them reliably).
INIT_TASK_ID: TaskId = "init"
READ_CODE_TASK_ID: TaskId = "read_code"
MERGE_TASK_ID: TaskId = "merge_inputs"
SPEC_WRITER_TASK_ID: TaskId = "spec_writer"
TRANSIT_TASK_ID: TaskId = "transit"


class SpecWriterPipelineTask(Task):
    """
    A composite Task that internally runs a cyclic feedback graph
    (init -> input -> {read_code | request} -> merge_inputs -> spec_writer ->
    reviewer -> {retry | edit | accept}).

    The outer `run()` returns `[{"specification": ...}]` once the reviewer
    emits `accept`, or `[]` if no accept happened (e.g. the graph quiesced
    on a retry/edit branch that produced nothing).

    Itself it acts as a normal `Task` in the outer graph, so it can be wired
    into a larger pipeline.
    """

    def __init__(
        self,
        task_id: TaskId,
        input_task: Task,
        reviewer_task: Task,
        ai_client: AiClient,
        allowed_extensions: Optional[Set[str]] = None,
        activation_policy: ActivationPolicy = OncePolicy(),
        merge_policy: Optional[InputMergePolicy] = None,
        failure_policy: Optional[FailurePolicy] = None,
    ):
        super().__init__(task_id, activation_policy, merge_policy, failure_policy)

        self._input_task = input_task
        self._reviewer_task = reviewer_task
        self._ai_client = ai_client

        # ------------------------------------------------------------------
        # Build the internal graph.
        # ------------------------------------------------------------------
        self._orchestrator = Orchestrator(InlineExecutor)

        # Initiating task: emits an empty payload once to seed the first run
        # of `input_task` (which otherwise has no fresh inputs to react to).
        self._init_task = ConstantTask(
            task_id=INIT_TASK_ID,
            output=[{}],
            activation_policy=OncePolicy(),
        )

        # Internal read code task: reads directories and files, outputs code.
        self._read_code = ReadCodeFilesTask(
            task_id=READ_CODE_TASK_ID,
            allowed_extensions=allowed_extensions,
            activation_policy=AnyPolicy(),
        )

        # Joining transit task: forwards its inputs verbatim, but activates only
        # when both the request (from input_task) and the code (from read_code)
        # are fresh. AllPolicy turns it into a barrier, so the writer never
        # receives a partially filled snapshot.
        self._merge_inputs = TransitTask(
            task_id=MERGE_TASK_ID,
            activation_policy=AllPolicy(),
            merge_policy=SoftCartesianMergePolicy(),
        )

        # Internal spec writer: fed either by the joined stream or by the
        # reviewer's `edit` branch; the cartesian merge lets code/request and
        # previous_specification be combined freely.
        self._spec_writer = SpecWriterTask(
            task_id=SPEC_WRITER_TASK_ID,
            ai_client=ai_client,
            activation_policy=AnyPolicy(),
            merge_policy=SoftCartesianMergePolicy(),
        )

        # Transit sink: accepts the specification on `accept` and forwards it
        # to the outer `run()`.
        self._transit = TransitTask(
            task_id=TRANSIT_TASK_ID,
            activation_policy=AnyPolicy(),
        )

        # Register all seven nodes: init, input, read_code, merge_inputs,
        # spec_writer, reviewer, transit.
        self._orchestrator.add_task(self._init_task)
        self._orchestrator.add_task(input_task)
        self._orchestrator.add_task(self._read_code)
        self._orchestrator.add_task(self._merge_inputs)
        self._orchestrator.add_task(self._spec_writer)
        self._orchestrator.add_task(reviewer_task)
        self._orchestrator.add_task(self._transit)

        # ------------------------------------------------------------------
        # Edges.
        # ------------------------------------------------------------------

        # init -> input_task: bootstrap edge, no aliasing, empty payload.
        self._orchestrator.add_edge(
            INIT_TASK_ID,
            input_task.task_id,
        )

        # input_task -> read_code: pass directories and files.
        self._orchestrator.add_edge(
            input_task.task_id,
            READ_CODE_TASK_ID,
            aliases={"directories": "directories", "files": "files"},
        )

        # input_task -> merge_inputs: the request side of the join.
        self._request_edge_id = self._orchestrator.add_edge(
            input_task.task_id,
            MERGE_TASK_ID,
            aliases={"request": "request"},
        )

        # read_code -> merge_inputs: the code side of the join.
        self._code_edge_id = self._orchestrator.add_edge(
            READ_CODE_TASK_ID,
            MERGE_TASK_ID,
            aliases={"code": "code"},
        )

        # merge_inputs -> spec_writer: the single, always complete data stream.
        self._merged_edge_id = self._orchestrator.add_edge(
            MERGE_TASK_ID,
            SPEC_WRITER_TASK_ID,
            aliases={"code": "code", "request": "request"},
        )

        # spec_writer -> reviewer_task: deliver the fresh specification.
        self._orchestrator.add_edge(
            SPEC_WRITER_TASK_ID,
            reviewer_task.task_id,
            aliases={"specification": "specification"},
        )

        # reviewer_task -> input_task: RETRY branch (retry with same inputs).
        self._retry_edge_id = self._orchestrator.add_edge(
            reviewer_task.task_id,
            input_task.task_id,
            groups=["retry"],
            priority=0,
        )

        # reviewer_task -> spec_writer: EDIT branch, elevated priority so that
        # on merge conflicts the newly edited data wins over the retained ones.
        self._edit_edge_id = self._orchestrator.add_edge(
            reviewer_task.task_id,
            SPEC_WRITER_TASK_ID,
            aliases={
                "specification": "previous_specification",
                "request": "request",
                "code": "code",
            },
            groups=["edit"],
            priority=10,
        )

        # reviewer_task -> transit: ACCEPT branch.
        self._accept_edge_id = self._orchestrator.add_edge(
            reviewer_task.task_id,
            TRANSIT_TASK_ID,
            aliases={"specification": "specification"},
            groups=["accept"],
        )

    # ----------------------------------------------------------------------
    # Task API
    # ----------------------------------------------------------------------

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Runs the internal graph to quiescence and returns the accepted
        specification (or an empty list if nothing was accepted).

        `inputs` is intentionally ignored: the pipeline seeds itself from the
        externally supplied `input_task` / `reviewer_task`. This keeps the
        composite transparent to the outer orchestrator while letting the
        caller own the concrete input/reviewer implementations.
        """
        _ = inputs  # Inputs are not used; the internal graph reads files itself.
        self._orchestrator.run()

        transit_state = self._transit.get_state
        if not transit_state:
            return []

        # TransitTask forwards its inputs verbatim, so the last committed
        # dict is the reviewed/accepted specification payload.
        last = transit_state[-1]
        if not isinstance(last, dict):
            return []

        specification = last.get("specification")
        if specification is None:
            return []

        return [{"specification": specification}]