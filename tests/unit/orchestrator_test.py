import time
import threading
from typing import Any, Dict, List, cast

import pytest

from src.fab.types import TaskId
from src.fab.task import Task, TaskStatus
from src.fab.policy import (
    ActivationPolicy, FailurePolicy,
    OncePolicy, AllPolicy, AnyPolicy, ExactPolicy, ActivationPolicyConfig,
    RetryFailurePolicy, SkipFailurePolicy, FailFailurePolicy,
    CartesianProductMergePolicy, SingleItemPerSourcePolicy, AllButOneSinglePolicy,
    FlattenMergePolicy,
)
from src.fab.executor import InlineExecutor, ThreadPoolExecutorBackend, CompletionEvent
from src.fab.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Test task implementations
# ---------------------------------------------------------------------------

class EchoTask(Task):
    """Returns a copy of its inputs under a fixed key."""

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"echo": dict(inputs)}]


class SumTask(Task):
    """Sums integer values from inputs."""

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        total = 0
        for v in inputs.values():
            if isinstance(v, dict) and "value" in v:
                total += int(v["value"])
        return [{"value": total}]


class ConstantTask(Task):
    """Always returns the same dict regardless of inputs."""

    def __init__(
        self,
        task_id: TaskId,
        activation_policy: ActivationPolicy,
        value: Dict[str, Any],
        failure_policy: FailurePolicy = FailFailurePolicy(),
    ):
        super().__init__(task_id, activation_policy, failure_policy=failure_policy)
        self._value = value

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [dict(self._value)]


class FailingTask(Task):
    """Always raises an exception."""

    def __init__(
        self,
        task_id: TaskId,
        activation_policy: ActivationPolicy,
        failure_policy: FailurePolicy = FailFailurePolicy(),
        message: str = "boom",
    ):
        super().__init__(task_id, activation_policy, failure_policy=failure_policy)
        self._message = message

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise RuntimeError(self._message)


class MutatingTask(Task):
    """Attempts to mutate the input dict — must not affect the graph state."""

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        for v in inputs.values():
            if isinstance(v, dict):
                v["mutated"] = True
        return [{"ok": True}]


class NonDictTask(Task):
    """Returns a list containing a non-dict element — orchestrator must reject it."""

    def run(self, inputs: Dict[str, Any]) -> Any:
        return ["not a dict"]


class CountingTask(Task):
    """Counts how many times it has been run."""

    def __init__(self, task_id: TaskId, activation_policy: ActivationPolicy):
        super().__init__(task_id, activation_policy)
        self.run_count_local = 0

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.run_count_local += 1
        return [{"count": self.run_count_local}]


class SlowTask(Task):
    """Sleeps for a configurable duration — used to simulate concurrency."""

    def __init__(self, task_id: TaskId, activation_policy: ActivationPolicy, delay: float):
        super().__init__(task_id, activation_policy)
        self._delay = delay
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.started.set()
        self.release.wait(timeout=5.0)
        time.sleep(self._delay)
        return [{"slow": True, "inputs": dict(inputs)}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_orchestrator(executor_cls=InlineExecutor, **kwargs) -> Orchestrator:
    return Orchestrator(executor_cls, **kwargs)


# ---------------------------------------------------------------------------
# Tests: graph construction
# ---------------------------------------------------------------------------

def test_add_task_and_edge():
    orch = build_orchestrator()
    t1 = Task("t1", OncePolicy())
    t2 = Task("t2", AllPolicy())
    orch.add_task(t1)
    orch.add_task(t2)
    edge_id = orch.add_edge("t1", "t2")

    assert edge_id in orch.edges
    assert orch.edges[edge_id].src == "t1"
    assert orch.edges[edge_id].dst == "t2"
    assert "t2" in [orch.edges[e].dst for e in orch.out_edges["t1"]]


def test_add_duplicate_task_raises():
    orch = build_orchestrator()
    orch.add_task(Task("t1", OncePolicy()))
    with pytest.raises(ValueError):
        orch.add_task(Task("t1", OncePolicy()))


def test_add_edge_missing_endpoint_raises():
    orch = build_orchestrator()
    orch.add_task(Task("t1", OncePolicy()))
    with pytest.raises(ValueError):
        orch.add_edge("t1", "missing")


def test_self_loop_is_allowed():
    orch = build_orchestrator()
    orch.add_task(Task("t1", AnyPolicy()))
    edge_id = orch.add_edge("t1", "t1")
    assert edge_id in orch.edges


def test_edge_id_is_uuid():
    orch = build_orchestrator()
    orch.add_task(Task("a", OncePolicy()))
    orch.add_task(Task("b", AllPolicy()))
    edge_id = orch.add_edge("a", "b")
    parts = edge_id.split("-")
    assert len(parts) == 5
    assert [len(p) for p in parts] == [8, 4, 4, 4, 12]


# ---------------------------------------------------------------------------
# Tests: basic execution
# ---------------------------------------------------------------------------

def test_linear_dag_once_policy():
    orch = build_orchestrator()
    orch.add_task(EchoTask("src", OncePolicy()))
    orch.add_task(EchoTask("dst", OncePolicy()))
    orch.add_edge("src", "dst")

    report = orch.run()

    assert report["tasks"]["src"]["status"] == "DONE"
    assert report["tasks"]["dst"]["status"] == "DONE"
    assert report["tasks"]["src"]["run_count"] == 1
    assert report["tasks"]["dst"]["run_count"] == 1
    assert report["tasks"]["src"]["version"] == 1

    dst_state = orch.tasks["dst"].state
    assert dst_state
    # state is now a list of dicts; each dict is one output snapshot
    assert "echo" in dst_state[0]


def test_all_policy_requires_all_fresh():
    orch = build_orchestrator()
    orch.add_task(EchoTask("a", OncePolicy()))
    orch.add_task(EchoTask("b", OncePolicy()))
    orch.add_task(EchoTask("c", AllPolicy()))
    orch.add_edge("a", "c")
    orch.add_edge("b", "c")

    report = orch.run()

    assert report["tasks"]["c"]["run_count"] == 1
    assert report["tasks"]["c"]["status"] == "DONE"


def test_any_policy_runs_on_first_input():
    orch = build_orchestrator()
    orch.add_task(EchoTask("a", OncePolicy()))
    orch.add_task(EchoTask("b", OncePolicy()))
    orch.add_task(EchoTask("c", AnyPolicy()))
    orch.add_edge("a", "c")
    orch.add_edge("b", "c")

    report = orch.run()

    assert report["tasks"]["c"]["run_count"] >= 1


def test_once_policy_does_not_rerun():
    orch = build_orchestrator()
    orch.add_task(EchoTask("src", OncePolicy()))
    orch.add_task(CountingTask("dst", OncePolicy()))
    orch.add_edge("src", "dst")

    orch.run()
    assert orch.tasks["dst"].meta.run_count == 1


def test_wave_damping_constant_result():
    orch = build_orchestrator()
    orch.add_task(ConstantTask("src", OncePolicy(), {"value": 42}))
    orch.add_task(ConstantTask("dst", OncePolicy(), {"value": 42}))
    orch.add_edge("src", "dst")

    orch.run()

    assert orch.tasks["src"].version == 1
    assert orch.tasks["dst"].version == 1


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------

def test_failure_marks_task_failed():
    orch = build_orchestrator()
    orch.add_task(FailingTask("src", OncePolicy()))
    orch.add_task(EchoTask("dst", AllPolicy()))
    orch.add_edge("src", "dst")

    report = orch.run()

    assert report["tasks"]["src"]["status"] == "FAILED"
    assert report["tasks"]["src"]["last_error"] is not None


def test_retry_policy_retries_up_to_max():
    orch = build_orchestrator()
    orch.add_task(FailingTask("src", OncePolicy(), RetryFailurePolicy()))

    report = orch.run()

    assert report["tasks"]["src"]["status"] == "FAILED"
    # Retries are tracked per instance; task-level run_count stays 0 because
    # the single instance never completes successfully.
    assert report["tasks"]["src"]["run_count"] == 0


def test_skip_policy_marks_done_without_state():
    orch = build_orchestrator()
    orch.add_task(FailingTask("src", OncePolicy(), SkipFailurePolicy()))
    orch.add_task(EchoTask("dst", AllPolicy()))
    orch.add_edge("src", "dst")

    report = orch.run()

    assert report["tasks"]["src"]["status"] == "DONE"
    # state now defaults to an empty list; skip leaves it untouched
    assert orch.tasks["src"].state == []
    assert orch.tasks["src"].version == 0
    assert report["tasks"]["dst"]["run_count"] == 0


def test_retry_counter_resets_on_success():
    class FlakyTask(Task):
        def __init__(self, task_id, activation_policy, failure_policy):
            super().__init__(task_id, activation_policy, failure_policy=failure_policy)
            self._calls = 0

        def run(self, inputs):
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("first call fails")
            return [{"ok": True}]

    orch = build_orchestrator()
    orch.add_task(FlakyTask("src", OncePolicy(), RetryFailurePolicy()))

    orch.run()

    assert orch.tasks["src"].status == TaskStatus.DONE
    assert orch.tasks["src"].state == [{"ok": True}]


def test_non_dict_result_treated_as_error():
    orch = build_orchestrator()
    orch.add_task(NonDictTask("src", OncePolicy()))

    report = orch.run()

    assert report["tasks"]["src"]["status"] == "FAILED"
    assert "expected dict" in report["tasks"]["src"]["last_error"]


def test_default_failure_policy_is_fail():
    orch = build_orchestrator()
    task = FailingTask("src", OncePolicy())
    assert isinstance(task.failure_policy, FailFailurePolicy)
    orch.add_task(task)

    report = orch.run()
    assert report["tasks"]["src"]["status"] == "FAILED"


# ---------------------------------------------------------------------------
# Tests: input isolation
# ---------------------------------------------------------------------------

def test_input_mutation_does_not_affect_graph_state():
    orch = build_orchestrator()
    orch.add_task(ConstantTask("src", OncePolicy(), {"nested": {"x": 1}}))
    orch.add_task(MutatingTask("dst", OncePolicy()))
    orch.add_edge("src", "dst")

    orch.run()

    src_state = orch.tasks["src"].state
    assert src_state
    # state is a list of dicts, index into the first snapshot
    assert "mutated" not in src_state[0]["nested"]


# ---------------------------------------------------------------------------
# Tests: self-re-evaluation (new inputs arrive during execution)
# ---------------------------------------------------------------------------

def test_self_reevaluation_with_concurrent_inputs():
    orch = Orchestrator(ThreadPoolExecutorBackend, max_workers=2)

    a1 = SlowTask("a1", OncePolicy(), delay=0.05)
    a2 = SlowTask("a2", OncePolicy(), delay=0.0)
    b = CountingTask("b", AnyPolicy())

    orch.add_task(a1)
    orch.add_task(a2)
    orch.add_task(b)
    orch.add_edge("a1", "b")
    orch.add_edge("a2", "b")

    def runner():
        orch.run()

    t = threading.Thread(target=runner)
    t.start()

    assert a1.started.wait(timeout=5.0), "a1 did not start"
    assert a2.started.wait(timeout=5.0), "a2 did not start"

    a1.release.set()
    a2.release.set()

    t.join(timeout=10.0)
    assert not t.is_alive(), "orchestrator did not finish"

    assert b.run_count_local >= 1
    assert orch.tasks["a1"].status == TaskStatus.DONE
    assert orch.tasks["a2"].status == TaskStatus.DONE


# ---------------------------------------------------------------------------
# Tests: executor shutdown
# ---------------------------------------------------------------------------

def test_thread_pool_executor_shuts_down():
    orch = Orchestrator(ThreadPoolExecutorBackend, max_workers=2)
    orch.add_task(EchoTask("src", OncePolicy()))
    orch.run()
    assert getattr(orch.executor, "pool", None) is not None


def test_inline_executor_shutdown_is_noop():
    orch = build_orchestrator()
    orch.add_task(EchoTask("src", OncePolicy()))
    orch.run()
    orch.executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Tests: completion event contract
# ---------------------------------------------------------------------------

def test_completion_event_success():
    # Per spec, payload on success is a list of dicts
    evt = CompletionEvent(
        instance_id="i1", task_id="t1", success=True, payload=[{"a": 1}]
    )
    assert evt.instance_id == "i1"
    assert evt.task_id == "t1"
    assert evt.success is True
    assert evt.payload == [{"a": 1}]


def test_completion_event_failure():
    err = RuntimeError("oops")
    evt = CompletionEvent(instance_id="i1", task_id="t1", success=False, payload=err)
    assert evt.success is False
    assert evt.payload is err


# ---------------------------------------------------------------------------
# Tests: report structure
# ---------------------------------------------------------------------------

def test_report_contains_tasks_and_edges():
    orch = build_orchestrator()
    orch.add_task(EchoTask("a", OncePolicy()))
    orch.add_task(EchoTask("b", AllPolicy()))
    edge_id = orch.add_edge("a", "b")

    report = orch.run()

    assert "tasks" in report
    assert "edges" in report
    assert "a" in report["tasks"]
    assert "b" in report["tasks"]
    assert edge_id in report["edges"]
    assert report["edges"][edge_id]["src"] == "a"
    assert report["edges"][edge_id]["dst"] == "b"
    assert report["edges"][edge_id]["consumed_version"] == 1


# ---------------------------------------------------------------------------
# Tests: exact policy
# ---------------------------------------------------------------------------

def test_exact_policy_requires_specific_edges():
    orch = build_orchestrator()
    orch.add_task(EchoTask("a", OncePolicy()))
    orch.add_task(EchoTask("b", OncePolicy()))
    orch.add_task(EchoTask("c", OncePolicy()))
    edge_a_id = orch.add_edge("a", "c")
    orch.add_edge("b", "c")

    orch.tasks["c"] = EchoTask(
        "c", ExactPolicy(ActivationPolicyConfig(required_edges=[edge_a_id]))
    )

    report = orch.run()

    assert report["tasks"]["c"]["run_count"] >= 1


def test_exact_policy_without_required_edges_never_activates():
    orch = build_orchestrator()
    orch.add_task(EchoTask("a", OncePolicy()))
    orch.add_task(EchoTask("b", ExactPolicy()))
    orch.add_edge("a", "b")

    report = orch.run()

    assert report["tasks"]["b"]["run_count"] == 0


# ---------------------------------------------------------------------------
# Tests: cycle handling
# ---------------------------------------------------------------------------

def test_self_loop_does_not_infinite_loop():
    orch = build_orchestrator()
    orch.add_task(CountingTask("t1", OncePolicy()))
    orch.add_edge("t1", "t1")

    orch.run()

    assert orch.tasks["t1"].meta.run_count == 0
    t1 = cast(CountingTask, orch.tasks["t1"])
    assert t1.run_count_local == 0


# ---------------------------------------------------------------------------
# Tests: Input Merge Policies
# ---------------------------------------------------------------------------

def test_cartesian_product_merge():
    policy = CartesianProductMergePolicy()
    incoming = {
        "A": [{"val": 1}, {"val": 2}],
        "B": [{"char": "x"}]
    }
    result = policy.merge(incoming)
    assert len(result) == 2
    assert {"val": 1, "char": "x"} in result
    assert {"val": 2, "char": "x"} in result


def test_cartesian_product_empty_source():
    policy = CartesianProductMergePolicy()
    incoming = {
        "A": [{"val": 1}],
        "B": []
    }
    result = policy.merge(incoming)
    assert len(result) == 0


def test_single_item_per_source_policy_valid():
    policy = SingleItemPerSourcePolicy()
    incoming = {
        "A": [{"val": 1}],
        "B": [{"char": "x"}]
    }
    result = policy.merge(incoming)
    assert len(result) == 1
    assert result[0] == {"val": 1, "char": "x"}


def test_single_item_per_source_policy_invalid():
    policy = SingleItemPerSourcePolicy()
    incoming = {
        "A": [{"val": 1}, {"val": 2}],
        "B": [{"char": "x"}]
    }
    with pytest.raises(ValueError):
        policy.merge(incoming)


def test_all_but_one_single_policy_valid():
    policy = AllButOneSinglePolicy()
    incoming = {
        "Stream": [{"id": 1}, {"id": 2}],
        "Context": [{"name": "test"}]
    }
    result = policy.merge(incoming)
    assert len(result) == 2


def test_all_but_one_single_policy_invalid():
    policy = AllButOneSinglePolicy()
    incoming = {
        "A": [{"val": 1}, {"val": 2}],
        "B": [{"char": "x"}, {"char": "y"}]
    }
    with pytest.raises(ValueError):
        policy.merge(incoming)


def test_flatten_merge_policy():
    policy = FlattenMergePolicy()
    incoming = {
        "A": [{"x": 1}, {"x": 2}],
        "B": [{"y": 3}]
    }
    result = policy.merge(incoming)
    assert len(result) == 3
    assert {"A": {"x": 1}} in result
    assert {"A": {"x": 2}} in result
    assert {"B": {"y": 3}} in result


def test_flatten_merge_policy_empty():
    policy = FlattenMergePolicy()
    incoming = {}
    result = policy.merge(incoming)
    assert result == []