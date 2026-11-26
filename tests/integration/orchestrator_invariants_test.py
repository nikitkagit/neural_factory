"""
Invariant tests for Orchestrator that currently FAIL on the shipped code.

Invariant under test:
    "After the event loop terminates, no node whose input edges have become
     fresh may remain unactivated."

These tests are intentionally written to exercise the real Orchestrator
(no mocks, no monkeypatching) and to fail on the current implementation,
where `Orchestrator.run()` checks `while any(running_instances...)` BEFORE
`_dispatch_wave()` and therefore may exit the loop while a downstream node
still has a fresh, unconsumed input edge.
"""

from typing import Any, Dict, List

import pytest

from src.fab.task import Task
from src.fab.policy import (
    OncePolicy,
    AllPolicy,
    AnyPolicy,
)
from src.fab.executor import InlineExecutor
from src.fab.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Helper tasks (intentionally tiny and side-effect free)
# ---------------------------------------------------------------------------

class EmitTask(Task):
    """Emits a fixed list of dicts. Records how many times it actually ran."""

    def __init__(self, task_id: str, output: List[Dict[str, Any]], activation_policy):
        super().__init__(task_id, activation_policy)
        self._output = output
        self.compute_calls = 0

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.compute_calls += 1
        return [dict(item) for item in self._output]


class PassthroughTask(Task):
    """Emits a fixed result. Records how many times it actually ran."""

    def __init__(self, task_id: str, activation_policy):
        super().__init__(task_id, activation_policy)
        self.compute_calls = 0

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.compute_calls += 1
        return [{"ran": True}]


# ---------------------------------------------------------------------------
# Invariant A: cascading activation through a chain must reach the tail.
# ---------------------------------------------------------------------------

def test_chain_of_three_reaches_tail_node():
    """
    A -> B -> C, all with OncePolicy.

    Contract: after Orchestrator.run() returns, every reachable node whose
    input edge has become fresh must have run exactly once.

    On the current implementation this test FAILS: `orchestrator.run()`
    exits the event loop after C's predecessor has committed, leaving C
    with a fresh input edge but status IDLE and run_count == 0.
    """
    orch = Orchestrator(InlineExecutor)

    a = EmitTask("a", [{}], activation_policy=OncePolicy())
    b = EmitTask("b", [{}], activation_policy=AnyPolicy())
    c = PassthroughTask("c", activation_policy=AllPolicy())

    orch.add_task(a)
    orch.add_task(b)
    orch.add_task(c)
    orch.add_edge("a", "b")
    orch.add_edge("b", "c")

    report = orch.run()

    # The invariant: nothing is left with an unconsumed fresh input edge.
    assert report["tasks"]["a"]["status"] == "DONE"
    assert report["tasks"]["b"]["status"] == "DONE"
    assert report["tasks"]["c"]["status"] == "DONE", (
        "Tail node 'c' was never activated although its input edge 'b -> c' "
        "became fresh. Orchestrator.run() terminated the loop before "
        "dispatching the final wave."
    )
    assert report["tasks"]["c"]["run_count"] == 1
    assert c.compute_calls == 1


# ---------------------------------------------------------------------------
# Invariant B: two sources with AnyPolicy must both be observed.
# ---------------------------------------------------------------------------

def test_any_policy_node_runs_after_slow_and_fast_sources():
    """
    fast -> sink, slow -> sink (sink: AnyPolicy).

    Contract: after Orchestrator.run(), `sink.run` must have been called at
    least once, and every fresh input edge must have been accounted for.

    On the current implementation this test FAILS when the sink finishes
    its first run on the 'fast' branch and the loop then exits before
    committing the 'slow' branch's fresh edge — leaving the sink with an
    unconsumed fresh input while `sink.run_count` stays at 1 instead of 2.
    """
    orch = Orchestrator(InlineExecutor)

    fast = EmitTask("fast", [{"branch": "fast"}], activation_policy=OncePolicy())
    slow = EmitTask("slow", [{"branch": "slow"}], activation_policy=OncePolicy())
    sink = PassthroughTask("sink", activation_policy=AnyPolicy())

    orch.add_task(fast)
    orch.add_task(slow)
    orch.add_task(sink)
    orch.add_edge("fast", "sink")
    orch.add_edge("slow", "sink")

    report = orch.run()

    assert report["tasks"]["fast"]["status"] == "DONE"
    assert report["tasks"]["slow"]["status"] == "DONE"
    assert report["tasks"]["sink"]["status"] == "DONE"
    # Both fresh edges must eventually be consumed by a second activation.
    assert report["tasks"]["sink"]["run_count"] >= 2, (
        "Sink under AnyPolicy was activated only once; the second fresh "
        "input edge was never dispatched because the event loop had already "
        "terminated. One activation of a fresh edge was silently dropped."
    )


# ---------------------------------------------------------------------------
# Invariant C: no fresh edge is left unconsumed when the loop exits.
# ---------------------------------------------------------------------------

def test_no_unconsumed_fresh_edge_at_quiescence():
    """
    Structural invariant check, independent of node counts.

    Contract: once Orchestrator.run() returns, for EVERY edge e:
        state[e.src].version <= e.consumed_version
    unless src's state is empty (nothing to consume).

    This test inspects the produced report directly and does not rely on
    any particular activation policy. On the current implementation it
    FAILS when a downstream node is skipped due to the early loop exit:
    the upstream edge's `consumed_version` remains behind.
    """
    orch = Orchestrator(InlineExecutor)

    a = EmitTask("a", [{"v": 1}], activation_policy=AnyPolicy())
    b = EmitTask("b", [{"v": 2}], activation_policy=AllPolicy())

    orch.add_task(a)
    orch.add_task(b)
    orch.add_edge("a", "b")

    report = orch.run()

    # Reconstruct the invariant from the report without touching internals.
    versions = {tid: report["tasks"][tid]["version"] for tid in report["tasks"]}
    for edge_id, edge_report in report["edges"].items():
        src = edge_report["src"]
        consumed = edge_report["consumed_version"]
        src_version = versions[src]
        if src_version == 0:
            # Nothing was produced by the source; nothing to consume.
            continue
        assert consumed >= src_version, (
            f"Edge {edge_id} ({src} -> {edge_report['dst']}) still has "
            f"unconsumed fresh data: src_version={src_version}, "
            f"consumed_version={consumed}. Orchestrator.run() returned while "
            f"a fresh input edge was left unaccounted for."
        )