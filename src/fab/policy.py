import itertools
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from .types import EdgeId
from .types import TaskId



# ---------------------------------------------------------------------------
# Activation policies
# ---------------------------------------------------------------------------

class TriggerType(Enum):
    """Event that caused the policy evaluation."""
    EDGE = auto()
    TIMER = auto()
    EXTERNAL = auto()


class PolicyNodeStatus(Enum):
    """Status of the node from the perspective of the policy context."""
    IDLE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass(frozen=True)
class PolicyContext:
    """
    Immutable context provided by the orchestrator for policy evaluation.
    Contains all necessary history and state without holding references to the graph.
    """
    fresh: Dict[EdgeId, bool]
    status: PolicyNodeStatus
    run_count: int
    last_error: Optional[Any]


@dataclass(frozen=True)
class ActivationPolicyConfig:
    """
    Immutable configuration parameters for an activation policy.
    """
    required_edges: Optional[List[EdgeId]] = None


class ActivationPolicy:
    """
    Base class for activation strategies.
    Pure logic, no side effects, no references to Task/Edge/graph objects.
    Answers the question: "Is it time to run?"
    """

    def __init__(self, config: Optional[ActivationPolicyConfig] = None):
        self.config = config or ActivationPolicyConfig()

    def should_activate(self, ctx: PolicyContext) -> bool:
        """Determines if the node should be activated based on the context."""
        raise NotImplementedError("Subclasses must implement should_activate")


class OncePolicy(ActivationPolicy):
    """
    ONCE policy: Activates if ALL inputs are fresh AND run_count == 0.
    Typically used as the default for DAG root nodes.
    """

    def should_activate(self, ctx: PolicyContext) -> bool:
        all_fresh = all(ctx.fresh.values()) if ctx.fresh else True
        return all_fresh and ctx.run_count == 0


class AllPolicy(ActivationPolicy):
    """
    ALL policy: Activates when all inputs are fresh.
    """

    def should_activate(self, ctx: PolicyContext) -> bool:
#        if not ctx.fresh:
#            raise ValueError("AllPolicy requires at least one input edge to be defined.")
        return all(ctx.fresh.values())


class AnyPolicy(ActivationPolicy):
    """
    ANY policy: Activates when at least one input is fresh.
    """

    def should_activate(self, ctx: PolicyContext) -> bool:
        return any(ctx.fresh.values())


class ExactPolicy(ActivationPolicy):
    """
    EXACT policy: Activates when all specifically required inputs are fresh.
    Requires 'required_edges' to be set in the ActivationPolicyConfig.
    """

    def should_activate(self, ctx: PolicyContext) -> bool:
        if not self.config.required_edges:
            return False
        
        # Check if all required edges are present and fresh
        return all(
            ctx.fresh.get(edge_id, False) 
            for edge_id in self.config.required_edges
        )


# ---------------------------------------------------------------------------
# Input Merge policies
# ---------------------------------------------------------------------------

class InputMergePolicy:
    """
    Base class for input merging strategies.
    Pure function, no side effects, no knowledge of graph/edges/versions.
    Answers the question: "How to combine incoming data packets into execution snapshots?"
    """

    def merge(self, incoming: Dict[TaskId, List[Dict[str, Any]]], priority_map: Optional[Dict[TaskId, int]] = None) -> List[Dict[str, Any]]:
        """
        Accepts a dictionary mapping source TaskId to a list of its fresh states.
        Returns a list of flat snapshot dictionaries, each representing one task instance execution.
        
        Args:
            incoming: Dictionary mapping source TaskId to list of state dicts from that source.
            priority_map: Optional dictionary mapping source TaskId to its priority.
                          Higher priority values win when keys conflict during merge.
                          If not provided, all sources are treated as equal priority (0).
        """
        raise NotImplementedError("Subclasses must implement merge")


class CartesianProductMergePolicy(InputMergePolicy):
    """
    Cartesian product of all incoming lists.
    If any source list is empty, returns an empty list (no execution).
    """

    def merge(self, incoming: Dict[TaskId, List[Dict[str, Any]]], priority_map: Optional[Dict[TaskId, int]] = None) -> List[Dict[str, Any]]:
        if not incoming:
            return [{}]
        
        if any(len(v) == 0 for v in incoming.values()):
            return []
        
        # Default priority_map if not provided
        if priority_map is None:
            priority_map = {k: 0 for k in incoming.keys()}
            
        keys = list(incoming.keys())
        combos = itertools.product(*(incoming[k] for k in keys))
        result = []
        for combo in combos:
            merged = {}
            # Merge with priority: for each key, keep value from highest priority source
            for d, src_id in zip(combo, keys):
                src_priority = priority_map.get(src_id, 0)
                for k, v in d.items():
                    if k not in merged:
                        merged[k] = (v, src_priority)
                    else:
                        existing_priority = merged[k][1]
                        if src_priority > existing_priority:
                            merged[k] = (v, src_priority)
            # Strip priority tuples to get final dict
            final_merged = {k: v[0] for k, v in merged.items()}
            result.append(final_merged)
        return result


class SingleItemPerSourcePolicy(InputMergePolicy):
    """
    Strict variant: each source must provide at most one state.
    Combines them into a single snapshot. Raises ValueError if any source has >1 item.
    """

    def merge(self, incoming: Dict[TaskId, List[Dict[str, Any]]], priority_map: Optional[Dict[TaskId, int]] = None) -> List[Dict[str, Any]]:
        for k, v in incoming.items():
            if len(v) > 1:
                raise ValueError(f"Source {k} provided {len(v)} items, expected <= 1")
        
        # Reuse cartesian logic, which is safe here since max length is 1
        return CartesianProductMergePolicy().merge(incoming, priority_map)


class AllButOneSinglePolicy(InputMergePolicy):
    """
    Relaxed variant: at most one source can provide multiple states (the "main axis").
    All other sources must provide <= 1 state (static context).
    Raises ValueError if more than one source has >1 item.
    """

    def merge(self, incoming: Dict[TaskId, List[Dict[str, Any]]], priority_map: Optional[Dict[TaskId, int]] = None) -> List[Dict[str, Any]]:
        multi_sources = [k for k, v in incoming.items() if len(v) > 1]
        if len(multi_sources) > 1:
            raise ValueError(f"Multiple sources provided >1 item: {multi_sources}")
            
        return CartesianProductMergePolicy().merge(incoming, priority_map)


class FlattenMergePolicy(InputMergePolicy):
    """
    Flattens all incoming states from all sources into a single list of snapshots.
    Each snapshot contains data from exactly one source state.
    
    Example:
        Input:  {"A": [{"x": 1}, {"x": 2}], "B": [{"y": 3}]}
        Output: [{"A": {"x": 1}}, {"A": {"x": 2}}, {"B": {"y": 3}}]
        
    This is useful for fan-in scenarios where multiple parallel branches 
    produce independent results that should be processed sequentially by the next task.
    """

    def merge(self, incoming: Dict[TaskId, List[Dict[str, Any]]], priority_map: Optional[Dict[TaskId, int]] = None) -> List[Dict[str, Any]]:
        result = []
        for src_id, states in incoming.items():
            for state in states:
                result.append({src_id: state})
        return result

class SoftCartesianMergePolicy(InputMergePolicy):
    """
    Cartesian product of all incoming lists.
    If any source list is empty, it is treated as a single empty dict [{}] instead of
    causing the entire result to be empty. This means empty lists act as neutral elements
    (like '1' in multiplication) rather than zeroing elements (like '0').
    """

    def merge(self, incoming: Dict[TaskId, List[Dict[str, Any]]], priority_map: Optional[Dict[TaskId, int]] = None) -> List[Dict[str, Any]]:
        if not incoming:
            return [{}]

        # Treat empty lists as [{}] — neutral element for cartesian product
        normalized = {
            src_id: (states if states else [{}])
            for src_id, states in incoming.items()
        }

        if priority_map is None:
            priority_map = {k: 0 for k in normalized.keys()}

        keys = list(normalized.keys())
        combos = itertools.product(*(normalized[k] for k in keys))
        result = []
        for combo in combos:
            merged = {}
            for d, src_id in zip(combo, keys):
                src_priority = priority_map.get(src_id, 0)
                for k, v in d.items():
                    if k not in merged:
                        merged[k] = (v, src_priority)
                    else:
                        existing_priority = merged[k][1]
                        if src_priority > existing_priority:
                            merged[k] = (v, src_priority)
            final_merged = {k: v[0] for k, v in merged.items()}
            result.append(final_merged)
        return result

# ---------------------------------------------------------------------------
# Failure policies
# ---------------------------------------------------------------------------

class FailureAction(Enum):
    """Action to take when a task instance fails."""
    RETRY = auto()
    SKIP = auto()
    FAIL = auto()


class FailurePolicy:
    """
    Base class for failure handling strategies.
    Pure logic, no side effects.
    Answers the question: "What to do after a failure?"
    """

    def on_failure(self, ctx: PolicyContext) -> FailureAction:
        """Determines the next step after a failure. Defaults to FAIL."""
        return FailureAction.FAIL


class RetryFailurePolicy(FailurePolicy):
    """Always requests RETRY on failure."""

    def on_failure(self, ctx: PolicyContext) -> FailureAction:
        return FailureAction.RETRY


class SkipFailurePolicy(FailurePolicy):
    """Always requests SKIP on failure."""

    def on_failure(self, ctx: PolicyContext) -> FailureAction:
        return FailureAction.SKIP


class FailFailurePolicy(FailurePolicy):
    """Always requests FAIL on failure (default behavior)."""

    def on_failure(self, ctx: PolicyContext) -> FailureAction:
        return FailureAction.FAIL