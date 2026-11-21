from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional

# Type aliases for clarity and decoupling
EdgeId = str


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
# Failure policies
# ---------------------------------------------------------------------------

class FailureAction(Enum):
    """Action to take when a task fails."""
    RETRY = auto()
    SKIP = auto()
    FAIL = auto()


class FailurePolicy:
    """
    Base class for failure handling strategies.
    Pure logic, no side effects, no references to Task/Edge/graph objects.
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