from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional

from .policy import ActivationPolicy, FailurePolicy, FailFailurePolicy

# Type alias for task identification
TaskId = str


class TaskStatus(Enum):
    """Execution status of the task."""
    IDLE = auto()
    RUNNING = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class TaskMeta:
    """Metadata associated with the task's execution history."""
    timestamp: Optional[float] = None
    last_error: Optional[Any] = None
    run_count: int = 0


class Task:
    """
    Represents a computation node ("input -> output") and stores the latest result.
    It does not know about its neighbors or the graph topology.
    
    Invariants:
    - The orchestrator is the sole writer to state, version, status, and meta.
    - During `run()`, the task must not mutate its own internal state.
    - DONE is not a terminal state; re-runs are possible via activation policy.
    """

    def __init__(
        self,
        task_id: TaskId,
        activation_policy: ActivationPolicy,
        failure_policy: Optional[FailurePolicy] = None,
    ):
        self.task_id: TaskId = task_id
        self.activation_policy: ActivationPolicy = activation_policy
        self.failure_policy: FailurePolicy = failure_policy or FailFailurePolicy()
        
        # Fields managed by the orchestrator
        self.state: Optional[Dict] = None
        self.version: int = 0
        self.status: TaskStatus = TaskStatus.IDLE
        self.meta: TaskMeta = TaskMeta()

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes the output based on a frozen snapshot of inputs.
        Subclasses must implement this method.
        
        Invariant: Must not write to self during execution.
        The orchestrator is responsible for committing the result, 
        updating the version, and changing the status.
        """
        raise NotImplementedError("Subclasses must implement the run() method")

    # --- Observation Getters ---
    # These are strictly for monitoring and result collection, 
    # not for data passing between tasks.

    @property
    def get_state(self) -> Optional[Dict]:
        return self.state

    @property
    def get_version(self) -> int:
        return self.version

    @property
    def get_status(self) -> TaskStatus:
        return self.status