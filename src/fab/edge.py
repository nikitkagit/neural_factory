from dataclasses import dataclass, field
from typing import Dict

from .task import TaskId

# Type alias for edge identification
EdgeId = str


@dataclass
class Edge:
    """
    Represents a delivery channel from the output of one task to the input of another.
    Maintained and updated exclusively by the orchestrator.
    
    Invariants:
    - The orchestrator is the sole writer to this object.
    - aliases maps source field names to destination field names.
    """
    
    edge_id: EdgeId
    src: TaskId
    dst: TaskId
    
    # The version of the src state that was consumed by dst during its last run
    consumed_version: int = 0
    
    # Field name remapping: {src_field_name: dst_field_name}
    aliases: Dict[str, str] = field(default_factory=dict)

    def mark_consumed(self, version: int) -> None:
        """
        Marks the edge as consumed up to the given version.
        Called by the orchestrator when the destination task (dst) runs.
        """
        self.consumed_version = version