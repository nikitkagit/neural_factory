import time
import queue
import copy
import uuid
from typing import Dict, Set, List, Any, Type

from .task import Task, TaskId, TaskStatus
from .edge import Edge, EdgeId
from .policy import PolicyContext, PolicyNodeStatus, TriggerType, FailureAction
from .executor import Executor, Inbox, CompletionEvent


class Orchestrator:
    """
    Owner of the graph and the sole writer of mutable state.
    Implements the event loop: dispatch READY -> wait for events -> commit -> evaluate policies -> repeat.
    Works with any Executor backend, ensuring latest-wins semantics, wave damping, and quiescence detection.
    """
    def __init__(self, executor_cls: Type[Executor], **executor_kwargs):
        # Inbox is created by the orchestrator and passed to the executor
        self.inbox: Inbox = queue.Queue()
        self.executor: Executor = executor_cls(self.inbox, **executor_kwargs)
        
        # Graph registries
        self.tasks: Dict[TaskId, Task] = {}
        self.edges: Dict[EdgeId, Edge] = {}
        self.out_edges: Dict[TaskId, List[EdgeId]] = {}
        self.in_edges: Dict[TaskId, List[EdgeId]] = {}
        
        # Execution state
        self.ready_queue: Set[TaskId] = set()
        self.running: Set[TaskId] = set()
        self.retry_attempts: Dict[TaskId, int] = {}
        
        # Configuration
        self.max_retries = 3

    def add_task(self, task: Task) -> None:
        """Add a task to the graph. Validates uniqueness."""
        if task.task_id in self.tasks:
            raise ValueError(f"Task {task.task_id} already exists")
        self.tasks[task.task_id] = task
        self.out_edges[task.task_id] = []
        self.in_edges[task.task_id] = []

    def add_edge(self, src_id: TaskId, dst_id: TaskId) -> EdgeId:
        """Add an edge to the graph. Validates existence of endpoints and uniqueness."""
        if src_id not in self.tasks:
            raise ValueError(f"Source task {src_id} does not exist")
        if dst_id not in self.tasks:
            raise ValueError(f"Destination task {dst_id} does not exist")
        
        # Use UUID to prevent accidental collisions from task names containing '->'
        edge_id = str(uuid.uuid4())
        if edge_id in self.edges:
            raise ValueError(f"Edge {edge_id} already exists")
            
        edge = Edge(edge_id=edge_id, src=src_id, dst=dst_id)
        self.edges[edge_id] = edge
        self.out_edges[src_id].append(edge_id)
        self.in_edges[dst_id].append(edge_id)
        return edge_id

    def run(self) -> Dict[str, Any]:
        """Main event loop. Returns a final report upon quiescence."""
        try:
            self._bootstrap()
            self._dispatch_wave()
            
            # Wait for events as long as there are running tasks
            while self.running:
                event = self.inbox.get()
                self._handle_event(event)
                self._dispatch_wave()
                
            return self._report()
        finally:
            # Ensure the execution pool is always shut down gracefully
            self.executor.shutdown(wait=True)

    def _bootstrap(self) -> None:
        """Initial policy evaluation for all tasks to seed the READY queue."""
        for task_id in self.tasks:
            ctx = self._build_policy_context(task_id, TriggerType.EXTERNAL)
            if self.tasks[task_id].activation_policy.should_activate(ctx):
                self.ready_queue.add(task_id)

    def _build_policy_context(self, task_id: TaskId, trigger: TriggerType) -> PolicyContext:
        """Constructs an immutable PolicyContext for a given task."""
        task = self.tasks[task_id]
        
        fresh = {}
        for edge_id in self.in_edges[task_id]:
            edge = self.edges[edge_id]
            src_task = self.tasks[edge.src]
            # Freshness is strictly derived from version comparison
            is_fresh = src_task.version > edge.consumed_version
            fresh[edge_id] = is_fresh
            
        status_map = {
            TaskStatus.IDLE: PolicyNodeStatus.IDLE,
            TaskStatus.DONE: PolicyNodeStatus.DONE,
            TaskStatus.FAILED: PolicyNodeStatus.FAILED,
            TaskStatus.RUNNING: PolicyNodeStatus.IDLE 
        }
        
        return PolicyContext(
            fresh=fresh,
            status=status_map.get(task.status, PolicyNodeStatus.IDLE),
            run_count=getattr(task.meta, 'run_count', 0),
            last_error=task.meta.last_error
        )

    def _dispatch_wave(self) -> None:
        """Dispatches all tasks in the READY queue."""
        while self.ready_queue:
            task_id = self.ready_queue.pop()
            task = self.tasks[task_id]
            
            # Guard: ensure no more than one RUNNING per task
            if task.status == TaskStatus.RUNNING:
                continue
                
            # Gather inputs at the moment of launch (latest-wins)
            inputs = {}
            for edge_id in self.in_edges[task_id]:
                edge = self.edges[edge_id]
                src_task = self.tasks[edge.src]
                if src_task.state is not None:
                    # Deep copy inputs to prevent tasks from mutating shared state
                    inputs[edge.src] = copy.deepcopy(src_task.state)
                    # Mark exactly what is read as consumed
                    edge.mark_consumed(src_task.version)
            
            task.status = TaskStatus.RUNNING
            self.running.add(task_id)
            
            # Pass the run function, not the task object itself
            self.executor.submit(task_id, task.run, inputs)

    def _handle_event(self, event: CompletionEvent) -> None:
        """Processes a completion event from the executor."""
        task_id = event.task_id
        task = self.tasks[task_id]
        self.running.remove(task_id)
        
        if not event.success:
            self._handle_error(task, event.payload)
        else:
            self._commit_result(task, event.payload)

    def _commit_result(self, task: Task, result: Any) -> None:
        """Commits the result, updates version, and propagates freshness."""
        # Validate that the result is strictly a dictionary
        if not isinstance(result, dict):
            self._handle_error(task, TypeError(f"Task {task.task_id} returned {type(result).__name__}, expected dict"))
            return

        # Wave damping: version only grows if the result actually changed
        if result != task.state:
            task.state = result
            task.version += 1
            version_grew = True
        else:
            version_grew = False
            
        task.status = TaskStatus.DONE
        task.meta.timestamp = time.time()
        task.meta.run_count = getattr(task.meta, 'run_count', 0) + 1
        task.meta.last_error = None
        
        # Reset retry counter on successful completion
        self.retry_attempts[task.task_id] = 0
        
        if version_grew:
            # Evaluate activation policies of downstream consumers
            for edge_id in self.out_edges[task.task_id]:
                edge = self.edges[edge_id]
                dst_id = edge.dst
                dst_task = self.tasks[dst_id]
                
                # Policies are evaluated only for tasks not in running and not already in ready
                if dst_task.status == TaskStatus.RUNNING or dst_id in self.ready_queue:
                    continue
                    
                ctx = self._build_policy_context(dst_id, TriggerType.EDGE)
                if dst_task.activation_policy.should_activate(ctx):
                    self.ready_queue.add(dst_id)
                    
        # Self-re-evaluation: check if the task itself needs to run again
        # because new inputs might have arrived while it was running.
        ctx = self._build_policy_context(task.task_id, TriggerType.EDGE)
        if task.activation_policy.should_activate(ctx):
            self.ready_queue.add(task.task_id)

    def _handle_error(self, task: Task, error: Exception) -> None:
        """Handles execution failures based on the task's failure policy."""
        task.status = TaskStatus.FAILED
        task.meta.last_error = error
        
        ctx = self._build_policy_context(task.task_id, TriggerType.EXTERNAL)
        action = task.failure_policy.on_failure(ctx)
        
        if action == FailureAction.RETRY:
            attempts = self.retry_attempts.get(task.task_id, 0) + 1
            self.retry_attempts[task.task_id] = attempts
            
            if attempts > self.max_retries:
                task.status = TaskStatus.FAILED
            else:
                # Reset to IDLE to allow re-dispatch
                task.status = TaskStatus.IDLE
                self.ready_queue.add(task.task_id)
                
        elif action == FailureAction.SKIP:
            # Mark as DONE without changing state/version and without propagation
            task.status = TaskStatus.DONE
            
        elif action == FailureAction.FAIL:
            # Branch dies out naturally as consumers won't see fresh inputs
            task.status = TaskStatus.FAILED

    def _report(self) -> Dict[str, Any]:
        """Generates a final snapshot report of the graph's state."""
        report = {
            "tasks": {},
            "edges": {}
        }
        for task_id, task in self.tasks.items():
            report["tasks"][task_id] = {
                "status": task.status.name,
                "run_count": getattr(task.meta, 'run_count', 0),
                "version": task.version,
                "last_error": str(task.meta.last_error) if task.meta.last_error else None
            }
        for edge_id, edge in self.edges.items():
            report["edges"][edge_id] = {
                "src": edge.src,
                "dst": edge.dst,
                "consumed_version": edge.consumed_version
            }
        return report