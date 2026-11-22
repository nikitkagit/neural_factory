import logging
import time
import queue
import copy
import uuid
from collections import defaultdict
from typing import Dict, Set, List, Any, Type, Optional, Tuple

from .types import TaskId, EdgeId, InstanceId
from .task import Task, TaskId, TaskStatus
from .edge import Edge
from .policy import PolicyContext, PolicyNodeStatus, TriggerType, FailureAction
from .executor import Executor, Inbox, CompletionEvent


logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Owner of the graph and the sole writer of mutable state.
    Implements the event loop: dispatch READY -> wait for events -> commit -> evaluate policies -> repeat.
    Supports multiple concurrent instances per task driven by InputMergePolicy.
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
        self.running_instances: Dict[TaskId, Set[InstanceId]] = defaultdict(set)
        self.instance_to_task: Dict[InstanceId, TaskId] = {}
        self.instance_snapshots: Dict[InstanceId, Dict[str, Any]] = {}
        self.instance_retries: Dict[InstanceId, int] = {}
        self.instance_results: Dict[TaskId, List[List[Dict[str, Any]]]] = defaultdict(list)

        # Configuration
        self.max_retries = 3
        self.failed = False

        logger.debug("Orchestrator initialized with %d tasks, %d edges", len(self.tasks), len(self.edges))

    def add_task(self, task: Task) -> None:
        """Add a task to the graph. Validates uniqueness."""
        if task.task_id in self.tasks:
            raise ValueError(f"Task {task.task_id} already exists")
        self.tasks[task.task_id] = task
        self.out_edges[task.task_id] = []
        self.in_edges[task.task_id] = []
        logger.debug("Task %s added to graph", task.task_id)

    def add_edge(
        self,
        src_id: TaskId,
        dst_id: TaskId,
        aliases: Optional[Dict[str, str]] = None,
        groups: Optional[List[str]] = None
    ) -> EdgeId:
        """Add an edge to the graph. Validates existence of endpoints and uniqueness."""
        if src_id not in self.tasks:
            raise ValueError(f"Source task {src_id} does not exist")
        if dst_id not in self.tasks:
            raise ValueError(f"Destination task {dst_id} does not exist")

        edge_id = str(uuid.uuid4())
        if edge_id in self.edges:
            raise ValueError(f"Edge {edge_id} already exists")

        edge = Edge(
            edge_id=edge_id,
            src=src_id,
            dst=dst_id,
            aliases=aliases or {},
            groups=groups or []
        )
        self.edges[edge_id] = edge
        self.out_edges[src_id].append(edge_id)
        self.in_edges[dst_id].append(edge_id)
        logger.debug("Edge %s added: %s -> %s (groups=%s)", edge_id, src_id, dst_id, edge.groups)
        return edge_id

    def run(self) -> Dict[str, Any]:
        """Main event loop. Returns a final report upon quiescence."""
        try:
            self._bootstrap()
            self._dispatch_wave()
            logger.debug("Event loop started")

            while any(len(instances) > 0 for instances in self.running_instances.values()):
                event = self.inbox.get()
                logger.debug("Received completion event for instance %s, task %s, success=%s", event.instance_id, event.task_id, event.success)
                self._handle_event(event)
                if self.failed:
                    logger.debug("Task failure detected, breaking event loop")
                    break
                self._dispatch_wave()

            return self._report()
        finally:
            logger.debug("Shutting down executor")
            # Ensure the execution pool is always shut down gracefully
            self.executor.shutdown(wait=True)

    def _bootstrap(self) -> None:
        """Initial policy evaluation for all tasks to seed the first wave."""
        for task_id in self.tasks:
            ctx = self._build_policy_context(task_id, TriggerType.EXTERNAL)
            logger.debug("Bootstrap: evaluating task %s, fresh=%s, status=%s", task_id, ctx.fresh, ctx.status)
            if self.tasks[task_id].activation_policy.should_activate(ctx):
                pass  # _dispatch_wave will handle the actual gathering and submission

    def _extract_groups(self, state: List[Dict[str, Any]]) -> Optional[List[str]]:
        """
        Extract the list of groups from the source task's state if present.
        Returns None if no __groups key is found in any dict.
        """
        for item in state:
            if isinstance(item, dict) and '__groups' in item:
                groups_val = item['__groups']
                if isinstance(groups_val, list):
                    return groups_val
        return None

    def _build_policy_context(self, task_id: TaskId, trigger: TriggerType) -> PolicyContext:
        """Constructs an immutable PolicyContext for a given task."""
        task = self.tasks[task_id]

        fresh = {}
        for edge_id in self.in_edges[task_id]:
            edge = self.edges[edge_id]
            src_task = self.tasks[edge.src]
            is_fresh = src_task.version > edge.consumed_version

            # Apply groups-based filtering for freshness
            if is_fresh and edge.groups:
                src_groups = self._extract_groups(src_task.state)
                if src_groups is not None:
                    # Intersection required
                    if not set(edge.groups) & set(src_groups):
                        is_fresh = False

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
        """Dispatches new instances for tasks whose activation policy is satisfied."""
        if self.failed:
            return

        for task_id, task in self.tasks.items():
            # Only evaluate tasks that are not currently running or permanently failed
            if task.status in (TaskStatus.RUNNING, TaskStatus.FAILED):
                continue

            ctx = self._build_policy_context(task_id, TriggerType.EDGE)
            if not task.activation_policy.should_activate(ctx):
                continue
            logger.debug("Task %s activated, gathering inputs", task_id)

            # Gather incoming fresh states grouped by source TaskId
            incoming: Dict[TaskId, List[Dict[str, Any]]] = defaultdict(list)
            edges_to_consume: List[Tuple[Edge, int]] = []
            added_src_versions: Set[Tuple[TaskId, int]] = set()

            logger.debug("INCOMING for %s: %s", task_id, incoming)
            logger.debug("Edge states: %s", {eid: (self.edges[eid].src, self.edges[eid].consumed_version) for eid in self.in_edges[task_id]})
            logger.debug("Src task versions: %s", {eid: self.tasks[self.edges[eid].src].version for eid in self.in_edges[task_id]})

            for edge_id in self.in_edges[task_id]:
                edge = self.edges[edge_id]
                src_task = self.tasks[edge.src]

                if src_task.version > edge.consumed_version and src_task.state:
                    # Apply groups-based filtering for incoming data
                    if edge.groups:
                        src_groups = self._extract_groups(src_task.state)
                        if src_groups is not None:
                            if not set(edge.groups) & set(src_groups):
                                # No intersection, skip this edge's data
                                continue

                    key = (edge.src, src_task.version)
                    if key not in added_src_versions:
                        # Apply field aliases before merge, per spec
                        for state_item in src_task.state:
                            aliased_item: Dict[str, Any] = {}
                            for k, v in state_item.items():
                                new_key = edge.aliases.get(k, k)
                                aliased_item[new_key] = copy.deepcopy(v)
                            incoming[edge.src].append(aliased_item)
                        added_src_versions.add(key)
                    edges_to_consume.append((edge, src_task.version))

            # Merge into snapshots via the task's merge policy
            try:
                snapshots = task.merge_policy.merge(incoming)
            except ValueError as e:
                task.status = TaskStatus.FAILED
                task.meta.last_error = e
                logger.error("Task %s merge policy failed: %s", task_id, e)
                continue

            if not snapshots:
                logger.debug("Task %s: no snapshots to run (empty merge result)", task_id)
                continue  # No instances to run (e.g., empty cartesian product)

            # Mark consumption BEFORE submitting to guarantee no data loss
            for edge, version in edges_to_consume:
                edge.mark_consumed(version)

            task.status = TaskStatus.RUNNING
            logger.debug("Dispatching %d instance(s) for task %s", len(snapshots), task_id)

            for snapshot in snapshots:
                instance_id = f"{task_id}_{uuid.uuid4().hex[:8]}"
                self.instance_to_task[instance_id] = task_id
                self.instance_snapshots[instance_id] = snapshot
                self.instance_retries[instance_id] = 0
                self.running_instances[task_id].add(instance_id)

                logger.debug("Submitting instance %s for task %s", instance_id, task_id)
                self.executor.submit(instance_id, task_id, task.run, snapshot)

    def _handle_event(self, event: CompletionEvent) -> None:
        """Processes a completion event from the executor."""
        instance_id = event.instance_id
        task_id = event.task_id
        task = self.tasks[task_id]

        # Ignore events for instances that were already cancelled/removed
        if instance_id not in self.running_instances[task_id]:
            logger.debug("Ignoring stale event for instance %s", instance_id)
            return

        if not event.success:
            # Save the exception to task metadata so it appears in the report
            task.meta.last_error = event.payload

            action = task.failure_policy.on_failure(self._build_policy_context(task_id, TriggerType.EXTERNAL))

            if action == FailureAction.RETRY:
                self.instance_retries[instance_id] = self.instance_retries.get(instance_id, 0) + 1
                if self.instance_retries[instance_id] > self.max_retries:
                    self._fail_task(task_id)
                else:
                    logger.debug("Retrying instance %s for task %s (attempt %d)", instance_id, task_id, self.instance_retries[instance_id])
                    # Resubmit only this instance with the exact same snapshot
                    self.executor.submit(instance_id, task_id, task.run, self.instance_snapshots[instance_id])
                return  # Do not remove from running yet

            elif action == FailureAction.SKIP:
                logger.debug("Skipping failed instance %s for task %s", instance_id, task_id)
                # Instance is considered done without result, doesn't affect final state
                pass

            elif action == FailureAction.FAIL:
                logger.debug("Failing task %s due to instance %s failure", task_id, instance_id)
                self._fail_task(task_id)
                return
        else:
            # Success: payload must be List[Dict[str, Any]]
            payload = event.payload
            if not isinstance(payload, list):
                task.meta.last_error = TypeError(
                    f"Task {task_id} returned {type(payload).__name__}, expected list"
                )
                logger.error("Task %s returned non-list payload: %s", task_id, type(payload).__name__)
                self._fail_task(task_id)
                return
            for item in payload:
                if not isinstance(item, dict):
                    task.meta.last_error = TypeError(
                        f"Task {task_id} returned list containing {type(item).__name__}, expected dict"
                    )
                    logger.error("Task %s returned list with non-dict element: %s", task_id, type(item).__name__)
                    self._fail_task(task_id)
                    return

            logger.debug("Instance %s completed successfully for task %s", instance_id, task_id)
            # Accumulate result; commit happens only after all instances complete
            self.instance_results[task_id].append(payload)

        # Remove instance from running state
        self.running_instances[task_id].discard(instance_id)
        self.instance_snapshots.pop(instance_id, None)
        self.instance_retries.pop(instance_id, None)

        # Check if all instances for this task have finished
        if not self.running_instances[task_id]:
            logger.debug("All instances of task %s completed, finalizing", task_id)
            self._finalize_task(task_id)

    def _fail_task(self, task_id: TaskId) -> None:
        """
        Marks the task as FAILED, cancels all pending instances across the entire graph,
        and signals the orchestrator to exit the event loop with an error.
        """
        task = self.tasks[task_id]
        logger.error("Failing task %s (status=%s, error=%s)", task_id, task.status, task.meta.last_error)
        task.status = TaskStatus.FAILED

        # Clear running instances across all tasks to stop the entire graph
        for tid in list(self.running_instances.keys()):
            for inst_id in list(self.running_instances[tid]):
                self.instance_snapshots.pop(inst_id, None)
                self.instance_retries.pop(inst_id, None)
            self.running_instances[tid].clear()

        self.instance_results[task_id] = []
        self.failed = True

    def _finalize_task(self, task_id: TaskId) -> None:
        """Commits the final state after all instances of a task have completed."""
        task = self.tasks[task_id]

        if task.status == TaskStatus.FAILED:
            logger.debug("Task %s already failed, skipping finalization", task_id)
            return

        # Default aggregation: use the last completed instance's result list
        if self.instance_results[task_id]:
            final_state = self.instance_results[task_id][-1]
        else:
            final_state = task.state  # Fallback if all instances were skipped

        if final_state != task.state:
            task.state = final_state
            task.version += 1

        task.status = TaskStatus.DONE
        logger.debug("Task %s finalized: version=%d, state items=%d", task_id, task.version, len(task.state))
        task.meta.timestamp = time.time()
        task.meta.run_count += 1  # Incremented once per task, not per instance
        task.meta.last_error = None

        # Clear results for potential future runs
        self.instance_results[task_id] = []

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
                "consumed_version": edge.consumed_version,
                "aliases": edge.aliases,
                "groups": edge.groups
            }
        return report