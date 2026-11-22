import queue
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Union
from concurrent.futures import ThreadPoolExecutor

from .types import TaskId, InstanceId

Inbox = queue.Queue


@dataclass(frozen=True)
class CompletionEvent:
    """
    Explicit completion event to avoid ambiguity between success and error.
    Ensures the orchestrator always receives exactly one clear signal per submission.

    On success: payload is List[Dict[str, Any]] returned by Task.run.
    On failure: payload is the raised Exception.
    """
    instance_id: InstanceId
    task_id: TaskId
    success: bool
    payload: Union[List[Dict[str, Any]], Exception]


class Executor:
    """
    Base class for task execution abstraction.
    Takes a ready-to-run task instance with an input snapshot, runs the computation,
    and reports the result to the orchestrator via the inbox.

    The executor does not do any internal parallelization of the list returned by run_fn;
    it simply invokes run_fn once and passes the result (or exception) back.
    """
    def __init__(self, inbox: Inbox):
        self.inbox = inbox

    def submit(self, instance_id: InstanceId, task_id: TaskId, run_fn: Callable[[Dict], List[Dict]], inputs: Dict) -> None:
        """Non-blocking: put the task instance in execution and return control."""
        raise NotImplementedError("Subclasses must implement submit")

    def shutdown(self, wait: bool = True) -> None:
        """Graceful shutdown of the execution pool."""
        raise NotImplementedError("Subclasses must implement shutdown")


class InlineExecutor(Executor):
    """
    Synchronous execution in the orchestrator's thread.
    Serves as a reference for tests and MVP.
    """
    def submit(self, instance_id: InstanceId, task_id: TaskId, run_fn: Callable[[Dict], List[Dict]], inputs: Dict) -> None:
        try:
            result = run_fn(inputs)
            self.inbox.put(CompletionEvent(instance_id, task_id, True, result))
        except Exception as e:
            self.inbox.put(CompletionEvent(instance_id, task_id, False, e))

    def shutdown(self, wait: bool = True) -> None:
        pass


class ThreadPoolExecutorBackend(Executor):
    """
    Thread pool backend with max_workers.
    Completion is handled via add_done_callback which only puts the event in the queue.
    """
    def __init__(self, inbox: Inbox, max_workers: int = 4):
        super().__init__(inbox)
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, instance_id: InstanceId, task_id: TaskId, run_fn: Callable[[Dict], List[Dict]], inputs: Dict) -> None:
        future = self.pool.submit(run_fn, inputs)
        future.add_done_callback(lambda f: self._handle_completion(instance_id, task_id, f))

    def _handle_completion(self, instance_id: InstanceId, task_id: TaskId, future) -> None:
        try:
            result = future.result()
            self.inbox.put(CompletionEvent(instance_id, task_id, True, result))
        except Exception as e:
            self.inbox.put(CompletionEvent(instance_id, task_id, False, e))

    def shutdown(self, wait: bool = True) -> None:
        self.pool.shutdown(wait=wait)