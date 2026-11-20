import queue
from dataclasses import dataclass
from typing import Callable, Dict, Any
from concurrent.futures import ThreadPoolExecutor

TaskId = str
Inbox = queue.Queue


@dataclass(frozen=True)
class CompletionEvent:
    """
    Explicit completion event to avoid ambiguity between success and error.
    Ensures the orchestrator always receives exactly one clear signal per submission.
    """
    task_id: TaskId
    success: bool
    payload: Any  # Result dict if success, Exception if failure


class Executor:
    """
    Base class for task execution abstraction.
    Takes a ready-to-run task with an input snapshot, runs the computation, 
    and reports the result to the orchestrator via the inbox.
    """
    def __init__(self, inbox: Inbox):
        self.inbox = inbox

    def submit(self, task_id: TaskId, run_fn: Callable[[Dict], Dict], inputs: Dict) -> None:
        """Non-blocking: put the task in execution and return control."""
        raise NotImplementedError("Subclasses must implement submit")

    def shutdown(self, wait: bool = True) -> None:
        """Graceful shutdown of the execution pool."""
        raise NotImplementedError("Subclasses must implement shutdown")


class InlineExecutor(Executor):
    """
    Synchronous execution in the orchestrator's thread.
    Serves as a reference for tests and MVP.
    """
    def submit(self, task_id: TaskId, run_fn: Callable[[Dict], Dict], inputs: Dict) -> None:
        try:
            result = run_fn(inputs)
            self.inbox.put(CompletionEvent(task_id, True, result))
        except Exception as e:
            self.inbox.put(CompletionEvent(task_id, False, e))

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

    def submit(self, task_id: TaskId, run_fn: Callable[[Dict], Dict], inputs: Dict) -> None:
        future = self.pool.submit(run_fn, inputs)
        future.add_done_callback(lambda f: self._handle_completion(task_id, f))

    def _handle_completion(self, task_id: TaskId, future) -> None:
        try:
            result = future.result()
            self.inbox.put(CompletionEvent(task_id, True, result))
        except Exception as e:
            self.inbox.put(CompletionEvent(task_id, False, e))

    def shutdown(self, wait: bool = True) -> None:
        self.pool.shutdown(wait=wait)