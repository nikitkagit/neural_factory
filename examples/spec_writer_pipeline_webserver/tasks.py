"""
Custom Task implementations for web-based user input and review.
Uses queue.Queue and threading.Event for asynchronous communication with the web server.
"""

import queue
import threading
from typing import Any, Callable, Dict, List, Optional

from src.fab.task import Task, TaskId
from src.fab.policy import (
    ActivationPolicy,
    InputMergePolicy,
    FailurePolicy,
)


def _wait_for_event(event: threading.Event, stop_event: Optional[threading.Event]) -> None:
    """
    Block until `event` is set.
    Polls with a timeout so a server shutdown can abort the wait instead of hanging forever.
    """
    while not event.wait(timeout=0.2):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Web server is stopping")


class WebInputTask(Task):
    """
    A Task that waits for user input from the web interface.
    When run() is called, it blocks until data is available in the input_queue.
    The expected state format is:
        {"request": "...", "directories": [...], "files": [...]}
    """

    def __init__(
        self,
        task_id: TaskId,
        input_queue: queue.Queue,
        input_ready: threading.Event,
        activation_policy: ActivationPolicy,
        merge_policy: Optional[InputMergePolicy] = None,
        failure_policy: Optional[FailurePolicy] = None,
        stop_event: Optional[threading.Event] = None,
    ):
        super().__init__(task_id, activation_policy, merge_policy, failure_policy)
        self._input_queue = input_queue
        self._input_ready = input_ready
        self._stop_event = stop_event

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Wait for the user to submit the form (Run button or an automatic retry).
        _wait_for_event(self._input_ready, self._stop_event)
        self._input_ready.clear()

        try:
            data = self._input_queue.get_nowait()
        except queue.Empty:
            return [{"request": "", "directories": [], "files": []}]

        return [{
            "request": data.get("request", ""),
            "directories": data.get("directories", []),
            "files": data.get("files", []),
        }]


class WebReviewerTask(Task):
    """
    A Task that waits for a review decision from the web interface.

    IMPORTANT: "__groups" MUST be a list (e.g. ["edit"]), because
    Orchestrator._extract_groups only recognizes list-valued groups.
    A string value silently disables group filtering and breaks routing.

    "request" and "code" are passed through from the incoming snapshot so the
    "edit" branch can feed spec_writer with a complete input set.
    """

    def __init__(
        self,
        task_id: TaskId,
        review_queue: queue.Queue,
        review_ready: threading.Event,
        activation_policy: ActivationPolicy,
        merge_policy: Optional[InputMergePolicy] = None,
        failure_policy: Optional[FailurePolicy] = None,
        spec_ready_callback: Optional[Callable[[str], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ):
        super().__init__(task_id, activation_policy, merge_policy, failure_policy)
        self._review_queue = review_queue
        self._review_ready = review_ready
        self._spec_ready_callback = spec_ready_callback
        self._stop_event = stop_event

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        specification = inputs.get("specification", "")
        request = inputs.get("request", "")
        code = inputs.get("code", "")

        # Publish the specification to the web UI BEFORE blocking on the user
        # decision. This breaks the deadlock between /run and the reviewer.
        if self._spec_ready_callback:
            self._spec_ready_callback(specification)

        _wait_for_event(self._review_ready, self._stop_event)
        self._review_ready.clear()

        try:
            data = self._review_queue.get_nowait()
        except queue.Empty:
            data = {}

        groups = data.get("__groups", ["retry"])
        # Normalize to a list: the orchestrator requires list-valued groups.
        if isinstance(groups, str):
            groups = [groups]

        # The user may have edited the specification text and/or the request.
        # Prefer the values coming from the web form, fall back to the snapshot.
        return [{
            "__groups": groups,
            "specification": data.get("specification") or specification,
            "request": data.get("request") or request,
            "code": code,
        }]