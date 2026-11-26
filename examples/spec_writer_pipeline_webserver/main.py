"""
Web server application for user input and review workflow.
Uses http.server to serve a web interface on localhost:8080.
"""

import json
import logging
import os
import queue
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional, Tuple

from src.fab.policy import AnyPolicy
from src.tasks.spec_writer_pipeline_task import SpecWriterPipelineTask
from src.utils import AiClient, setup_logging

from .tasks import WebInputTask, WebReviewerTask

logger = logging.getLogger(__name__)

VALID_DECISIONS = ("accept", "edit", "retry")
WAIT_TIMEOUT = 120


class UserInputWebServer:
    """
    Manages the web server lifecycle and coordinates with custom tasks
    via queues and events.
    """

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Queues for communication with tasks
        self.input_queue: queue.Queue = queue.Queue()
        self.review_queue: queue.Queue = queue.Queue()

        # Events to signal that data is available
        self.input_ready = threading.Event()
        self.review_ready = threading.Event()

        # Storage for the latest specification result
        self.specification_result: Optional[str] = None
        self.specification_ready = threading.Event()
        self.error_message: Optional[str] = None

    def start(self) -> None:
        """Starts the HTTP server in a background thread."""
        handler_factory = lambda *args, **kwargs: RequestHandler(
            *args, server_app=self, **kwargs
        )
        # ThreadingHTTPServer allows a long-polling /specification request to
        # block without freezing /review, /stop and other concurrent requests.
        self._server = ThreadingHTTPServer((self.host, self.port), handler_factory)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        logger.info("Web server started on http://%s:%d", self.host, self.port)

    def stop(self) -> None:
        """Stops the HTTP server gracefully and unblocks any waiting tasks."""
        # Signal tasks first so their blocking waits abort even if no request arrives.
        self._stop_event.set()
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        logger.info("Web server stopped")

    def _on_spec_ready(self, specification: str) -> None:
        """Callback invoked by the reviewer task when a spec is ready for review."""
        self.error_message = None
        self.specification_result = specification
        self.specification_ready.set()

    def _publish_failure(self, message: str) -> None:
        """Releases any HTTP handlers waiting for a specification with an error."""
        self.error_message = message
        self.specification_result = None
        self.specification_ready.set()

    def run_pipeline(self, ai_client: AiClient) -> Dict[str, Any]:
        input_task = WebInputTask(
            task_id="web_input",
            activation_policy=AnyPolicy(),
            input_queue=self.input_queue,
            input_ready=self.input_ready,
            stop_event=self._stop_event,
        )

        reviewer_task = WebReviewerTask(
            task_id="web_reviewer",
            activation_policy=AnyPolicy(),
            review_queue=self.review_queue,
            review_ready=self.review_ready,
            spec_ready_callback=self._on_spec_ready,
            stop_event=self._stop_event,
        )

        pipeline = SpecWriterPipelineTask(
            task_id="spec_pipeline",
            input_task=input_task,
            reviewer_task=reviewer_task,
            ai_client=ai_client,
        )

        try:
            result = pipeline.run({})
        except Exception as e:
            logger.error("Pipeline crashed: %s", e)
            self._publish_failure(str(e))
            return {"status": "error", "message": str(e)}

        if result:
            return {"status": "success", "specification": result[0].get("specification", "")}

        self._publish_failure("No specification was accepted")
        return {"status": "error", "message": "No specification was accepted"}


class RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler that routes requests to the appropriate handlers."""

    def __init__(self, *args, server_app: UserInputWebServer, **kwargs):
        self.server_app = server_app
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if self.path == "/":
            self._serve_index()
        elif self.path == "/specification":
            self._handle_wait_specification()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self) -> None:
        if self.path == "/run":
            self._handle_run()
        elif self.path == "/stop":
            self._handle_stop()
        elif self.path == "/review":
            self._handle_review()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _read_json(self) -> Optional[Dict[str, Any]]:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            data = json.loads(body)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    def _serve_index(self) -> None:
        try:
            with open("examples/spec_writer_pipeline_webserver/templates/index.html", "r", encoding="utf-8") as f:
                html = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        except FileNotFoundError:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"Template not found")

    def _wait_result(self) -> Dict[str, Any]:
        """Shared wait logic for /run and /specification."""
        if self.server_app.specification_ready.wait(timeout=WAIT_TIMEOUT):
            spec = self.server_app.specification_result
            if spec is not None:
                return {"status": "success", "specification": spec}
            return {"status": "error", "message": self.server_app.error_message or "Pipeline failed"}
        return {"status": "error", "message": "Pipeline timed out"}

    def _handle_run(self) -> None:
        """Handle the /run endpoint: parse input and trigger the pipeline."""
        data = self._read_json()
        if data is None:
            self._send_json({"status": "error", "message": "Invalid JSON"}, status=400)
            return

        request_text = data.get("request", "")
        input_text = data.get("input", "")
        directories, files = self._parse_input(input_text)

        # Reset the ready flag so we wait for the NEW specification, not a stale one.
        self.server_app.specification_ready.clear()
        self.server_app.specification_result = None
        self.server_app.error_message = None

        self.server_app.input_queue.put({
            "request": request_text,
            "directories": directories,
            "files": files,
        })
        self.server_app.input_ready.set()

        self._send_json(self._wait_result())

    def _handle_wait_specification(self) -> None:
        """
        Long-polling endpoint used after an "edit" decision:
        blocks until the reviewer publishes the regenerated specification.
        """
        result = self._wait_result()
        if result.get("status") == "success":
            # Consume the event so the next cycle waits for a fresh specification.
            self.server_app.specification_ready.clear()
        self._send_json(result)

    def _handle_stop(self) -> None:
        self._send_json({"status": "stopped"})
        # Schedule server shutdown in a separate thread to avoid deadlock.
        threading.Thread(target=self.server_app.stop, daemon=True).start()

    def _handle_review(self) -> None:
        """Handle the /review endpoint: accept a reviewer decision."""
        data = self._read_json()
        if data is None:
            self._send_json({"status": "error", "message": "Invalid JSON"}, status=400)
            return

        decision = data.get("decision", "")
        if decision not in VALID_DECISIONS:
            self._send_json({"status": "error", "message": "Invalid decision"}, status=400)
            return

        # For edit/retry a NEW specification will be produced. Reset the ready
        # flag BEFORE waking the reviewer so /specification waits for the new one.
        if decision in ("edit", "retry"):
            self.server_app.specification_ready.clear()
            self.server_app.specification_result = None

        # "__groups" must be a LIST for the orchestrator's group filtering.
        # "request" is forwarded so the user can change the task before "edit".
        self.server_app.review_queue.put({
            "__groups": [decision],
            "specification": data.get("specification", ""),
            "request": data.get("request", ""),
        })
        self.server_app.review_ready.set()

        self._send_json({"status": "received"})

    def _parse_input(self, input_text: str) -> Tuple[list, list]:
        """
        Parse a space-separated string into directories and files.
        Directories are paths that end with '/' or are existing directories.
        """
        parts = input_text.strip().split()
        directories = []
        files = []

        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.endswith("/") or (os.path.isdir(part) if os.path.exists(part) else False):
                directories.append(part.rstrip("/"))
            else:
                files.append(part)

        return directories, files


def main() -> None:
    """Entry point for the web server application."""
    setup_logging()

    ai_client = AiClient()

    server_app = UserInputWebServer()
    server_app.start()

    try:
        result = server_app.run_pipeline(ai_client)
        logger.info("Pipeline result: %s", result)
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
    finally:
        server_app.stop()


if __name__ == "__main__":
    main()