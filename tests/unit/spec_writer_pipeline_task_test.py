"""
Tests for SpecWriterPipelineTask.

Verifies the composite task's behavior as a cyclic feedback graph,
focusing on external API (constructor, run(), introspection properties)
and interaction with passed external tasks (input_task, reviewer_task) and AiClient.
All mocks are implemented via inheritance, without unittest.mock.
"""

import unittest
from typing import Any, Dict, List, Optional

from src.fab.task import Task, TaskId, TaskStatus
from src.fab.policy import (
    ActivationPolicy, OncePolicy, AnyPolicy,
    InputMergePolicy, FailurePolicy,
)
from src.tasks.spec_writer_pipeline_task import SpecWriterPipelineTask
from src.utils import AiClient


# ---------------------------------------------------------------------------
# Mock AiClient
# ---------------------------------------------------------------------------

class MockAiClient(AiClient):
    """Mock AiClient that returns a fixed JSON response."""

    def __init__(self, response_text: str = '{"choices": [{"message": {"content": "mock response"}}]}'):
        self._response_text = response_text
        self.call_count = 0
        self.last_payload: Optional[Dict[str, Any]] = None

    def call(self, payload: Dict[str, Any], **override_params) -> str:
        self.call_count += 1
        self.last_payload = payload
        return self._response_text


# ---------------------------------------------------------------------------
# Mock Task base class
# ---------------------------------------------------------------------------

class MockTask(Task):
    """
    Base mock task that records calls and returns configurable results.
    Subclasses override run() to provide specific behavior.
    """

    def __init__(
        self,
        task_id: TaskId,
        activation_policy: Optional[ActivationPolicy] = None,
        merge_policy: Optional[InputMergePolicy] = None,
        failure_policy: Optional[FailurePolicy] = None,
        run_result: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(
            task_id=task_id,
            activation_policy=activation_policy or OncePolicy(),
            merge_policy=merge_policy,
            failure_policy=failure_policy,
        )
        self._run_result = run_result or []
        self.run_call_count = 0
        self.last_inputs: Optional[Dict[str, Any]] = None

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.run_call_count += 1
        self.last_inputs = inputs
        return self._run_result


# ---------------------------------------------------------------------------
# Mock Input Task
# ---------------------------------------------------------------------------

class MockInputTask(MockTask):
    """Mock input task that returns directories and files data."""

    def __init__(self, task_id: str = "input_task"):
        super().__init__(
            task_id=task_id,
            activation_policy=AnyPolicy(),
            run_result=[{"request": "test request", "directories": ["src/tasks"], "files": ["src/tasks/spec_writer_pipeline_task.py"]}],
        )

    def set_output(self, output: List[Dict[str, Any]]) -> None:
        """Dynamically change the output data for the next run."""
        self._run_result = output


# ---------------------------------------------------------------------------
# Mock Reviewer Task
# ---------------------------------------------------------------------------

class MockReviewerTask(MockTask):
    """
    Mock reviewer task that returns configurable results per call.
    Supports sequential responses for retry/edit/accept scenarios.
    """

    def __init__(self, task_id: str = "reviewer_task", responses: Optional[List[List[Dict[str, Any]]]] = None):
        super().__init__(
            task_id=task_id,
            activation_policy=AnyPolicy(),
        )
        self._responses = responses or [[{"__groups": ["accept"], "specification": "test spec"}]]
        self._call_index = 0

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.run_call_count += 1
        self.last_inputs = inputs
        if self._call_index < len(self._responses):
            result = self._responses[self._call_index]
            self._call_index += 1
            return result
        return [{"__groups": ["accept"], "specification": "final spec"}]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSpecWriterPipelineTask(unittest.TestCase):
    """Test suite for SpecWriterPipelineTask."""

    def setUp(self):
        """Set up common mocks for tests."""
        # Mock AiClient
        self.ai_client = MockAiClient()

        # Mock input task
        self.input_task = MockInputTask()

        # Mock reviewer task
        self.reviewer_task = MockReviewerTask()

    def test_constructor_accepts_parameters(self):
        """Test that SpecWriterPipelineTask can be constructed with all parameters."""
        pipeline_task = SpecWriterPipelineTask(
            task_id="pipeline",
            input_task=self.input_task,
            reviewer_task=self.reviewer_task,
            ai_client=self.ai_client,
        )
        self.assertIsNotNone(pipeline_task)
        self.assertIsInstance(pipeline_task, Task)

    def test_run_returns_specification_on_accept(self):
        """Test that run() returns specification when reviewer accepts."""
        pipeline_task = SpecWriterPipelineTask(
            task_id="pipeline",
            input_task=self.input_task,
            reviewer_task=self.reviewer_task,
            ai_client=self.ai_client,
        )
        result = pipeline_task.run({})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIn("specification", result[0])
        # The specification value may differ from "test spec" because code is read from real files
        self.assertIsInstance(result[0]["specification"], str)
        self.assertGreater(len(result[0]["specification"]), 0)

    def test_run_calls_input_task_run(self):
        """Test that input_task.run() is called during pipeline execution."""
        pipeline_task = SpecWriterPipelineTask(
            task_id="pipeline",
            input_task=self.input_task,
            reviewer_task=self.reviewer_task,
            ai_client=self.ai_client,
        )
        pipeline_task.run({})
        self.assertGreaterEqual(self.input_task.run_call_count, 1)

    def test_run_calls_reviewer_task_run(self):
        """Test that reviewer_task.run() is called during pipeline execution."""
        pipeline_task = SpecWriterPipelineTask(
            task_id="pipeline",
            input_task=self.input_task,
            reviewer_task=self.reviewer_task,
            ai_client=self.ai_client,
        )
        pipeline_task.run({})
        self.assertGreaterEqual(self.reviewer_task.run_call_count, 1)

    def test_retry_triggers_input_task_again(self):
        """Test that retry causes input_task to be called again."""
        reviewer = MockReviewerTask(responses=[
            [{"__groups": ["retry"]}],
            [{"__groups": ["accept"], "specification": "final spec"}],
        ])
        pipeline_task = SpecWriterPipelineTask(
            task_id="pipeline",
            input_task=self.input_task,
            reviewer_task=reviewer,
            ai_client=self.ai_client,
        )
        result = pipeline_task.run({})
        self.assertEqual(self.input_task.run_call_count, 2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["specification"], "final spec")

    def test_edit_passes_data_back_to_spec_writer(self):
        """Test that edit branch passes data back to spec_writer without re-running input_task."""
        reviewer = MockReviewerTask(responses=[
            [{"__groups": ["edit"], "specification": "edited spec", "request": "edited request", "code": "test code"}],
            [{"__groups": ["accept"], "specification": "final spec"}],
        ])
        pipeline_task = SpecWriterPipelineTask(
            task_id="pipeline",
            input_task=self.input_task,
            reviewer_task=reviewer,
            ai_client=self.ai_client,
        )
        result = pipeline_task.run({})
        # input_task should be called only once (initial run, not on edit)
        self.assertEqual(self.input_task.run_call_count, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["specification"], "final spec")

    def test_ai_client_passed_to_spec_writer(self):
        """Test that ai_client.call() is invoked by the internal SpecWriterTask."""
        pipeline_task = SpecWriterPipelineTask(
            task_id="pipeline",
            input_task=self.input_task,
            reviewer_task=self.reviewer_task,
            ai_client=self.ai_client,
        )
        pipeline_task.run({})
        self.assertGreaterEqual(self.ai_client.call_count, 1)

    def test_run_handles_empty_file_lists(self):
        """Test that SpecWriterPipelineTask does not crash with empty directories and files."""
        input_task = MockInputTask()
        input_task.set_output([{"request": "test request", "directories": [], "files": []}])
        pipeline_task = SpecWriterPipelineTask(
            task_id="pipeline",
            input_task=input_task,
            reviewer_task=self.reviewer_task,
            ai_client=self.ai_client,
        )
        result = pipeline_task.run({})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIn("specification", result[0])
        self.assertIsInstance(result[0]["specification"], str)

    def test_run_returns_specification_with_correct_structure(self):
        """Test that run() returns a list with a dict containing 'specification' as a string."""
        pipeline_task = SpecWriterPipelineTask(
            task_id="pipeline",
            input_task=self.input_task,
            reviewer_task=self.reviewer_task,
            ai_client=self.ai_client,
        )
        result = pipeline_task.run({})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], dict)
        self.assertIn("specification", result[0])
        self.assertIsInstance(result[0]["specification"], str)