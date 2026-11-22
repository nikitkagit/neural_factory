import pytest
from unittest.mock import patch, MagicMock

from src.tasks.ai_tasks import TestSpecWriterTask
from src.prompt import (
    GEN_SPEC_FOR_TESTS,
    make_code_prompt,
    make_specification_prompt,
)
from src.fab.policy import (
    OncePolicy,
    InputMergePolicy,
    FailurePolicy,
)


class TestTestSpecWriterTask:
    """Test suite for TestSpecWriterTask."""

    def setup_method(self) -> None:
        """Initialize a TestSpecWriterTask instance for each test."""
        self.task = TestSpecWriterTask(task_id="test_spec_writer")

    def test_run_valid_inputs(self) -> None:
        """Test that valid inputs produce correct output structure."""
        inputs = {
            "specification": "Test specification content",
            "code": "def foo():\n    pass",
        }
        result = self.task.run(inputs)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)

        output = result[0]
        assert "specification" in output
        assert "code" in output
        assert "specification_input" in output

        assert output["code"] == inputs["code"]
        assert output["specification_input"] == inputs["specification"]

    def test_run_missing_specification(self) -> None:
        """Test that missing 'specification' raises ValueError."""
        inputs = {"code": "def foo():\n    pass"}
        with pytest.raises(ValueError, match="Both 'specification' and 'code' must be provided"):
            self.task.run(inputs)

    def test_run_missing_code(self) -> None:
        """Test that missing 'code' raises ValueError."""
        inputs = {"specification": "Test spec"}
        with pytest.raises(ValueError, match="Both 'specification' and 'code' must be provided"):
            self.task.run(inputs)

    def test_run_non_string_specification(self) -> None:
        """Test that non-string specification raises ValueError."""
        inputs = {"specification": 123, "code": "def foo():\n    pass"}
        with pytest.raises(ValueError, match="Both 'specification' and 'code' must be strings"):
            self.task.run(inputs)

    def test_run_non_string_code(self) -> None:
        """Test that non-string code raises ValueError."""
        inputs = {"specification": "Test spec", "code": ["def foo():\n    pass"]}
        with pytest.raises(ValueError, match="Both 'specification' and 'code' must be strings"):
            self.task.run(inputs)

    @patch("src.tasks.ai_tasks.call_ai")
    def test_run_prompt_construction(self, mock_call_ai: MagicMock) -> None:
        """Test that the prompt is correctly constructed and passed to call_ai."""
        mock_call_ai.return_value = "Generated specification for tests"

        specification = "Original specification"
        code = "def bar():\n    return 42"
        inputs = {"specification": specification, "code": code}

        self.task.run(inputs)

        assert mock_call_ai.call_count == 1
        payload = mock_call_ai.call_args[0][0]

        assert "messages" in payload
        messages = payload["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

        content = messages[0]["content"]
        expected_prompt = str(
            GEN_SPEC_FOR_TESTS + make_code_prompt(code) + make_specification_prompt(specification)
        )
        assert content == expected_prompt

    @patch("src.tasks.ai_tasks.call_ai")
    def test_run_returns_correct_structure(self, mock_call_ai: MagicMock) -> None:
        """Test that the returned structure matches expectations with mocked AI response."""
        expected_response = "Generated test specification"
        mock_call_ai.return_value = expected_response

        specification = "Original specification"
        code = "def bar():\n    return 42"
        inputs = {"specification": specification, "code": code}

        result = self.task.run(inputs)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)

        output = result[0]
        assert "specification" in output
        assert "code" in output
        assert "specification_input" in output

        assert output["specification"] == expected_response
        assert output["code"] == code
        assert output["specification_input"] == specification

    def test_run_with_different_policies(self) -> None:
        """Test that different policy combinations produce correct results."""
        policy_combinations = [
            (OncePolicy(), None, None),
            (OncePolicy(), InputMergePolicy(), None),
            (OncePolicy(), None, FailurePolicy()),
            (OncePolicy(), InputMergePolicy(), FailurePolicy()),
        ]

        inputs = {
            "specification": "Test specification",
            "code": "def baz():\n    return True",
        }

        for activation_policy, merge_policy, failure_policy in policy_combinations:
            task = TestSpecWriterTask(
                task_id="test_spec_writer",
                activation_policy=activation_policy,
                merge_policy=merge_policy,
                failure_policy=failure_policy,
            )

            result = task.run(inputs)

            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], dict)

            output = result[0]
            assert "specification" in output
            assert "code" in output
            assert "specification_input" in output
            assert output["code"] == inputs["code"]
            assert output["specification_input"] == inputs["specification"]