import pytest
import re
from unittest.mock import MagicMock, patch, call

from src.tasks.ai_tasks import CallAITask, SpecWriterTask, SpecTestWriterTask, SpecToScriptTask
from src.fab.policy import OncePolicy
from src.prompt import Prompt, CODE_SEARCH_REPLACE_ONLY
from src.utils import AiClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ai_client():
    """Returns a MagicMock with a configured call method returning a string."""
    client = MagicMock(spec=AiClient)
    client.call.return_value = "Mocked AI response"
    return client


@pytest.fixture
def default_prompt():
    """Returns a default Prompt instance."""
    return Prompt()


# ---------------------------------------------------------------------------
# TestCallAITask
# ---------------------------------------------------------------------------

class TestCallAITask:
    """Tests for CallAITask."""

    def test_init_default_prompt(self, mock_ai_client):
        """Checks that CallAITask without prompt uses Prompt() by default."""
        task = CallAITask(task_id="test_call_ai", ai_client=mock_ai_client)
        assert isinstance(task._prompt, Prompt)
        assert str(task._prompt) == ""

    def test_init_custom_prompt(self, mock_ai_client):
        """Checks that CallAITask with a custom prompt saves it."""
        custom_prompt = Prompt("Custom system prompt", priority=50)
        task = CallAITask(
            task_id="test_call_ai_custom",
            ai_client=mock_ai_client,
            prompt=custom_prompt,
        )
        assert task._prompt is custom_prompt
        assert str(task._prompt) == "Custom system prompt"

    def test_run_success(self, mock_ai_client):
        """Checks successful run with correct input, payload formation, and result."""
        task = CallAITask(task_id="test_run_success", ai_client=mock_ai_client)
        inputs = {"input": "Hello, AI!"}
        result = task.run(inputs)

        # Verify AiClient.call was invoked
        mock_ai_client.call.assert_called_once()

        # Verify payload structure
        call_args = mock_ai_client.call.call_args[0][0]
        assert "messages" in call_args
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert "Hello, AI!" in call_args["messages"][0]["content"]

        # Verify result structure
        assert len(result) == 1
        assert result[0]["response"] == "Mocked AI response"
        assert result[0]["input"] == "Hello, AI!"

    def test_run_missing_input(self, mock_ai_client):
        """Checks that missing 'input' key raises ValueError."""
        task = CallAITask(task_id="test_missing_input", ai_client=mock_ai_client)
        with pytest.raises(ValueError, match=re.escape("Input must contain 'input' (str).")):
            task.run({})

    def test_run_empty_input(self, mock_ai_client):
        """Checks that empty string in input raises ValueError."""
        task = CallAITask(task_id="test_empty_input", ai_client=mock_ai_client)
        with pytest.raises(ValueError, match=re.escape("Input must contain 'input' (str).")):
            task.run({"input": ""})

    def test_run_non_string_input(self, mock_ai_client):
        """Checks that non-string input raises ValueError."""
        task = CallAITask(task_id="test_non_string_input", ai_client=mock_ai_client)
        with pytest.raises(ValueError, match=re.escape("Input must contain 'input' (str).")):
            task.run({"input": 123})

    def test_run_ai_client_called(self, mock_ai_client):
        """Checks that AiClient.call is called exactly once."""
        task = CallAITask(task_id="test_ai_called", ai_client=mock_ai_client)
        task.run({"input": "test"})
        assert mock_ai_client.call.call_count == 1

    def test_run_prompt_combination(self, mock_ai_client):
        """Checks that system prompt and input_text are correctly combined via Prompt.__add__."""
        system_prompt = Prompt("System instruction", priority=10)
        task = CallAITask(
            task_id="test_prompt_comb",
            ai_client=mock_ai_client,
            prompt=system_prompt,
        )
        task.run({"input": "User request"})

        call_args = mock_ai_client.call.call_args[0][0]
        content = call_args["messages"][0]["content"]

        # Both parts should be present
        assert "System instruction" in content
        assert "User request" in content


# ---------------------------------------------------------------------------
# TestSpecWriterTask
# ---------------------------------------------------------------------------

class TestSpecWriterTask:
    """Tests for SpecWriterTask."""

    def test_init(self, mock_ai_client):
        """Checks correct initialization of SpecWriterTask."""
        task = SpecWriterTask(task_id="test_spec_writer", ai_client=mock_ai_client)
        assert task.task_id == "test_spec_writer"
        assert task._ai_client is mock_ai_client
        assert isinstance(task.activation_policy, OncePolicy)

    def test_run_success(self, mock_ai_client):
        """Checks successful run with code and request."""
        task = SpecWriterTask(task_id="test_spec_success", ai_client=mock_ai_client)
        inputs = {"code": "def foo(): pass", "request": "Add docstring"}
        result = task.run(inputs)

        mock_ai_client.call.assert_called_once()
        assert len(result) == 1
        assert result[0]["specification"] == "Mocked AI response"
        assert result[0]["code"] == "def foo(): pass"
        assert result[0]["request"] == "Add docstring"

    def test_run_missing_code(self, mock_ai_client):
        """Checks that missing 'code' raises ValueError."""
        task = SpecWriterTask(task_id="test_missing_code", ai_client=mock_ai_client)
        with pytest.raises(ValueError, match=re.escape("Input must contain 'code' (str).")):
            task.run({"request": "test"})

    def test_run_missing_request(self, mock_ai_client):
        """Checks that missing 'request' raises ValueError."""
        task = SpecWriterTask(task_id="test_missing_request", ai_client=mock_ai_client)
        with pytest.raises(ValueError, match=re.escape("Input must contain 'request' (str).")):
            task.run({"code": "def foo(): pass"})

    def test_run_non_string_code(self, mock_ai_client):
        """Checks that non-string code raises ValueError."""
        task = SpecWriterTask(task_id="test_non_string_code", ai_client=mock_ai_client)
        with pytest.raises(ValueError, match=re.escape("Input must contain 'code' (str).")):
            task.run({"code": 123, "request": "test"})

    def test_run_non_string_request(self, mock_ai_client):
        """Checks that non-string request raises ValueError."""
        task = SpecWriterTask(task_id="test_non_string_request", ai_client=mock_ai_client)
        with pytest.raises(ValueError, match=re.escape("Input must contain 'request' (str).")):
            task.run({"code": "def foo(): pass", "request": 456})

    def test_run_with_previous_specification(self, mock_ai_client):
        """Checks that previous_specification is added to the prompt."""
        task = SpecWriterTask(task_id="test_with_prev_spec", ai_client=mock_ai_client)
        inputs = {
            "code": "def foo(): pass",
            "request": "Add docstring",
            "previous_specification": "Old spec content",
        }
        result = task.run(inputs)

        call_args = mock_ai_client.call.call_args[0][0]
        content = call_args["messages"][0]["content"]

        assert "Old spec content" in content
        assert result[0]["previous_specification"] == "Old spec content"

    def test_run_without_previous_specification(self, mock_ai_client):
        """Checks that without previous_specification, prompt is formed without it."""
        task = SpecWriterTask(task_id="test_without_prev_spec", ai_client=mock_ai_client)
        inputs = {"code": "def foo(): pass", "request": "Add docstring"}
        result = task.run(inputs)

        call_args = mock_ai_client.call.call_args[0][0]
        content = call_args["messages"][0]["content"]

        assert "Previous specification" not in content
        assert "previous_specification" not in result[0]

    def test_run_payload_structure(self, mock_ai_client):
        """Checks payload structure: model, reasoning, max_tokens, temperature."""
        task = SpecWriterTask(task_id="test_payload_struct", ai_client=mock_ai_client)
        inputs = {"code": "def foo(): pass", "request": "Add docstring"}
        task.run(inputs)

        call_args = mock_ai_client.call.call_args[0][0]
        assert call_args["model"] == "deepseek/deepseek-v4-flash"
        assert call_args["reasoning"] == {"effort": "none"}
        assert call_args["max_tokens"] == 4999
        assert call_args["temperature"] == 1.0

    def test_run_result_structure(self, mock_ai_client):
        """Checks result structure: specification, code, request, optional previous_specification."""
        task = SpecWriterTask(task_id="test_result_struct", ai_client=mock_ai_client)
        inputs = {
            "code": "def foo(): pass",
            "request": "Add docstring",
            "previous_specification": "Old spec",
        }
        result = task.run(inputs)

        assert len(result) == 1
        assert "specification" in result[0]
        assert "code" in result[0]
        assert "request" in result[0]
        assert "previous_specification" in result[0]

        # Without previous_specification
        result2 = task.run({"code": "x = 1", "request": "test"})
        assert "previous_specification" not in result2[0]


# ---------------------------------------------------------------------------
# TestSpecToScriptTask
# ---------------------------------------------------------------------------

class TestSpecToScriptTask:
    """Tests for SpecToScriptTask."""

    def test_run_with_valid_inputs(self, mock_ai_client):
        """Checks that run calls ai_client once and returns correct structure."""
        task = SpecToScriptTask(task_id="test", ai_client=mock_ai_client)
        mock_ai_client.call.return_value = (
            "[/path/to/file.py]\n"
            "[SEARCH]\n"
            "old code\n"
            "[REPLACE]\n"
            "new code\n"
        )
        result = task.run({"specification": "some spec", "code": "some code"})

        mock_ai_client.call.assert_called_once()
        payload = mock_ai_client.call.call_args[0][0]
        content = payload["messages"][0]["content"]
        assert str(CODE_SEARCH_REPLACE_ONLY) in content
        assert "some code" in content
        assert "some spec" in content

        assert len(result) == 1
        assert result[0]["filepath"] == "/path/to/file.py"
        assert result[0]["search"] == "old code"
        assert result[0]["replace"] == "new code"

    def test_run_missing_specification(self, mock_ai_client):
        """Checks that missing specification raises ValueError."""
        task = SpecToScriptTask(task_id="test", ai_client=mock_ai_client)
        with pytest.raises(ValueError, match="Input must contain 'specification' \\(str\\)\\."):
            task.run({"code": "some code"})

    def test_run_missing_code(self, mock_ai_client):
        """Checks that missing code raises ValueError."""
        task = SpecToScriptTask(task_id="test", ai_client=mock_ai_client)
        with pytest.raises(ValueError, match="Input must contain 'code' \\(str\\)\\."):
            task.run({"specification": "some spec"})

    def test_run_empty_specification(self, mock_ai_client):
        """Checks that empty specification is passed in prompt and result contains output."""
        task = SpecToScriptTask(task_id="test", ai_client=mock_ai_client)
        mock_ai_client.call.return_value = ""
        result = task.run({"specification": "", "code": "some code"})

        mock_ai_client.call.assert_called_once()
        payload = mock_ai_client.call.call_args[0][0]
        content = payload["messages"][0]["content"]
        assert "some code" in content

        assert result == []

    def test_run_empty_code(self, mock_ai_client):
        """Checks that empty code is passed in prompt and result contains output."""
        task = SpecToScriptTask(task_id="test", ai_client=mock_ai_client)
        mock_ai_client.call.return_value = ""
        result = task.run({"specification": "some spec", "code": ""})

        mock_ai_client.call.assert_called_once()
        payload = mock_ai_client.call.call_args[0][0]
        content = payload["messages"][0]["content"]
        assert "some spec" in content

        assert result == []

    def test_parse_response_with_valid_blocks(self, mock_ai_client):
        """Checks that multiple valid blocks are parsed correctly."""
        task = SpecToScriptTask(task_id="test", ai_client=mock_ai_client)
        mock_ai_client.call.return_value = (
            "[/path/to/a.py]\n"
            "[SEARCH]\n"
            "old a\n"
            "[REPLACE]\n"
            "new a\n"
            "[/path/to/b.py]\n"
            "[SEARCH]\n"
            "old b\n"
            "[REPLACE]\n"
            "new b\n"
        )
        result = task.run({"specification": "spec", "code": "code"})

        assert len(result) == 2
        assert result[0]["filepath"] == "/path/to/a.py"
        assert result[0]["search"] == "old a"
        assert result[0]["replace"] == "new a"
        assert result[1]["filepath"] == "/path/to/b.py"
        assert result[1]["search"] == "old b"
        assert result[1]["replace"] == "new b"

    def test_parse_response_with_no_blocks(self, mock_ai_client):
        """Checks that response without blocks returns empty output."""
        task = SpecToScriptTask(task_id="test", ai_client=mock_ai_client)
        mock_ai_client.call.return_value = "OK"
        result = task.run({"specification": "spec", "code": "code"})

        assert result == []

    def test_parse_response_with_malformed_blocks(self, mock_ai_client):
        """Checks that malformed blocks are skipped and only valid ones are returned."""
        task = SpecToScriptTask(task_id="test", ai_client=mock_ai_client)
        mock_ai_client.call.return_value = (
            "[/path/to/valid.py]\n"
            "[SEARCH]\n"
            "valid search\n"
            "[REPLACE]\n"
            "valid replace\n"
            "[/path/to/malformed.py]\n"
            "[SEARCH]\n"
            "missing replace\n"
            "[/path/to/valid2.py]\n"
            "[SEARCH]\n"
            "valid2 search\n"
            "[REPLACE]\n"
            "valid2 replace\n"
        )
        result = task.run({"specification": "spec", "code": "code"})

        assert len(result) == 2
        assert result[0]["filepath"] == "/path/to/valid.py"
        assert result[1]["filepath"] == "/path/to/valid2.py"

    def test_run_with_non_string_inputs(self, mock_ai_client):
        """Checks that non-string specification or code raises ValueError."""
        task = SpecToScriptTask(task_id="test", ai_client=mock_ai_client)

        with pytest.raises(ValueError, match="Input must contain 'specification' \\(str\\)\\."):
            task.run({"specification": 123, "code": "some code"})

        with pytest.raises(ValueError, match="Input must contain 'code' \\(str\\)\\."):
            task.run({"specification": "some spec", "code": 456})


# ---------------------------------------------------------------------------
# TestSpecTestWriterTask
# ---------------------------------------------------------------------------

class TestSpecTestWriterTask:
    """Tests for SpecTestWriterTask."""

    def test_init(self, mock_ai_client):
        """Checks correct initialization of SpecTestWriterTask."""
        task = SpecTestWriterTask(task_id="test_test_spec", ai_client=mock_ai_client)
        assert task.task_id == "test_test_spec"
        assert task._ai_client is mock_ai_client
        assert isinstance(task.activation_policy, OncePolicy)

    def test_run_with_specification(self, mock_ai_client):
        """Checks that with specification, GEN_SPEC_FOR_TESTS + make_specification_prompt is used."""
        task = SpecTestWriterTask(task_id="test_with_spec", ai_client=mock_ai_client)
        inputs = {
            "code": "def foo(): pass",
            "specification": "Test spec content",
        }
        result = task.run(inputs)

        call_args = mock_ai_client.call.call_args[0][0]
        content = call_args["messages"][0]["content"]

        assert "Test spec content" in content
        assert result[0]["specification_input"] == "Test spec content"
        assert result[0]["specification"] == "Mocked AI response"

    def test_run_with_request(self, mock_ai_client):
        """Checks that without specification but with request, GEN_SPEC_FOR_TESTS + make_request_prompt is used."""
        task = SpecTestWriterTask(task_id="test_with_request", ai_client=mock_ai_client)
        inputs = {
            "code": "def foo(): pass",
            "request": "Test request content",
        }
        result = task.run(inputs)

        call_args = mock_ai_client.call.call_args[0][0]
        content = call_args["messages"][0]["content"]

        assert "Test request content" in content
        assert result[0]["request"] == "Test request content"
        assert "specification_input" not in result[0]

    def test_run_specification_priority(self, mock_ai_client):
        """Checks that specification has priority over request."""
        task = SpecTestWriterTask(task_id="test_spec_priority", ai_client=mock_ai_client)
        inputs = {
            "code": "def foo(): pass",
            "specification": "Spec content",
            "request": "Request content",
        }
        result = task.run(inputs)

        call_args = mock_ai_client.call.call_args[0][0]
        content = call_args["messages"][0]["content"]

        # Specification should be used, not request
        assert "Spec content" in content
        assert "Request content" not in content
        assert result[0]["specification_input"] == "Spec content"
        assert "request" in result[0]

    def test_run_missing_code(self, mock_ai_client):
        """Checks that missing code raises ValueError."""
        task = SpecTestWriterTask(task_id="test_missing_code", ai_client=mock_ai_client)
        with pytest.raises(ValueError, match=re.escape("Input must contain 'code' (str).")):
            task.run({"specification": "test"})

    def test_run_non_string_code(self, mock_ai_client):
        """Checks that non-string code raises ValueError."""
        task = SpecTestWriterTask(task_id="test_non_string_code", ai_client=mock_ai_client)
        with pytest.raises(ValueError, match=re.escape("Input must contain 'code' (str).")):
            task.run({"code": 123, "specification": "test"})

    def test_run_missing_both_spec_and_request(self, mock_ai_client):
        """Checks that missing both specification and request raises ValueError."""
        task = SpecTestWriterTask(task_id="test_missing_both", ai_client=mock_ai_client)
        with pytest.raises(ValueError, match=re.escape("Either 'specification' or 'request' must be provided")):
            task.run({"code": "def foo(): pass"})

    def test_run_result_structure_with_spec(self, mock_ai_client):
        """Checks result structure when using specification."""
        task = SpecTestWriterTask(task_id="test_result_spec", ai_client=mock_ai_client)
        inputs = {
            "code": "def foo(): pass",
            "specification": "Spec content",
        }
        result = task.run(inputs)

        assert len(result) == 1
        assert "specification" in result[0]
        assert "code" in result[0]
        assert "specification_input" in result[0]
        assert "request" not in result[0]

    def test_run_result_structure_with_request(self, mock_ai_client):
        """Checks result structure when using request."""
        task = SpecTestWriterTask(task_id="test_result_request", ai_client=mock_ai_client)
        inputs = {
            "code": "def foo(): pass",
            "request": "Request content",
        }
        result = task.run(inputs)

        assert len(result) == 1
        assert "specification" in result[0]
        assert "code" in result[0]
        assert "request" in result[0]
        assert "specification_input" not in result[0]

    def test_run_payload_temperature(self, mock_ai_client):
        """Checks that temperature equals 0.03."""
        task = SpecTestWriterTask(task_id="test_temperature", ai_client=mock_ai_client)
        inputs = {
            "code": "def foo(): pass",
            "specification": "test",
        }
        task.run(inputs)

        call_args = mock_ai_client.call.call_args[0][0]
        assert call_args["temperature"] == 0.03