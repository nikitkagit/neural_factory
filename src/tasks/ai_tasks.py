from typing import Any, Dict, List, Optional

from src.fab.task import Task, TaskId
from src.prompt import (
    Prompt, make_request_prompt, make_code_prompt, make_specification_prompt,
    make_previous_specification_prompt,
    GEN_SPEC, NO_CODE, GEN_SPEC_FOR_TESTS
)
from src.fab.policy import (
    ActivationPolicy, OncePolicy, 
    InputMergePolicy, FailurePolicy
)
from src.utils import AiClient


class CallAITask(Task):
    """
    Sends a request to the AI API with a composable prompt.
    Accepts a single 'input' (str) in inputs and merges it with an optional system Prompt.
    """
    
    def __init__(
        self, 
        task_id: TaskId, 
        ai_client: AiClient,
        prompt: Optional[Prompt] = None,
        activation_policy: ActivationPolicy = OncePolicy(),
        merge_policy: Optional[InputMergePolicy] = None,
        failure_policy: Optional[FailurePolicy] = None,
    ):
        super().__init__(task_id, activation_policy, merge_policy, failure_policy)
        self._ai_client = ai_client
        self._prompt = prompt or Prompt()

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        input_text = inputs.get("input")
        if not input_text or not isinstance(input_text, str):
            raise ValueError("Input must contain 'input' (str).")

        # Combine the system prompt with the user input
        combined_prompt = self._prompt + make_request_prompt(input_text)
        messages = [
            {"role": "user", "content": str(combined_prompt)}
        ]
        
        payload = {"messages": messages}
        response_text = self._ai_client.call(payload)
        
        return [{
            "response": response_text,
            "input": input_text
        }]


class SpecWriterTask(Task):
    """
    Writes a specification for existing code.
    Accepts 'code' (str) and 'request' (str) in inputs.
    Optionally accepts 'previous_specification' (str) in inputs.
    Combines them with a system prompt to generate a specification in Russian.
    """

    def __init__(
        self,
        task_id: TaskId,
        ai_client: AiClient,
        activation_policy: ActivationPolicy = OncePolicy(),
        merge_policy: Optional[InputMergePolicy] = None,
        failure_policy: Optional[FailurePolicy] = None,
    ):
        super().__init__(task_id, activation_policy, merge_policy, failure_policy)
        self._ai_client = ai_client

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        code = inputs.get("code")
        request = inputs.get("request")
        previous_specification = inputs.get("previous_specification")

        if not code or not isinstance(code, str):
            raise ValueError("Input must contain 'code' (str).")
        if not request or not isinstance(request, str):
            raise ValueError("Input must contain 'request' (str).")

        # Build the combined prompt with code and request
        combined = GEN_SPEC + NO_CODE
        combined += make_code_prompt(code) + make_request_prompt(request)

        # Add previous specification if present and is a string
        if previous_specification is not None and isinstance(previous_specification, str):
            combined += make_previous_specification_prompt(previous_specification)

        messages = [
            {"role": "user", "content": str(combined)}
        ]

        payload = {
            "model": "deepseek/deepseek-v4-flash",
            "messages": messages,
            "reasoning": {"effort": "none"},
            "max_tokens": 4999,
            "temperature": 1.0
        }
        
        response_text = self._ai_client.call(payload)

        result = {
            "specification": response_text,
            "code": code,
            "request": request
        }
        if previous_specification is not None and isinstance(previous_specification, str):
            result["previous_specification"] = previous_specification

        return [result]

class SpecTestWriterTask(Task):
    """
    Writes a test specification.
    Accepts either 'specification' (str) or 'request' (str) in inputs.
    If 'specification' is provided, it is used directly (current behavior).
    If 'request' is provided (and no 'specification'), the prompt is formed
    as in SpecWriterTask: GEN_SPEC_FOR_TESTS + NO_CODE + make_code_prompt(code) + make_request_prompt(request).
    If both are provided, 'specification' takes priority.
    """

    def __init__(
        self,
        task_id: TaskId,
        ai_client: AiClient,
        activation_policy: ActivationPolicy = OncePolicy(),
        merge_policy: Optional[InputMergePolicy] = None,
        failure_policy: Optional[FailurePolicy] = None,
    ):
        super().__init__(task_id, activation_policy, merge_policy, failure_policy)
        self._ai_client = ai_client

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        specification = inputs.get("specification")
        request = inputs.get("request")
        code = inputs.get("code")

        if code is None or not isinstance(code, str):
            raise ValueError("Input must contain 'code' (str).")

        # Priority: specification > request
        if specification is not None and isinstance(specification, str):
            # Use specification directly (current behavior)
            combined = GEN_SPEC_FOR_TESTS + NO_CODE + make_code_prompt(code) + make_specification_prompt(specification)
        elif request is not None and isinstance(request, str):
            # Form prompt as in SpecWriterTask
            combined = GEN_SPEC_FOR_TESTS + NO_CODE + make_code_prompt(code) + make_request_prompt(request)
        else:
            raise ValueError("Either 'specification' or 'request' must be provided in inputs.")

        messages = [
            {"role": "user", "content": str(combined)}
        ]

        payload = {
            "model": "deepseek/deepseek-v4-flash",
            "messages": messages,
            "reasoning": {"effort": "none"},
            "max_tokens": 4999,
            "temperature": 0.03
        }

        response_text = self._ai_client.call(payload)

        result = {
            "specification": response_text,
            "code": code,
        }
        if request is not None:
            result["request"] = request
        if specification is not None:
            result["specification_input"] = specification

        return [result]