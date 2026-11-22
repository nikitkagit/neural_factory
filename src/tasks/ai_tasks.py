from typing import Any, Dict, List, Optional

from src.fab.task import Task, TaskId
from src.prompt import (
    Prompt, make_request_prompt, make_code_prompt,
    GEN_SPEC, NO_CODE
)
from src.fab.policy import (
    ActivationPolicy, OncePolicy, 
    InputMergePolicy, FailurePolicy
)
from ..utils import call_ai


class CallAITask(Task):
    """
    Sends a request to the AI API with a composable prompt.
    Accepts a single 'input' (str) in inputs and merges it with an optional system Prompt.
    """
    
    def __init__(
        self, 
        task_id: TaskId, 
        prompt: Optional[Prompt] = None,
        activation_policy: ActivationPolicy = OncePolicy(),
        merge_policy: Optional[InputMergePolicy] = None,
        failure_policy: Optional[FailurePolicy] = None,
    ):
        super().__init__(task_id, activation_policy, merge_policy, failure_policy)
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
        response_text = call_ai(payload)
        
        return [{
            "response": response_text,
            "input": input_text
        }]

class SpecWriterTask(Task):
    """
    Writes a specification for existing code.
    Accepts 'code' (str) and 'request' (str) in inputs.
    Combines them with a system prompt to generate a specification in Russian.
    """

    def __init__(
        self,
        task_id: TaskId,
        activation_policy: ActivationPolicy = OncePolicy(),
        merge_policy: Optional[InputMergePolicy] = None,
        failure_policy: Optional[FailurePolicy] = None,
    ):
        super().__init__(task_id, activation_policy, merge_policy, failure_policy)

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        code = inputs.get("code")
        request = inputs.get("request")

        if not code or not isinstance(code, str):
            raise ValueError("Input must contain 'code' (str).")
        if not request or not isinstance(request, str):
            raise ValueError("Input must contain 'request' (str).")

        # Build the combined prompt with code and request
        combined = GEN_SPEC + NO_CODE
        combined += make_code_prompt(code) + make_request_prompt(request)

        messages = [
            {"role": "user", "content": str(combined)}
        ]

        payload = {
            "model": "deepseek/deepseek-v4-flash",
            "messages": messages,
            "reasoning": {"effort": "none"},
            "max_tokens": 4999,
            }
        
        response_text = call_ai(payload)

        return [{
            "specification": response_text,
            "code": code,
            "request": request
        }]