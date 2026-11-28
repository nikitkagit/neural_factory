import re

from typing import Any, Dict, List, Optional

from src.fab.task import Task, TaskId
from src.prompt import (
    Prompt, make_request_prompt, make_code_prompt, make_specification_prompt,
    make_previous_specification_prompt,
    SPEC_FORMAT, GEN_SPEC, REGEN_SPEC, NO_CODE, GEN_SPEC_FOR_TESTS, CODE_SEARCH_REPLACE_ONLY
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

        if code is None or not isinstance(code, str):
            raise ValueError("Input must contain 'code' (str).")
        if not request or not isinstance(request, str):
            raise ValueError("Input must contain 'request' (str).")

        # Build the combined prompt with code and request
        combined = SPEC_FORMAT + NO_CODE
        combined += make_code_prompt(code) + make_request_prompt(request)

        # Add previous specification if present and is a string
        if previous_specification is not None and isinstance(previous_specification, str):
            combined += REGEN_SPEC + make_previous_specification_prompt(previous_specification)
        else:
            combined += GEN_SPEC

        messages = [
            {"role": "user", "content": str(combined)}
        ]

        payload = {
            "model": "deepseek/deepseek-v4-flash",
            "messages": messages,
            "reasoning": {"effort": "none"},
            "max_tokens": 4999,
            "temperature": 1.0,
            "provider": {
                "only": ["baidu", "streamlake"]
            }
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
            "temperature": 0.03,
            "provider": {
                "only": ["baidu", "streamlake"]
            }
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


class SpecToScriptTask(Task):
    """
    Generates a script according to a specification using an AI agent.
    Accepts 'specification' (str) and 'code' (str) in inputs.
    Returns a list of dicts: [{"filepath": str, "search": str, "replace": str}, ...].
    The agent's response must strictly follow the CODE_SEARCH_REPLACE_ONLY format.
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
        code = inputs.get("code")

        if specification is None or not isinstance(specification, str):
            raise ValueError("Input must contain 'specification' (str).")
        if code is None or not isinstance(code, str):
            raise ValueError("Input must contain 'code' (str).")

        # Build the combined prompt
        combined_prompt = CODE_SEARCH_REPLACE_ONLY + make_code_prompt(code) + make_specification_prompt(specification)

        payload = {"messages": [{"role": "user", "content": str(combined_prompt)}]}
        response_text = self._ai_client.call(payload)

        # Parse the response: extract all blocks in CODE_SEARCH_REPLACE_ONLY format
        triples = self._parse_search_replace_blocks(response_text)

        return triples

    def _parse_search_replace_blocks(self, text: str) -> List[Dict[str, str]]:
        """
        Parses a response text and extracts all blocks in the format:
        [</path/to/code/file>]
        [SEARCH]
        <exact snippet of the old code>
        [REPLACE]
        <new code snippet>

        Returns a list of dicts with keys: filepath, search, replace.
        If no valid blocks are found, returns an empty list.
        """
        blocks: List[Dict[str, str]] = []

        # Split text into individual blocks by filepath headers
        # Pattern matches [</path/to/file>] optionally preceded by whitespace
        block_pattern = re.compile(
            r'\[\s*(/[^\]]+)\s*\]\s*'  # [</path/to/file>]
            r'(.*?)(?=\s*\[\s*/[^\]]+\s*\]|\s*$)',  # content until next block or end
            re.DOTALL
        )

        for match in block_pattern.finditer(text):
            filepath = match.group(1).strip()
            content = match.group(2).strip()

            # Parse search/replace within this block
            search_match = re.search(r'\[SEARCH\]\s*(.*?)\s*\[REPLACE\]\s*(.*)', content, re.DOTALL)
            if search_match:
                search = search_match.group(1).strip()
                replace = search_match.group(2).strip()
                if search and replace:
                    blocks.append({
                        "filepath": filepath,
                        "search": search,
                        "replace": replace
                    })

        return blocks
