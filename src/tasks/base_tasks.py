import os
from typing import Any, Dict, List, Optional, Set

from src.fab.task import Task, TaskId
from src.fab.policy import (
    ActivationPolicy, OncePolicy, 
    InputMergePolicy, FailurePolicy
)

class ReadFileTask(Task):
    """Worker that reads a file and returns its text."""
    
    def __init__(
        self, 
        task_id: TaskId, 
        activation_policy: ActivationPolicy = OncePolicy(),
        merge_policy: Optional[InputMergePolicy] = None,
        failure_policy: Optional[FailurePolicy] = None,
    ):
        super().__init__(task_id, activation_policy, merge_policy, failure_policy)

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        filepath = inputs.get("filepath")
        if filepath is None:
            raise ValueError("Input 'filepath' is required to read a file.")
            
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return [{"content": content, "filepath": filepath}]


class ReadCodeFilesTask(Task):
    """
    Reads multiple directories and files, concatenates their contents into a single string.
    Each file is prefixed with its relative path and filename (e.g., "PATH/TO/FILE.PY:").
    Supports a whitelist of file extensions to include.
    """

    DEFAULT_EXTENSIONS: Set[str] = {
        ".cpp", ".hpp", ".h",                     # C++
        ".ex", ".exs",                            # Elixir
        ".js",                                    # JavaScript
        ".py",                                    # Python
    }

    def __init__(
        self,
        task_id: TaskId,
        allowed_extensions: Optional[Set[str]] = None,
        activation_policy: ActivationPolicy = OncePolicy(),
        merge_policy: Optional[InputMergePolicy] = None,
        failure_policy: Optional[FailurePolicy] = None,
    ):
        super().__init__(task_id, activation_policy, merge_policy, failure_policy)
        self.allowed_extensions = allowed_extensions or self.DEFAULT_EXTENSIONS

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        directories: List[str] = inputs.get("directories", [])
        files: List[str] = inputs.get("files", [])

        if not directories and not files:
            raise ValueError("At least one of 'directories' or 'files' must be provided.")

        code_parts: List[str] = []

        # Process explicitly listed files
        for filepath in files:
            if not os.path.isfile(filepath):
                continue
            ext = os.path.splitext(filepath)[1].lower()
            if ext not in self.allowed_extensions:
                continue
            content = self._read_file(filepath)
            code_parts.append(f"{filepath}:\n{content}")

        # Process directories recursively
        for directory in directories:
            if not os.path.isdir(directory):
                continue
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in self.allowed_extensions:
                        continue
                    filepath = os.path.join(root, filename)
                    content = self._read_file(filepath)
                    code_parts.append(f"{filepath}:\n{content}")

        code = "\n\n".join(code_parts)

        return [{"code": code}]

    def _read_file(self, filepath: str) -> str:
        """Reads a file and returns its contents as a string."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


class WriteToFileTask(Task):
    """Worker that writes text content to a file."""

    def __init__(
        self,
        task_id: TaskId,
        activation_policy: ActivationPolicy = OncePolicy(),
        merge_policy: Optional[InputMergePolicy] = None,
        failure_policy: Optional[FailurePolicy] = None,
    ):
        super().__init__(task_id, activation_policy, merge_policy, failure_policy)

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = inputs.get("input")
        filepath = inputs.get("filepath")

        if content is None:
            raise ValueError("Input 'content' is required to write a file.")
        if filepath is None:
            raise ValueError("Input 'filepath' is required to write a file.")

        parent_dir = os.path.dirname(os.path.abspath(filepath))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return [{"filepath": filepath, "status": "written"}]


class ConstantTextTask(Task):
    """
    Emits a fixed text provided in the constructor.
    Uses OncePolicy, which activates immediately without requiring any input edges.
    """
    
    def __init__(
        self, 
        task_id: TaskId, 
        text: str, 
        activation_policy: ActivationPolicy = OncePolicy(),
        merge_policy: Optional[InputMergePolicy] = None,
        failure_policy: Optional[FailurePolicy] = None,
    ):
        super().__init__(task_id, activation_policy, merge_policy, failure_policy)
        self.text = text

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"content": self.text}]