import subprocess
import os
import re
from typing import Any, Dict, List, Optional, Set

from src.fab.task import Task, TaskId
from src.fab.policy import (
    ActivationPolicy, OncePolicy,
    InputMergePolicy, FailurePolicy
)


class RunPythonTask(Task):
    """
    Executes a Python script at the given filepath.
    Raises an exception on non-zero return code to trigger FailurePolicy.
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
        filepath = inputs.get("filepath")
        if filepath is None:
            raise ValueError("Input 'filepath' is required to run a Python script.")

        result = subprocess.run(
            ["python", filepath],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"Python script failed with code {result.returncode}.\n"
                f"Stderr: {result.stderr}\nStdout: {result.stdout}"
            )

        return [{
            "filepath": filepath,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }]


class RunPytestTask(Task):
    """
    Worker that runs pytest on a specified path or module.
    Raises an exception if tests fail to trigger FailurePolicy.
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
        filepath = inputs.get("filepath", ".")

        cmd = ["pytest", filepath]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Pytest failed with code {result.returncode}.\n"
                f"Stdout:\n{result.stdout}\n"
                f"Stderr:\n{result.stderr}"
            )

        return [{
            "filepath": filepath,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }]


class SnoopInjectorProcessTask(Task):
    """
    Processes loaded Python files: adds or removes snoop import and @snoop decorator.
    If clear is False (default): adds 'from snoop import snoop' and '@snoop' before each function.
    If clear is True: removes all snoop imports and @snoop decorators.
    Does NOT write changes to disk — returns modified contents in the result.
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
        files: List[Dict[str, Any]] = inputs.get("files", [])
        clear: bool = inputs.get("clear", False)

        modified_files: List[str] = []
        modified_contents: List[Dict[str, Any]] = []
        total_files_scanned = len(files)

        for file_entry in files:
            file_content: str = file_entry.get("file_content", "")
            file_path: str = file_entry.get("file_path", "")

            if clear:
                new_content = self._remove_snoop(file_content)
            else:
                new_content = self._add_snoop(file_content)

            if new_content != file_content:
                modified_files.append(file_path)
                modified_contents.append({
                    "file_path": file_path,
                    "file_content": new_content
                })

        return [{
            "modified_files": modified_files,
            "total_files_scanned": total_files_scanned,
            "modified_contents": modified_contents
        }]

    def _add_snoop(self, content: str) -> str:
        """
        Adds 'from snoop import snoop' if not present, and '@snoop' decorator
        before each function definition if not already present.
        Import is added after shebang and __future__ imports.
        """
        lines = content.splitlines(keepends=True)

        # Check if snoop import already exists
        has_snoop_import = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import snoop") or stripped.startswith("from snoop"):
                has_snoop_import = True
                break

        # Find insertion point for import: after shebang and __future__ imports
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#!") or stripped.startswith("from __future__"):
                insert_idx = i + 1
            else:
                break

        if not has_snoop_import:
            lines.insert(insert_idx, "from snoop import snoop\n")

        # Add @snoop decorator before each function definition if not present
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            # Check if this line is a function definition
            if re.match(r'^(\s*)def\s+', stripped):
                match = re.match(r'^(\s*)', line)
                indent = match.group(1) if match else ''
                # Check if previous non-empty line is @snoop
                prev_line = ""
                for j in range(len(new_lines) - 1, -1, -1):
                    prev_candidate = new_lines[j]
                    if prev_candidate.strip():
                        prev_line = prev_candidate.strip()
                        break
                if prev_line != "@snoop":
                    new_lines.append(f"{indent}@snoop\n")
            new_lines.append(line)
            i += 1

        return "".join(new_lines)

    def _remove_snoop(self, content: str) -> str:
        """
        Removes all lines containing 'import snoop' or 'from snoop',
        and removes all '@snoop' lines (with any indentation) that are
        directly before a 'def' line. Does not leave extra blank lines.
        """
        lines = content.splitlines(keepends=True)

        # First pass: remove import lines
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import snoop") or stripped.startswith("from snoop"):
                continue
            filtered_lines.append(line)

        # Second pass: remove @snoop decorator lines before def
        result_lines = []
        i = 0
        while i < len(filtered_lines):
            line = filtered_lines[i]
            stripped = line.strip()
            # Check if this line is @snoop (with any indentation)
            if re.match(r'^\s*@snoop\s*$', stripped):
                # Check if next non-empty line is a def
                next_idx = i + 1
                while next_idx < len(filtered_lines):
                    next_stripped = filtered_lines[next_idx].strip()
                    if next_stripped:
                        if re.match(r'^\s*def\s+', next_stripped):
                            # Skip this @snoop line
                            i += 1
                            break
                        else:
                            # Not before a def, keep it
                            result_lines.append(line)
                            i += 1
                            break
                    else:
                        next_idx += 1
                else:
                    # No non-empty line after, keep it
                    result_lines.append(line)
                    i += 1
            else:
                result_lines.append(line)
                i += 1

        # Remove consecutive blank lines left after deletions
        final_lines = []
        prev_blank = False
        for line in result_lines:
            is_blank = line.strip() == ""
            if is_blank and prev_blank:
                continue
            final_lines.append(line)
            prev_blank = is_blank

        return "".join(final_lines)