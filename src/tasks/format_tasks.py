import os
from typing import Any, Dict, List, Optional, Set

from fab.task import Task
from fab.types import TaskId
from fab.policy import ActivationPolicy, OncePolicy, InputMergePolicy, FailurePolicy


class ExtractSpecFilesTask(Task):
    """
    Task that extracts file paths from specification sections
    "Необходимые файлы" and "Изменяемые файлы" (case-insensitive).
    Returns a list of unique, sorted, existing absolute file paths.
    """

    def __init__(
        self,
        task_id: TaskId,
        activation_policy: ActivationPolicy = OncePolicy(),
        merge_policy: Optional[InputMergePolicy] = None,
        failure_policy: Optional[FailurePolicy] = None,
    ):
        super().__init__(
            task_id=task_id,
            activation_policy=activation_policy,
            merge_policy=merge_policy,
            failure_policy=failure_policy,
        )

    def run(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Extract required fields
        specification = inputs.get("specification")
        if specification is None:
            raise ValueError("Missing required input field: 'specification'")

        project_path = inputs.get("project_path")
        if project_path is None:
            raise ValueError("Missing required input field: 'project_path'")

        # Normalize project_path
        project_path = os.path.normpath(project_path)

        # Split specification into lines
        lines = specification.splitlines()

        # Find section boundaries for "Необходимые файлы" and "Изменяемые файлы"
        # A section starts with a line containing the section header (case-insensitive)
        # and ends at the next line starting with "##" or end of text.
        section_headers = ["необходимые файлы", "изменяемые файлы"]
        sections: List[List[str]] = []
        current_section: List[str] = []
        in_target_section = False

        for line in lines:
            stripped = line.strip()
            # Check if this line starts a new section (##)
            if stripped.startswith("##"):
                # Check if this is a target section header
                lower_stripped = stripped.lower()
                is_target = any(header in lower_stripped for header in section_headers)
                if is_target:
                    # Start a new target section
                    if current_section is not None:
                        sections.append(current_section)
                    current_section = []
                    in_target_section = True
                else:
                    # Non-target section ends any current target section
                    if current_section is not None:
                        sections.append(current_section)
                        current_section = []
                    in_target_section = False
            else:
                if in_target_section and current_section is not None:
                    current_section.append(line)

        # Append the last section if it was a target section
        if current_section is not None:
            sections.append(current_section)

        # Extract file paths from sections
        paths: Set[str] = set()

        for section_lines in sections:
            for line in section_lines:
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                # Remove list markers: -, *, digits with dot, etc.
                # Pattern: optional whitespace, then list marker, then whitespace, then path
                # List markers can be: -, *, +, digits followed by dot, letters followed by dot
                import re
                # Match optional list marker at start: -, *, +, \d+., \w+.
                marker_match = re.match(r'^(\s*[-*+]|\s*\d+\.|\s*\w+\.)\s+', stripped_line)
                if marker_match:
                    path_str = stripped_line[marker_match.end():]
                else:
                    path_str = stripped_line

                # Remove surrounding quotes (single or double)
                path_str = path_str.strip()
                if (path_str.startswith('"') and path_str.endswith('"')) or \
                   (path_str.startswith("'") and path_str.endswith("'")):
                    path_str = path_str[1:-1].strip()

                if not path_str:
                    continue

                # Convert to absolute path if relative
                if not path_str.startswith('/'):
                    path_str = os.path.abspath(os.path.join(project_path, path_str))
                else:
                    path_str = os.path.normpath(path_str)

                # Normalize path
                path_str = os.path.normpath(path_str)

                # Check if file exists
                if os.path.isfile(path_str):
                    paths.add(path_str)

        # Sort paths lexicographically
        sorted_paths = sorted(paths)

        return [{"file_paths": sorted_paths}]