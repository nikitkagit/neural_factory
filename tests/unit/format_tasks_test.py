import os
import tempfile
import shutil
import pytest
from src.tasks.format_tasks import ExtractSpecFilesTask


class TestExtractSpecFilesTask:

    def test_extract_paths_from_specification_with_various_formats(self):
        task = ExtractSpecFilesTask(task_id="test_task")
        spec = (
            "## Необходимые файлы\n"
            "/etc/hosts\n"
            "src/main.py\n"
            '"docs/readme.md"\n'
            "- lib/utils.py\n"
            "1. config/settings.json\n"
            "## Изменяемые файлы\n"
            "src/app.py\n"
        )

        # Create temporary project directory
        tmp_dir = tempfile.mkdtemp()
        try:
            # Create files that exist (skip /etc/hosts as it's absolute and exists on most systems)
            os.makedirs(os.path.join(tmp_dir, "src"), exist_ok=True)
            os.makedirs(os.path.join(tmp_dir, "docs"), exist_ok=True)
            os.makedirs(os.path.join(tmp_dir, "lib"), exist_ok=True)
            os.makedirs(os.path.join(tmp_dir, "config"), exist_ok=True)
            open(os.path.join(tmp_dir, "src", "main.py"), "w").close()
            open(os.path.join(tmp_dir, "docs", "readme.md"), "w").close()
            open(os.path.join(tmp_dir, "lib", "utils.py"), "w").close()
            open(os.path.join(tmp_dir, "config", "settings.json"), "w").close()
            open(os.path.join(tmp_dir, "src", "app.py"), "w").close()

            result = task.run({"specification": spec, "project_path": tmp_dir})
            file_paths = result[0]["file_paths"]

            # Check that /etc/hosts is included if it exists
            expected_paths = [
                os.path.normpath(os.path.join(tmp_dir, "config", "settings.json")),
                os.path.normpath(os.path.join(tmp_dir, "docs", "readme.md")),
                os.path.normpath(os.path.join(tmp_dir, "lib", "utils.py")),
                os.path.normpath(os.path.join(tmp_dir, "src", "app.py")),
                os.path.normpath(os.path.join(tmp_dir, "src", "main.py")),
            ]
            if os.path.isfile("/etc/hosts"):
                expected_paths.insert(0, "/etc/hosts")

            assert file_paths == sorted(expected_paths), f"Expected {sorted(expected_paths)}, got {file_paths}"
        finally:
            shutil.rmtree(tmp_dir)

    def test_only_extracts_from_required_sections(self):
        task = ExtractSpecFilesTask(task_id="test_task")
        spec = (
            "## Необходимые файлы\n"
            "src/main.py\n"
            "## Изменяемые файлы\n"
            "src/app.py\n"
            "## Описание\n"
            "src/other.py\n"
        )

        tmp_dir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmp_dir, "src"), exist_ok=True)
            open(os.path.join(tmp_dir, "src", "main.py"), "w").close()
            open(os.path.join(tmp_dir, "src", "app.py"), "w").close()
            open(os.path.join(tmp_dir, "src", "other.py"), "w").close()

            result = task.run({"specification": spec, "project_path": tmp_dir})
            file_paths = result[0]["file_paths"]

            expected = sorted([
                os.path.normpath(os.path.join(tmp_dir, "src", "main.py")),
                os.path.normpath(os.path.join(tmp_dir, "src", "app.py")),
            ])
            assert file_paths == expected, f"Expected {expected}, got {file_paths}"
        finally:
            shutil.rmtree(tmp_dir)

    def test_filters_non_existent_files(self):
        task = ExtractSpecFilesTask(task_id="test_task")
        spec = (
            "## Необходимые файлы\n"
            "exists.py\n"
            "missing.py\n"
        )

        tmp_dir = tempfile.mkdtemp()
        try:
            open(os.path.join(tmp_dir, "exists.py"), "w").close()

            result = task.run({"specification": spec, "project_path": tmp_dir})
            file_paths = result[0]["file_paths"]

            expected = [os.path.normpath(os.path.join(tmp_dir, "exists.py"))]
            assert file_paths == expected, f"Expected {expected}, got {file_paths}"
        finally:
            shutil.rmtree(tmp_dir)

    def test_removes_duplicate_paths(self):
        task = ExtractSpecFilesTask(task_id="test_task")
        spec = (
            "## Необходимые файлы\n"
            "src/file.py\n"
            "## Изменяемые файлы\n"
            "src/file.py\n"
        )

        tmp_dir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmp_dir, "src"), exist_ok=True)
            open(os.path.join(tmp_dir, "src", "file.py"), "w").close()

            result = task.run({"specification": spec, "project_path": tmp_dir})
            file_paths = result[0]["file_paths"]

            expected = [os.path.normpath(os.path.join(tmp_dir, "src", "file.py"))]
            assert file_paths == expected, f"Expected {expected}, got {file_paths}"
            assert len(file_paths) == 1, "Duplicate path should appear only once"
        finally:
            shutil.rmtree(tmp_dir)

    def test_sorts_paths_lexicographically(self):
        task = ExtractSpecFilesTask(task_id="test_task")
        spec = (
            "## Необходимые файлы\n"
            "z.py\n"
            "a.py\n"
            "m.py\n"
        )

        tmp_dir = tempfile.mkdtemp()
        try:
            for f in ["z.py", "a.py", "m.py"]:
                open(os.path.join(tmp_dir, f), "w").close()

            result = task.run({"specification": spec, "project_path": tmp_dir})
            file_paths = result[0]["file_paths"]

            expected = sorted([
                os.path.normpath(os.path.join(tmp_dir, "a.py")),
                os.path.normpath(os.path.join(tmp_dir, "m.py")),
                os.path.normpath(os.path.join(tmp_dir, "z.py")),
            ])
            assert file_paths == expected, f"Expected {expected}, got {file_paths}"
        finally:
            shutil.rmtree(tmp_dir)

    def test_raises_value_error_when_specification_missing(self):
        task = ExtractSpecFilesTask(task_id="test_task")
        with pytest.raises(ValueError, match="Missing required input field: 'specification'"):
            task.run({"project_path": "/tmp"})

    def test_raises_value_error_when_project_path_missing(self):
        task = ExtractSpecFilesTask(task_id="test_task")
        with pytest.raises(ValueError, match="Missing required input field: 'project_path'"):
            task.run({"specification": "## Необходимые файлы\n- file.txt"})

    def test_returns_empty_list_when_sections_absent_or_empty(self):
        task = ExtractSpecFilesTask(task_id="test_task")
        spec = "## Описание\nSome text without file sections."

        tmp_dir = tempfile.mkdtemp()
        try:
            result = task.run({"specification": spec, "project_path": tmp_dir})
            file_paths = result[0]["file_paths"]
            assert file_paths == [], f"Expected empty list, got {file_paths}"
        finally:
            shutil.rmtree(tmp_dir)

    def test_converts_relative_paths_using_project_path(self):
        task = ExtractSpecFilesTask(task_id="test_task")
        spec = "## Необходимые файлы\nsrc/file.py\n"

        tmp_dir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmp_dir, "src"), exist_ok=True)
            open(os.path.join(tmp_dir, "src", "file.py"), "w").close()

            result = task.run({"specification": spec, "project_path": tmp_dir})
            file_paths = result[0]["file_paths"]

            expected = [os.path.normpath(os.path.join(tmp_dir, "src", "file.py"))]
            assert file_paths == expected, f"Expected {expected}, got {file_paths}"
        finally:
            shutil.rmtree(tmp_dir)