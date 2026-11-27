import os

from src.tasks.python_tasks import SnoopInjectorProcessTask


class TestSnoopInjectorProcessTask:
    """Tests for SnoopInjectorProcessTask behavior through its public API."""

    def test_run_adds_snoop_import_and_decorator(self) -> None:
        task = SnoopInjectorProcessTask(task_id="process_task")
        result = task.run({
            "files": [{"file_content": "def foo():\n    pass", "file_path": "/tmp/test.py"}],
            "clear": False
        })

        assert isinstance(result, list)
        assert len(result) == 1

        output = result[0]
        assert output["modified_files"] == ["/tmp/test.py"]
        assert output["total_files_scanned"] == 1
        assert len(output["modified_contents"]) == 1

        modified = output["modified_contents"][0]
        assert modified["file_path"] == "/tmp/test.py"
        assert "from snoop import snoop" in modified["file_content"]
        assert "@snoop" in modified["file_content"]

        # Verify @snoop appears before def foo()
        content = modified["file_content"]
        snoop_pos = content.index("@snoop")
        def_pos = content.index("def foo()")
        assert snoop_pos < def_pos

    def test_run_does_not_add_duplicate_import(self) -> None:
        task = SnoopInjectorProcessTask(task_id="process_task")
        result = task.run({
            "files": [{
                "file_content": "from snoop import snoop\ndef foo():\n    pass",
                "file_path": "/tmp/test.py"
            }],
            "clear": False
        })

        content = result[0]["modified_contents"][0]["file_content"]
        import_count = content.count("from snoop import snoop")
        assert import_count == 1

    def test_run_does_not_add_duplicate_decorator(self) -> None:
        task = SnoopInjectorProcessTask(task_id="process_task")
        result = task.run({
            "files": [{
                "file_content": "@snoop\ndef foo():\n    pass",
                "file_path": "/tmp/test.py"
            }],
            "clear": False
        })

        content = result[0]["modified_contents"][0]["file_content"]
        decorator_count = content.count("@snoop")
        assert decorator_count == 1

    def test_run_adds_import_after_shebang_and_future(self) -> None:
        task = SnoopInjectorProcessTask(task_id="process_task")
        result = task.run({
            "files": [{
                "file_content": "#!/usr/bin/env python\nfrom __future__ import annotations\ndef foo():\n    pass",
                "file_path": "/tmp/test.py"
            }],
            "clear": False
        })

        content = result[0]["modified_contents"][0]["file_content"]
        lines = content.splitlines()
        shebang_idx = lines.index("#!/usr/bin/env python")
        future_idx = lines.index("from __future__ import annotations")
        snoop_idx = lines.index("from snoop import snoop")
        assert shebang_idx < future_idx < snoop_idx

    def test_run_adds_decorator_with_correct_indent(self) -> None:
        task = SnoopInjectorProcessTask(task_id="process_task")
        result = task.run({
            "files": [{
                "file_content": "class A:\n    def bar(self):\n        pass",
                "file_path": "/tmp/test.py"
            }],
            "clear": False
        })

        content = result[0]["modified_contents"][0]["file_content"]
        assert "    @snoop" in content
        assert "    def bar(self):" in content

    def test_run_removes_import_and_decorator(self) -> None:
        task = SnoopInjectorProcessTask(task_id="process_task")
        result = task.run({
            "files": [{
                "file_content": "from snoop import snoop\n@snoop\ndef foo():\n    pass",
                "file_path": "/tmp/test.py"
            }],
            "clear": True
        })

        content = result[0]["modified_contents"][0]["file_content"]
        assert "from snoop import snoop" not in content
        assert "@snoop" not in content
        assert "def foo():\n    pass" in content

    def test_run_removes_only_snoop_decorator_before_def(self) -> None:
        task = SnoopInjectorProcessTask(task_id="process_task")
        result = task.run({
            "files": [{
                "file_content": "@snoop\ndef foo():\n    pass\n@snoop\nx = 1",
                "file_path": "/tmp/test.py"
            }],
            "clear": True
        })

        content = result[0]["modified_contents"][0]["file_content"]
        # The @snoop before def should be removed, but @snoop before x = 1 should remain
        assert "@snoop" in content
        # Verify the remaining @snoop is before x = 1, not before def
        lines = content.splitlines()
        snoop_indices = [i for i, line in enumerate(lines) if line.strip() == "@snoop"]
        assert len(snoop_indices) == 1
        snoop_line = snoop_indices[0]
        assert lines[snoop_line + 1].strip() == "x = 1"

    def test_run_does_not_leave_extra_blank_lines(self) -> None:
        task = SnoopInjectorProcessTask(task_id="process_task")
        result = task.run({
            "files": [{
                "file_content": "from snoop import snoop\n\n\n@snoop\ndef foo():\n    pass",
                "file_path": "/tmp/test.py"
            }],
            "clear": True
        })

        content = result[0]["modified_contents"][0]["file_content"]
        lines = content.splitlines()
        max_consecutive_blanks = 0
        current_blanks = 0
        for line in lines:
            if line.strip() == "":
                current_blanks += 1
                max_consecutive_blanks = max(max_consecutive_blanks, current_blanks)
            else:
                current_blanks = 0
        assert max_consecutive_blanks <= 1

    def test_run_no_changes_when_nothing_to_remove(self) -> None:
        task = SnoopInjectorProcessTask(task_id="process_task")
        result = task.run({
            "files": [{
                "file_content": "def foo():\n    pass",
                "file_path": "/tmp/test.py"
            }],
            "clear": True
        })

        output = result[0]
        assert output["modified_files"] == []
        assert output["modified_contents"] == []

    def test_run_no_changes_when_already_has_snoop(self) -> None:
        task = SnoopInjectorProcessTask(task_id="process_task")
        result = task.run({
            "files": [{
                "file_content": "from snoop import snoop\n@snoop\ndef foo():\n    pass",
                "file_path": "/tmp/test.py"
            }],
            "clear": False
        })

        output = result[0]
        assert output["modified_files"] == []
        assert output["modified_contents"] == []

    def test_run_does_not_write_to_disk(self) -> None:
        # Create a temporary file with known content
        tmp_path = "/tmp/test_snoop_injector.txt"
        original_content = "x = 1"
        with open(tmp_path, "w") as f:
            f.write(original_content)

        try:
            task = SnoopInjectorProcessTask(task_id="process_task")
            task.run({
                "files": [{"file_content": "x = 1", "file_path": tmp_path}],
                "clear": False
            })

            # Verify file on disk is unchanged
            with open(tmp_path, "r") as f:
                disk_content = f.read()
            assert disk_content == original_content
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_run_with_multiple_files(self) -> None:
        task = SnoopInjectorProcessTask(task_id="process_task")
        result = task.run({
            "files": [
                {"file_content": "def a():\n    pass", "file_path": "/tmp/a.py"},
                {"file_content": "def b():\n    pass", "file_path": "/tmp/b.py"}
            ],
            "clear": False
        })

        output = result[0]
        assert set(output["modified_files"]) == {"/tmp/a.py", "/tmp/b.py"}
        assert output["total_files_scanned"] == 2
        assert len(output["modified_contents"]) == 2

    def test_run_with_clear_defaults_to_false(self) -> None:
        task = SnoopInjectorProcessTask(task_id="process_task")
        result = task.run({
            "files": [{"file_content": "def foo():\n    pass", "file_path": "/tmp/test.py"}]
        })

        output = result[0]
        assert "/tmp/test.py" in output["modified_files"]

    def test_run_handles_empty_files_list(self) -> None:
        task = SnoopInjectorProcessTask(task_id="process_task")
        result = task.run({"files": [], "clear": False})

        assert result == [{"modified_files": [], "total_files_scanned": 0, "modified_contents": []}]