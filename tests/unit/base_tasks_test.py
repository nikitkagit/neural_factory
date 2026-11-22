import os
import tempfile
import unittest

from src.fab.policy import (
    OncePolicy, InputMergePolicy, FailurePolicy
)

from src.tasks.base_tasks import ReadFileTask, ReadCodeFilesTask, WriteToFileTask, ConstantTask


class TestReadFileTask(unittest.TestCase):
    """Tests for ReadFileTask."""

    def setUp(self):
        self.task = ReadFileTask(task_id="read_file_test")
        self.temp_dir = tempfile.mkdtemp()

    def _create_temp_file(self, content: str, filename: str = "test.txt") -> str:
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

    def test_read_existing_file(self):
        """Should read file content and return it with filepath."""
        content = "Hello, World!"
        filepath = self._create_temp_file(content)
        result = self.task.run({"filepath": filepath})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], content)
        self.assertEqual(result[0]["filepath"], filepath)

    def test_read_file_with_unicode(self):
        """Should handle Unicode content correctly."""
        content = "Привет, мир! 🌍"
        filepath = self._create_temp_file(content)
        result = self.task.run({"filepath": filepath})
        self.assertEqual(result[0]["content"], content)

    def test_read_empty_file(self):
        """Should handle empty files."""
        filepath = self._create_temp_file("")
        result = self.task.run({"filepath": filepath})
        self.assertEqual(result[0]["content"], "")

    def test_missing_filepath_raises_error(self):
        """Should raise ValueError when filepath is missing."""
        with self.assertRaises(ValueError) as ctx:
            self.task.run({})
        self.assertIn("filepath", str(ctx.exception))

    def test_file_not_found_raises_error(self):
        """Should raise FileNotFoundError for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.task.run({"filepath": "/nonexistent/file.txt"})

    def test_default_policies(self):
        """Should use OncePolicy by default."""
        self.assertIsInstance(self.task.activation_policy, OncePolicy)
        self.assertIsNotNone(self.task.merge_policy)
        self.assertIsNotNone(self.task.failure_policy)


class TestReadCodeFilesTask(unittest.TestCase):
    """Tests for ReadCodeFilesTask."""

    def setUp(self):
        self.task = ReadCodeFilesTask(task_id="read_code_test")
        self.temp_dir = tempfile.mkdtemp()

    def _create_file(self, rel_path: str, content: str) -> str:
        full_path = os.path.join(self.temp_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return full_path

    def test_read_single_file(self):
        """Should read a single allowed file."""
        self._create_file("main.py", "print('hello')")
        result = self.task.run({"files": [os.path.join(self.temp_dir, "main.py")]})
        self.assertEqual(len(result), 1)
        self.assertIn("print('hello')", result[0]["code"])
        self.assertIn("main.py", result[0]["code"])

    def test_read_multiple_files(self):
        """Should read multiple files and concatenate them."""
        self._create_file("a.py", "file_a")
        self._create_file("b.py", "file_b")
        files = [
            os.path.join(self.temp_dir, "a.py"),
            os.path.join(self.temp_dir, "b.py"),
        ]
        result = self.task.run({"files": files})
        self.assertIn("file_a", result[0]["code"])
        self.assertIn("file_b", result[0]["code"])

    def test_skip_disallowed_extension(self):
        """Should skip files with extensions not in allowed set."""
        self._create_file("data.txt", "text content")
        self._create_file("script.py", "python code")
        result = self.task.run({"files": [
            os.path.join(self.temp_dir, "data.txt"),
            os.path.join(self.temp_dir, "script.py"),
        ]})
        self.assertNotIn("text content", result[0]["code"])
        self.assertIn("python code", result[0]["code"])

    def test_read_directory_recursively(self):
        """Should recursively read all allowed files in a directory."""
        self._create_file("src/main.py", "main code")
        self._create_file("src/utils/helper.py", "helper code")
        self._create_file("src/readme.md", "markdown")
        result = self.task.run({"directories": [os.path.join(self.temp_dir, "src")]})
        self.assertIn("main code", result[0]["code"])
        self.assertIn("helper code", result[0]["code"])
        self.assertNotIn("markdown", result[0]["code"])

    def test_both_files_and_directories(self):
        """Should process both explicit files and directories."""
        self._create_file("explicit.py", "explicit")
        self._create_file("dir/file.py", "from_dir")
        result = self.task.run({
            "files": [os.path.join(self.temp_dir, "explicit.py")],
            "directories": [os.path.join(self.temp_dir, "dir")],
        })
        self.assertIn("explicit", result[0]["code"])
        self.assertIn("from_dir", result[0]["code"])

    def test_missing_both_raises_error(self):
        """Should raise ValueError when neither files nor directories provided."""
        with self.assertRaises(ValueError) as ctx:
            self.task.run({})
        self.assertIn("directories", str(ctx.exception))
        self.assertIn("files", str(ctx.exception))

    def test_nonexistent_paths_ignored(self):
        """Should silently skip non-existent files and directories."""
        result = self.task.run({
            "files": ["/nonexistent/file.py"],
            "directories": ["/nonexistent/dir"],
        })
        self.assertEqual(result[0]["code"], "")

    def test_custom_allowed_extensions(self):
        """Should respect custom allowed extensions."""
        task = ReadCodeFilesTask(
            task_id="custom_ext",
            allowed_extensions={".txt", ".md"}
        )
        self._create_file("data.txt", "text")
        self._create_file("script.py", "python")
        result = task.run({"files": [
            os.path.join(self.temp_dir, "data.txt"),
            os.path.join(self.temp_dir, "script.py"),
        ]})
        self.assertIn("text", result[0]["code"])
        self.assertNotIn("python", result[0]["code"])

    def test_default_extensions(self):
        """Should use default extensions when none provided."""
        task = ReadCodeFilesTask(task_id="default_ext")
        self.assertEqual(task.allowed_extensions, ReadCodeFilesTask.DEFAULT_EXTENSIONS)

    def test_file_with_errors_replaced(self):
        """Should handle files with encoding errors gracefully."""
        filepath = os.path.join(self.temp_dir, "bad.py")
        with open(filepath, "wb") as f:
            f.write(b"valid ascii \xff\xfe invalid utf-8")
        result = self.task.run({"files": [filepath]})
        # Should not raise; errors='replace' handles bad bytes
        self.assertIn("valid ascii", result[0]["code"])

    def test_filepath_prefix_format(self):
        """Each file should be prefixed with its path."""
        self._create_file("src/module.py", "code")
        result = self.task.run({"files": [os.path.join(self.temp_dir, "src/module.py")]})
        expected_prefix = os.path.join(self.temp_dir, "src/module.py") + ":"
        self.assertIn(expected_prefix, result[0]["code"])

    def test_empty_directories_list(self):
        """Should handle empty directories list."""
        self._create_file("test.py", "content")
        result = self.task.run({
            "files": [os.path.join(self.temp_dir, "test.py")],
            "directories": [],
        })
        self.assertIn("content", result[0]["code"])


class TestWriteToFileTask(unittest.TestCase):
    """Tests for WriteToFileTask."""

    def setUp(self):
        self.task = WriteToFileTask(task_id="write_test")
        self.temp_dir = tempfile.mkdtemp()

    def test_write_to_file(self):
        """Should write content to a file and return status."""
        filepath = os.path.join(self.temp_dir, "output.txt")
        content = "Hello, World!"
        result = self.task.run({"input": content, "filepath": filepath})
        self.assertEqual(result[0]["filepath"], filepath)
        self.assertEqual(result[0]["status"], "written")
        with open(filepath, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), content)

    def test_write_creates_parent_directories(self):
        """Should create parent directories if they don't exist."""
        filepath = os.path.join(self.temp_dir, "subdir", "nested", "output.txt")
        result = self.task.run({"input": "nested content", "filepath": filepath})
        self.assertTrue(os.path.exists(filepath))
        with open(filepath, "r") as f:
            self.assertEqual(f.read(), "nested content")

    def test_missing_content_raises_error(self):
        """Should raise ValueError when content is missing."""
        with self.assertRaises(ValueError) as ctx:
            self.task.run({"filepath": "/tmp/test.txt"})
        self.assertIn("content", str(ctx.exception))

    def test_missing_filepath_raises_error(self):
        """Should raise ValueError when filepath is missing."""
        with self.assertRaises(ValueError) as ctx:
            self.task.run({"input": "content"})
        self.assertIn("filepath", str(ctx.exception))

    def test_overwrite_existing_file(self):
        """Should overwrite existing file content."""
        filepath = os.path.join(self.temp_dir, "existing.txt")
        with open(filepath, "w") as f:
            f.write("old content")
        self.task.run({"input": "new content", "filepath": filepath})
        with open(filepath, "r") as f:
            self.assertEqual(f.read(), "new content")

    def test_write_unicode_content(self):
        """Should write Unicode content correctly."""
        filepath = os.path.join(self.temp_dir, "unicode.txt")
        content = "日本語 Español العربية"
        self.task.run({"input": content, "filepath": filepath})
        with open(filepath, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), content)

    def test_write_empty_content(self):
        """Should write empty content."""
        filepath = os.path.join(self.temp_dir, "empty.txt")
        self.task.run({"input": "", "filepath": filepath})
        with open(filepath, "r") as f:
            self.assertEqual(f.read(), "")


if __name__ == "__main__":
    unittest.main()