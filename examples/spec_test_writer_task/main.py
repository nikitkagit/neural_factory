"""
Example pipeline: specification file + file list -> read code -> read spec -> write test spec -> save to file.
Or: user request + file list -> read code -> write test spec -> save to file (when --spec is not provided).

Usage:
    python -m examples.test_spec_writer_task.main --spec <spec_file> [directories/files...]
    python -m examples.test_spec_writer_task.main [directories/files...]

Graph (with --spec):
  ConstantTask (directories, files)
    ├── directories/files ──► ReadCodeFilesTask ──► code ──► SpecTestWriterTask ──► WriteToFileTask
    └── filepath ──► ReadFileTask ──► specification ──► SpecTestWriterTask

Graph (without --spec):
  ConstantTask (directories, files)
    ├── directories/files ──► ReadCodeFilesTask ──► code ──► SpecTestWriterTask ──► WriteToFileTask
    └── user request (via ConstantTask) ──► request ──► SpecTestWriterTask
"""

import sys
import os
import logging
import argparse

# Ensure the project root is on sys.path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.fab.orchestrator import Orchestrator
from src.fab.executor import InlineExecutor
from src.fab.policy import OncePolicy, AllPolicy, CartesianProductMergePolicy
from src.tasks.base_tasks import ConstantTask, ReadCodeFilesTask, WriteToFileTask, ReadFileTask
from src.tasks.ai_tasks import SpecTestWriterTask
from src.config import config


def setup_logging() -> None:
    """Configures logging based on the debug flag in config."""
    log_level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def main():
    setup_logging()

    # ------------------------------------------------------------------
    # 1. Parse command-line arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Generate test specifications from a specification file or a user request."
    )
    parser.add_argument(
        "--spec", "-s",
        type=str,
        default=None,
        help="Path to the specification file (optional). If not provided, a user request is prompted."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Directories and files to scan for code (optional)."
    )
    args = parser.parse_args()

    spec_filepath = args.spec
    directories_to_scan = []
    files_to_scan = []

    for arg in args.paths:
        if "." in arg:
            files_to_scan.append(arg)
        else:
            directories_to_scan.append(arg)

    # ------------------------------------------------------------------
    # 2. Create the orchestrator (inline executor for simplicity)
    # ------------------------------------------------------------------
    orchestrator = Orchestrator(executor_cls=InlineExecutor)

    # ------------------------------------------------------------------
    # 3. Create tasks
    # ------------------------------------------------------------------
    constant_task = ConstantTask(
        task_id="user_input",
        output=[
            {
                "directories": directories_to_scan,
                "files": files_to_scan,
            }
        ],
        activation_policy=OncePolicy(),
    )

    read_code_task = ReadCodeFilesTask(
        task_id="read_code",
        activation_policy=OncePolicy(),
    )

    test_spec_writer_task = SpecTestWriterTask(
        task_id="write_test_spec",
        activation_policy=AllPolicy(),          # needs both inputs fresh
        merge_policy=CartesianProductMergePolicy(),
    )

    write_task = WriteToFileTask(
        task_id="save_spec",
        activation_policy=OncePolicy(),
    )

    # Register common tasks
    orchestrator.add_task(constant_task)
    orchestrator.add_task(read_code_task)
    orchestrator.add_task(test_spec_writer_task)
    orchestrator.add_task(write_task)

    # ------------------------------------------------------------------
    # 4. Connect common edges
    # ------------------------------------------------------------------
    # Edge 1: user_input -> read_code  (pass directories and files)
    orchestrator.add_edge(
        src_id="user_input",
        dst_id="read_code",
        aliases={
            "directories": "directories",
            "files": "files",
        },
    )

    # Edge 2: read_code -> test_spec_writer  (pass the code)
    orchestrator.add_edge(
        src_id="read_code",
        dst_id="write_test_spec",
        aliases={
            "code": "code",
        },
    )

    # Edge 3: test_spec_writer -> save_spec  (pass the specification)
    orchestrator.add_edge(
        src_id="write_test_spec",
        dst_id="save_spec",
        aliases={
            "specification": "input",       # WriteToFileTask expects 'input'
        },
    )

    # ------------------------------------------------------------------
    # 5. Conditional logic: with --spec or without
    # ------------------------------------------------------------------
    if spec_filepath is not None:
        # --- With --spec: read specification from file ---
        if not os.path.exists(spec_filepath):
            print(f"Error: Specification file '{spec_filepath}' does not exist.")
            sys.exit(1)

        if os.path.getsize(spec_filepath) == 0:
            print(f"Error: Specification file '{spec_filepath}' is empty.")
            sys.exit(1)

        read_spec_task = ReadFileTask(
            task_id="read_spec",
            activation_policy=OncePolicy(),
        )
        orchestrator.add_task(read_spec_task)

        # Edge: user_input -> read_spec (pass the spec file path)
        orchestrator.add_edge(
            src_id="user_input",
            dst_id="read_spec",
            aliases={
                "filepath": "filepath",  # will be overridden by spec_path
            },
        )

        # Edge: read_spec -> test_spec_writer (pass the specification)
        orchestrator.add_edge(
            src_id="read_spec",
            dst_id="write_test_spec",
            aliases={
                "content": "specification",
            },
        )

        # Inject the spec file path
        spec_path_task = ConstantTask(
            task_id="spec_path",
            output=[{"filepath": spec_filepath}],
            activation_policy=OncePolicy(),
        )
        orchestrator.add_task(spec_path_task)

        orchestrator.add_edge(
            src_id="spec_path",
            dst_id="read_spec",
            aliases={
                "filepath": "filepath",
            },
        )

    else:
        # --- Without --spec: prompt user for request ---
        user_request = input("Enter request: ")

        request_task = ConstantTask(
            task_id="user_request",
            output=[{"request": user_request}],
            activation_policy=OncePolicy(),
        )
        orchestrator.add_task(request_task)

        # Edge: user_request -> test_spec_writer (pass the request)
        orchestrator.add_edge(
            src_id="user_request",
            dst_id="write_test_spec",
            aliases={
                "request": "request",
            },
        )

    # ------------------------------------------------------------------
    # 6. Inject the output file path
    # ------------------------------------------------------------------
    output_path_task = ConstantTask(
        task_id="output_path",
        output=[{"filepath": "output/test_spec.md"}],
        activation_policy=OncePolicy(),
    )
    orchestrator.add_task(output_path_task)

    orchestrator.add_edge(
        src_id="output_path",
        dst_id="save_spec",
        aliases={
            "filepath": "filepath",
        },
    )

    # ------------------------------------------------------------------
    # 7. Run the pipeline
    # ------------------------------------------------------------------
    print("Starting pipeline...")
    report = orchestrator.run()

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    print("\n=== Pipeline finished ===")
    for tid, info in report["tasks"].items():
        print(f"  {tid}: status={info['status']}, version={info['version']}, "
              f"run_count={info['run_count']}, error={info['last_error']}")

    # Check if the output file was created
    output_path = "output/test_spec.md"
    if os.path.exists(output_path):
        print(f"\nTest specification written to {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            print(f"\n--- First 500 characters ---\n{f.read()[:500]}")
    else:
        print("\nOutput file was not created — check for errors above.")


if __name__ == "__main__":
    main()