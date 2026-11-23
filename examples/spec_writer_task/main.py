"""
Example pipeline: user request + file list -> read code -> write spec -> save to file.

Graph:
  ConstantTask (request, file lists)
    ├── request ──────────────────────────► SpecWriterTask ──► WriteToFileTask
    └── directories/files ──► ReadCodeFilesTask ──► code ──► SpecWriterTask

using: python -m examples.spec_writer_task.main src/fab file1.py src/tasks
"""

import sys
import os
import logging

# Ensure the project root is on sys.path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.fab.orchestrator import Orchestrator
from src.fab.executor import InlineExecutor
from src.fab.policy import OncePolicy, AllPolicy, CartesianProductMergePolicy
from src.tasks.base_tasks import ConstantTask, ReadCodeFilesTask, WriteToFileTask
from src.tasks.ai_tasks import SpecWriterTask
from src.config import config
from src.utils import AiClient


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
    # 1. Define the user input
    # ------------------------------------------------------------------
    user_request = input("Enter rewuest: ")

    args = sys.argv[0:]

    directories_to_scan = []
    files_to_scan = []

    for arg in args[1:]:
        if "." in arg:
            files_to_scan.append(arg)
        else:
            directories_to_scan.append(arg)

    # ------------------------------------------------------------------
    # 2. Create AiClient instance
    # ------------------------------------------------------------------
    ai_client = AiClient()

    # ------------------------------------------------------------------
    # 3. Create the orchestrator (inline executor for simplicity)
    # ------------------------------------------------------------------
    orchestrator = Orchestrator(executor_cls=InlineExecutor)

    # ------------------------------------------------------------------
    # 4. Create tasks
    # ------------------------------------------------------------------
    constant_task = ConstantTask(
        task_id="user_input",
        output=[
            {
                "request": user_request,
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

    spec_writer_task = SpecWriterTask(
        task_id="write_spec",
        ai_client=ai_client,
        activation_policy=AllPolicy(),          # needs both inputs fresh
        merge_policy=CartesianProductMergePolicy(),
    )

    write_task = WriteToFileTask(
        task_id="save_spec",
        activation_policy=OncePolicy(),
    )

    # ------------------------------------------------------------------
    # 5. Register tasks in the graph
    # ------------------------------------------------------------------
    orchestrator.add_task(constant_task)
    orchestrator.add_task(read_code_task)
    orchestrator.add_task(spec_writer_task)
    orchestrator.add_task(write_task)

    # ------------------------------------------------------------------
    # 6. Connect edges
    #
    #   constant_task ──request──────────────► spec_writer_task
    #   constant_task ──directories/files────► read_code_task
    #   read_code_task ──code────────────────► spec_writer_task
    #   spec_writer_task ──specification─────► write_task
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

    # Edge 2: user_input -> spec_writer  (pass the request)
    orchestrator.add_edge(
        src_id="user_input",
        dst_id="write_spec",
        aliases={
            "request": "request",
        },
    )

    # Edge 3: read_code -> spec_writer  (pass the code)
    orchestrator.add_edge(
        src_id="read_code",
        dst_id="write_spec",
        aliases={
            "code": "code",
        },
    )

    # Edge 4: spec_writer -> save_spec  (pass the specification and request)
    orchestrator.add_edge(
        src_id="write_spec",
        dst_id="save_spec",
        aliases={
            "specification": "input",       # WriteToFileTask expects 'input'
        },
    )

    # ------------------------------------------------------------------
    # 7. Set output filepath to output/spec.md
    # ------------------------------------------------------------------
    # We add a second constant task that only provides the output path.
    # This is a clean way to inject a fixed filepath without modifying
    # the main constant task.
    path_task = ConstantTask(
        task_id="output_path",
        output=[{"filepath": "output/spec.md"}],
        activation_policy=OncePolicy(),
    )
    orchestrator.add_task(path_task)

    orchestrator.add_edge(
        src_id="output_path",
        dst_id="save_spec",
        aliases={
            "filepath": "filepath",
        },
    )

    # ------------------------------------------------------------------
    # 8. Run the pipeline
    # ------------------------------------------------------------------
    print("Starting pipeline...")
    report = orchestrator.run()

    # ------------------------------------------------------------------
    # 9. Print summary
    # ------------------------------------------------------------------
    print("\n=== Pipeline finished ===")
    for tid, info in report["tasks"].items():
        print(f"  {tid}: status={info['status']}, version={info['version']}, "
              f"run_count={info['run_count']}, error={info['last_error']}")

    # Check if the output file was created
    output_path = "output/spec.md"
    if os.path.exists(output_path):
        print(f"\nSpecification written to {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            print(f"\n--- First 500 characters ---\n{f.read()[:500]}")
    else:
        print("\nOutput file was not created — check for errors above.")


if __name__ == "__main__":
    main()