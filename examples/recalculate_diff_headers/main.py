import sys
from pathlib import Path

# Assuming recalculate_diff_headers is in the same directory or importable
from src.utils import recalculate_diff_headers


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_diff_file>")
        sys.exit(1)

    diff_path = Path(sys.argv[1])

    if not diff_path.exists():
        print(f"Error: File '{diff_path}' not found.")
        sys.exit(1)

    original_text = diff_path.read_text(encoding="utf-8")
    fixed_text = recalculate_diff_headers(original_text)
    diff_path.write_text(fixed_text, encoding="utf-8")

    print(f"Diff headers recalculated and saved to '{diff_path}'.")


if __name__ == "__main__":
    main()