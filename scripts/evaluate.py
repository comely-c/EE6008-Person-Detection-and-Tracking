"""
Lists the MOT-format prediction files produced by run_baseline.py.

For real metrics (HOTA / MOTA / IDF1 / ID switches / Precision /
Recall), feed these .txt files into the official TrackEval toolkit
rather than computing them by hand:
    https://github.com/JonathonLuiten/TrackEval

Run:
    python scripts/evaluate.py
"""

import sys

sys.path.append(".")

from pathlib import Path


def main():
    output_root = Path("outputs")

    files = list(output_root.rglob("*.txt"))

    print(f"Found {len(files)} result files.")

    for file in files:
        print(file)


if __name__ == "__main__":
    main()
