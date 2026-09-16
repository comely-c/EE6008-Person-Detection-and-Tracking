"""
Step 1 sanity check: confirm MOT17/MOT20 sequences and frames are
read correctly, using the paths in configs/config.yaml.

Run:
    python scripts/test_dataset.py
"""

import sys

sys.path.append(".")

import yaml

from src.datasets.mot import MOTDataset


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset = MOTDataset(
        root=config["dataset"]["mot17"]["root"],
        split=config["dataset"]["mot17"]["split"],
    )

    print(f"Found {len(dataset)} sequences.")

    for sequence in dataset:
        frames = sequence.get_frames()

        print(f"{sequence.name}: {len(frames)} frames")

        if len(frames) > 0:
            print(f"  First frame: {frames[0]}")


if __name__ == "__main__":
    main()
