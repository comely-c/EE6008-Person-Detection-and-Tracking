"""
Step 4 sanity check: run Qwen3-VL alone on a single frame.

Run:
    python scripts/run_qwen_test.py
"""

import sys

sys.path.append(".")

import yaml

from src.datasets.mot import MOTDataset
from src.models.qwen3_vl import Qwen3VLModel


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = f"cuda:{config['device']['gpu_id']}"
    qwen_config = config["models"]["qwen3_vl"]

    qwen = Qwen3VLModel(
        model_name=qwen_config["model_name"],
        device=device,
        max_new_tokens=qwen_config["max_new_tokens"],
    )

    qwen.load()

    dataset = MOTDataset(
        config["dataset"]["mot17"]["root"],
        config["dataset"]["mot17"]["split"],
    )

    sequence = dataset.get_sequences()[0]
    frame_path = sequence.get_frames()[0]

    print(f"Testing frame: {frame_path}")

    result = qwen.analyze(
        str(frame_path),
        "Identify the people in this image. Describe their approximate locations.",
    )

    print("Qwen3-VL output:")
    print(result)


if __name__ == "__main__":
    main()
