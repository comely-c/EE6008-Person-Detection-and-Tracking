"""
Step 3 sanity check: run Grounding DINO alone on a single frame and
save a visualization, without SAM 3.1, Qwen3-VL, or tracking.

Run:
    python scripts/run_detection.py

Output:
    outputs/dino_test.jpg
"""

import sys

sys.path.append(".")

import cv2
import yaml

from src.datasets.mot import MOTDataset
from src.models.grounding_dino import GroundingDINOModel
from src.utils.visualization import draw_detections


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = f"cuda:{config['device']['gpu_id']}"

    dino_config = config["models"]["grounding_dino"]

    dino = GroundingDINOModel(
        config_path=dino_config["config"],
        checkpoint_path=dino_config["checkpoint"],
        device=device,
        box_threshold=dino_config["box_threshold"],
        text_threshold=dino_config["text_threshold"],
    )

    dino.load()

    dataset = MOTDataset(
        config["dataset"]["mot17"]["root"],
        config["dataset"]["mot17"]["split"],
    )

    sequence = dataset.get_sequences()[0]
    frames = sequence.get_frames()
    frame_path = frames[0]

    print(f"Testing frame: {frame_path}")

    image = sequence.read_frame(frame_path)

    detections = dino.detect(image, prompt=dino_config["prompt"])

    print(f"Detected {len(detections)} persons.")

    image_height, image_width = image.shape[:2]

    converted = []
    for d in detections:
        cx, cy, w, h = d["box"]
        x1 = (cx - w / 2) * image_width
        y1 = (cy - h / 2) * image_height
        x2 = (cx + w / 2) * image_width
        y2 = (cy + h / 2) * image_height
        converted.append({"box": [x1, y1, x2, y2], "score": d["score"], "label": d["label"]})

    output = draw_detections(image, converted)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    output_path = "outputs/dino_test.jpg"
    cv2.imwrite(output_path, output_bgr)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
