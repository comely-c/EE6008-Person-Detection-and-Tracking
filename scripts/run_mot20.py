"""
Same as run_baseline.py but reads the MOT20 config block instead of MOT17.

Run:
    python scripts/run_mot20.py
"""

import sys

sys.path.append(".")

from pathlib import Path

import cv2
import yaml

from src.datasets.mot import MOTDataset
from src.models.grounding_dino import GroundingDINOModel
from src.models.qwen3_vl import Qwen3VLModel
from src.models.sam31 import SAM31Model
from src.tracking.tracker import IoUTracker
from src.pipeline.baseline_pipeline import BaselinePipeline
from src.evaluation.mot_metrics import MOTResultWriter


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = f"cuda:{config['device']['gpu_id']}"

    qwen = None
    if config["models"]["qwen3_vl"]["enabled"]:
        qwen_config = config["models"]["qwen3_vl"]
        qwen = Qwen3VLModel(
            model_name=qwen_config["model_name"],
            device=device,
            max_new_tokens=qwen_config["max_new_tokens"],
        )

    dino = None
    if config["models"]["grounding_dino"]["enabled"]:
        dino_config = config["models"]["grounding_dino"]
        dino = GroundingDINOModel(
            config_path=dino_config["config"],
            checkpoint_path=dino_config["checkpoint"],
            device=device,
            box_threshold=dino_config["box_threshold"],
            text_threshold=dino_config["text_threshold"],
        )

    sam = None
    if config["models"]["sam31"]["enabled"]:
        sam_config = config["models"]["sam31"]
        sam = SAM31Model(checkpoint_path=sam_config["checkpoint"], device=device)

    tracker = None
    if config["tracking"]["enabled"]:
        tracker = IoUTracker(
            iou_threshold=config["tracking"]["iou_threshold"],
            max_age=config["tracking"]["max_age"],
        )

    pipeline = BaselinePipeline(
        qwen=qwen, dino=dino, sam=sam, tracker=tracker,
        output_dir=config["output"]["root"],
    )

    pipeline.load_models()

    dataset = MOTDataset(
        root=config["dataset"]["mot20"]["root"],
        split=config["dataset"]["mot20"]["split"],
    )

    max_frames = config["dataset"]["max_frames"]

    for sequence in dataset:
        print(f"\nProcessing {sequence.name}")

        frames = sequence.get_frames()
        if max_frames > 0:
            frames = frames[:max_frames]

        output_dir = Path(config["output"]["root"]) / sequence.name
        output_dir.mkdir(parents=True, exist_ok=True)

        result_writer = None
        if config["output"]["save_tracking"]:
            result_writer = MOTResultWriter(str(output_dir / f"{sequence.name}.txt"))

        for frame_idx, frame_path in enumerate(frames):
            print(f"\rFrame {frame_idx + 1}/{len(frames)}", end="")

            image = sequence.read_frame(frame_path)
            results = pipeline.process_frame(image, frame_path)

            if result_writer is not None:
                for track in results["tracks"]:
                    result_writer.write(
                        frame_id=frame_idx + 1,
                        track_id=track["track_id"],
                        box=track["box"],
                        score=track["score"],
                    )

            if config["output"]["save_visualization"]:
                visualized = pipeline.visualize(image, results)
                visualized_bgr = cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_dir / frame_path.name), visualized_bgr)

        if result_writer is not None:
            result_writer.close()

        print()

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
