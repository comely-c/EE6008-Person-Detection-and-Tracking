"""
Quick smoke test: everything in src/ should at least be importable
without needing GPUs, model weights, or GroundingDINO/SAM3 installed
(those are lazily imported inside load()/detect() calls, not at module
import time).

Run:
    python -m pytest tests/test_imports.py -v
or:
    python tests/test_imports.py
"""

import sys

sys.path.append(".")


def test_imports():
    from src.datasets.mot import MOTDataset, MOTSequence  # noqa: F401
    from src.models.qwen3_vl import Qwen3VLModel  # noqa: F401
    from src.models.grounding_dino import GroundingDINOModel  # noqa: F401
    from src.models.sam31 import SAM31Model  # noqa: F401
    from src.tracking.tracker import IoUTracker, compute_iou  # noqa: F401
    from src.pipeline.baseline_pipeline import BaselinePipeline  # noqa: F401
    from src.evaluation.mot_metrics import MOTResultWriter  # noqa: F401
    from src.utils.visualization import draw_tracks, draw_detections  # noqa: F401

    print("All modules imported successfully.")


def test_iou_tracker_basic():
    from src.tracking.tracker import IoUTracker

    tracker = IoUTracker(iou_threshold=0.3, max_age=5)

    det_frame_1 = [{"box": [10, 10, 50, 50], "score": 0.9, "label": "person"}]
    tracks_1 = tracker.update(det_frame_1)
    assert len(tracks_1) == 1
    assert tracks_1[0]["track_id"] == 1

    # Same box next frame -> same track id, not a new one
    det_frame_2 = [{"box": [11, 11, 51, 51], "score": 0.9, "label": "person"}]
    tracks_2 = tracker.update(det_frame_2)
    assert len(tracks_2) == 1
    assert tracks_2[0]["track_id"] == 1

    print("IoU tracker basic test passed.")

def test_config():
    import yaml

    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "dataset" in config
    assert "models" in config
    assert "tracking" in config
    assert "output" in config

    print("Config test passed.")

if __name__ == "__main__":
    test_imports()
    test_iou_tracker_basic()
    test_config()
