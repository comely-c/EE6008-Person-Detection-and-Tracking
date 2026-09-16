"""
Baseline pipeline: Grounding DINO detection -> SAM 3.1 segmentation
-> IoU tracking, with Qwen3-VL running alongside as an independent,
non-blocking analysis step.
"""

from pathlib import Path

from src.utils.visualization import draw_tracks


class BaselinePipeline:

    def __init__(self, qwen=None, dino=None, sam=None, tracker=None, output_dir="outputs"):
        self.qwen = qwen
        self.dino = dino
        self.sam = sam
        self.tracker = tracker

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_models(self):
        if self.qwen is not None:
            self.qwen.load()

        if self.dino is not None:
            self.dino.load()

        if self.sam is not None:
            self.sam.load()

    def process_frame(self, image, image_path=None):
        """
        Args:
            image: RGB numpy array (H, W, 3)
            image_path: optional path to the frame on disk, needed by
                Qwen3-VL, which takes an image path rather than an
                in-memory array.

        Returns:
            dict with keys: qwen_result, detections, masks, tracks
        """

        # 1. Qwen3-VL (independent, does not block detection/tracking)
        qwen_result = None

        if self.qwen is not None and image_path is not None:
            qwen_result = self.qwen.analyze(
                str(image_path),
                "Identify the people in this image. "
                "Describe their approximate locations.",
            )

        # 2. Grounding DINO detection
        detections = []

        if self.dino is not None:
            detections = self.dino.detect(image, prompt="person.")

        # 3. Convert normalized (cx, cy, w, h) boxes to absolute (x1, y1, x2, y2)
        image_height, image_width = image.shape[:2]

        converted_detections = []

        for detection in detections:
            cx, cy, w, h = detection["box"]

            x1 = (cx - w / 2) * image_width
            y1 = (cy - h / 2) * image_height
            x2 = (cx + w / 2) * image_width
            y2 = (cy + h / 2) * image_height

            x1 = max(0, min(image_width - 1, x1))
            y1 = max(0, min(image_height - 1, y1))
            x2 = max(0, min(image_width - 1, x2))
            y2 = max(0, min(image_height - 1, y2))

            converted_detections.append(
                {
                    "box": [x1, y1, x2, y2],
                    "score": detection["score"],
                    "label": detection["label"],
                }
            )

        # 4. SAM 3.1 segmentation from boxes
        masks = None

        if self.sam is not None and len(converted_detections) > 0:
            boxes = [d["box"] for d in converted_detections]
            masks = self.sam.segment_from_boxes(image, boxes)

        # 5. Tracking
        tracks = []

        if self.tracker is not None:
            tracks = self.tracker.update(converted_detections)

        return {
            "qwen_result": qwen_result,
            "detections": converted_detections,
            "masks": masks,
            "tracks": tracks,
        }

    def visualize(self, image, results):
        return draw_tracks(image, results["tracks"])
