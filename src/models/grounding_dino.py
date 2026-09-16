"""
Grounding DINO wrapper.

Requires the official repo to be installed first:

    git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
    cd third_party/GroundingDINO
    pip install -e .

And a checkpoint, e.g. groundingdino_swint_ogc.pth, placed under weights/.

Note: GroundingDINO's `predict()` returns boxes in normalized
(cx, cy, w, h) format relative to image width/height — convert to
absolute (x1, y1, x2, y2) before handing off to SAM/tracking (this is
done in src/pipeline/baseline_pipeline.py).
"""

from typing import List, Dict

import numpy as np


class GroundingDINOModel:

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.model = None
        self._transform = None

    def load(self):
        print("[GroundingDINO] Loading model...")

        from groundingdino.util.inference import load_model

        self.model = load_model(
            self.config_path,
            self.checkpoint_path,
            device=self.device,
        )

        import groundingdino.datasets.transforms as T

        self._transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

        print("[GroundingDINO] Model loaded.")

    def detect(self, image, prompt: str = "person.") -> List[Dict]:
        """
        Args:
            image: RGB numpy array (H, W, 3)
            prompt: text prompt, e.g. "person."

        Returns:
            List of dicts: {"box": [cx, cy, w, h] normalized 0-1,
                             "score": float, "label": str}
        """
        if self.model is None:
            raise RuntimeError("GroundingDINO model is not loaded. Call load() first.")

        from groundingdino.util.inference import predict
        from PIL import Image as PILImage

        image = np.asarray(image)
        pil_image = PILImage.fromarray(image)

        image_tensor, _ = self._transform(pil_image, None)
        image_tensor = image_tensor.to(self.device)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        detections = []

        for box, score, phrase in zip(boxes, logits, phrases):
            detections.append(
                {
                    "box": box.detach().cpu().numpy().tolist(),
                    "score": float(score.detach().cpu().item()),
                    "label": phrase,
                }
            )

        return detections
