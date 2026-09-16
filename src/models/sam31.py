"""
SAM 3.1 wrapper.

Important: SAM 3.1 is Meta's 2026 release at facebookresearch/sam3 —
it is NOT the older segment-anything (SAM 1/2) package, and its API is
different (adds video tracking / "Object Multiplex" support). Do not
install the old `segment-anything` pip package for this.

Install:
    git clone https://github.com/facebookresearch/sam3.git third_party/sam3
    cd third_party/sam3
    pip install -e .

Official requirements at time of writing: Python >= 3.12,
PyTorch >= 2.7, CUDA >= 12.6, and Hugging Face access to the checkpoint.
Confirm current requirements against the repo before installing, since
these change between releases.

This wrapper intentionally does NOT fake the segmentation call — once
the official package is installed, fill in `load()` and
`segment_from_boxes()` using the real sam3 API (`build_sam3_image_model`
or whatever the installed version exposes).
"""

from pathlib import Path


class SAM31Model:

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device

        self.model = None
        self.predictor = None

    def load(self):
        print("[SAM3.1] Loading model...")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM3.1 checkpoint not found: {self.checkpoint_path}"
            )

        try:
            from sam3.model_builder import build_sam3_image_model

            self.model = build_sam3_image_model(
                checkpoint_path=str(self.checkpoint_path)
            )

        except ImportError as exc:
            raise ImportError(
                "SAM3 is not installed. Clone and install the official "
                "facebookresearch/sam3 repository first "
                "(see module docstring)."
            ) from exc

        print("[SAM3.1] Model loaded.")

    def segment_from_boxes(self, image, boxes):
        """
        Args:
            image: RGB numpy array (H, W, 3)
            boxes: list of [x1, y1, x2, y2] absolute pixel boxes

        Returns:
            list of {"box": box, "mask": mask_or_None}

        TODO: replace this with the real SAM3.1 predictor/session call
        once the model is loaded (e.g. set_image + predict per box, or
        the batched box-prompt API the installed sam3 version exposes).
        """
        if self.model is None:
            raise RuntimeError("SAM3.1 model is not loaded. Call load() first.")

        results = []

        for box in boxes:
            results.append({"box": box, "mask": None})

        return results
