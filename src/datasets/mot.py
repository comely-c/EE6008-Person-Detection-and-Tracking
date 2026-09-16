"""
MOT17 / MOT20 dataset loader.

Expected directory layout:

    MOT17/
      train/
        MOT17-02-FRCNN/
          img1/
            000001.jpg
            000002.jpg
            ...
          gt/
            gt.txt
          seqinfo.ini
        MOT17-04-FRCNN/
          ...
      test/
        ...

If your actual folder structure differs (e.g. no "-FRCNN" suffix, or a
different image extension), adjust `_find_sequences` / `get_frames` below.
"""

from pathlib import Path
from typing import List

import cv2


class MOTSequence:
    """A single MOT sequence, e.g. MOT17-02-FRCNN."""

    def __init__(self, sequence_path: str):
        self.root = Path(sequence_path)

        if not self.root.exists():
            raise FileNotFoundError(f"Sequence not found: {self.root}")

        self.name = self.root.name

        self.img_dir = self.root / "img1"
        self.gt_file = self.root / "gt" / "gt.txt"
        self.seqinfo_file = self.root / "seqinfo.ini"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"img1 directory not found: {self.img_dir}")

    def get_frames(self) -> List[Path]:
        """Return all frame paths, sorted by filename."""
        frames = sorted(self.img_dir.glob("*.jpg"))

        if len(frames) == 0:
            frames = sorted(self.img_dir.glob("*.png"))

        return frames

    def get_frame_count(self) -> int:
        return len(self.get_frames())

    def read_frame(self, frame_path: Path):
        """Read a frame and return it as an RGB numpy array."""
        image = cv2.imread(str(frame_path))

        if image is None:
            raise RuntimeError(f"Failed to read image: {frame_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image


class MOTDataset:
    """Wraps a full MOT17/MOT20 split (e.g. 'train' or 'test')."""

    def __init__(self, root: str, split: str = "train"):
        self.root = Path(root)
        self.split = split

        self.split_root = self.root / split

        if not self.split_root.exists():
            raise FileNotFoundError(f"MOT split not found: {self.split_root}")

        self.sequences = self._find_sequences()

    def _find_sequences(self) -> List[MOTSequence]:
        sequences = []

        for path in sorted(self.split_root.iterdir()):
            if not path.is_dir():
                continue

            img_dir = path / "img1"

            if img_dir.exists():
                sequences.append(MOTSequence(str(path)))

        return sequences

    def get_sequences(self) -> List[MOTSequence]:
        return self.sequences

    def __len__(self):
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]
