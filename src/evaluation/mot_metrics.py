"""
Writes tracking results in the standard MOTChallenge prediction format:

    frame_id, track_id, x, y, w, h, score, -1, -1, -1

For actual HOTA / MOTA / IDF1 / ID-switch evaluation, use the official
TrackEval toolkit (https://github.com/JonathonLuiten/TrackEval) against
these output files rather than reimplementing the metrics by hand.
"""

from pathlib import Path


class MOTResultWriter:

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(self.output_path, "w", encoding="utf-8")

    def write(self, frame_id, track_id, box, score):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        self.file.write(
            f"{frame_id},{track_id},{x1:.2f},{y1:.2f},"
            f"{width:.2f},{height:.2f},{score:.4f},-1,-1,-1\n"
        )

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
