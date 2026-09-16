"""
Simple IoU-based multi-object tracker.

This is a baseline tracker to get the pipeline running end-to-end.
Once SAM 3.1's own video-tracking / Object Multiplex capability is
integrated, this can be replaced or used as a fallback/comparison.
"""

from typing import List, Dict


def compute_iou(box_a, box_b) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_w = max(0, x2 - x1)
    intersection_h = max(0, y2 - y1)
    intersection = intersection_w * intersection_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


class Track:

    def __init__(self, track_id: int, detection: Dict):
        self.track_id = track_id
        self.box = detection["box"]
        self.score = detection["score"]

        self.age = 0
        self.hits = 1
        self.time_since_update = 0


class IoUTracker:

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age

        self.tracks: List[Track] = []
        self.next_id = 1

    def update(self, detections: List[Dict]) -> List[Dict]:
        if len(self.tracks) == 0:
            for detection in detections:
                self.tracks.append(Track(self.next_id, detection))
                self.next_id += 1

            return self._get_active_tracks()

        matched_tracks = set()
        matched_detections = set()
        matches = []

        for t_idx, track in enumerate(self.tracks):
            best_iou = 0.0
            best_detection = None

            for d_idx, detection in enumerate(detections):
                if d_idx in matched_detections:
                    continue

                iou = compute_iou(track.box, detection["box"])

                if iou > best_iou:
                    best_iou = iou
                    best_detection = d_idx

            if best_detection is not None and best_iou >= self.iou_threshold:
                matches.append((t_idx, best_detection))
                matched_tracks.add(t_idx)
                matched_detections.add(best_detection)

        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            detection = detections[d_idx]

            track.box = detection["box"]
            track.score = detection["score"]
            track.hits += 1
            track.time_since_update = 0

        for idx, track in enumerate(self.tracks):
            if idx not in matched_tracks:
                track.time_since_update += 1

        for d_idx, detection in enumerate(detections):
            if d_idx not in matched_detections:
                self.tracks.append(Track(self.next_id, detection))
                self.next_id += 1

        self.tracks = [
            track for track in self.tracks if track.time_since_update <= self.max_age
        ]

        return self._get_active_tracks()

    def _get_active_tracks(self) -> List[Dict]:
        return [
            {"track_id": t.track_id, "box": t.box, "score": t.score}
            for t in self.tracks
            if t.time_since_update == 0
        ]
