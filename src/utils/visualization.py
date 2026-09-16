"""Drawing helpers for detections/tracks."""

import cv2


def draw_tracks(image, tracks):
    """
    Args:
        image: RGB numpy array
        tracks: list of {"track_id", "box": [x1,y1,x2,y2], "score"}

    Returns:
        RGB numpy array with boxes drawn.
    """
    output = image.copy()

    for track in tracks:
        box = track["box"]
        track_id = track["track_id"]
        score = track["score"]

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"ID:{track_id} {score:.2f}"

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return output


def draw_detections(image, detections):
    """
    Args:
        image: RGB numpy array
        detections: list of {"box": [x1,y1,x2,y2], "score", "label"}

    Returns:
        RGB numpy array with boxes drawn.
    """
    output = image.copy()

    for detection in detections:
        box = detection["box"]
        score = detection["score"]
        label = detection.get("label", "person")

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 128, 0), 2)

        cv2.putText(
            output,
            f"{label} {score:.2f}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 128, 0),
            2,
        )

    return output
