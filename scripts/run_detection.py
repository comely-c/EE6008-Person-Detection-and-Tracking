import os
import cv2
import torch

from groundingdino.util.inference import load_model, load_image, predict, annotate


# =========================
# Paths
# =========================
CONFIG_PATH = (
    "third_party/GroundingDINO/"
    "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)

CHECKPOINT_PATH = "weights/groundingdino_swint_ogc.pth"

IMAGE_PATH = (
    "/projects/_hdd/EE6008cca9/datasets/"
    "MOT17/train/MOT17-02-DPM/img1/000001.jpg"
)

OUTPUT_PATH = "outputs/groundingdino_mot17_000001.jpg"


# =========================
# Check GPU
# =========================
print("=" * 60)
print("Grounding DINO single-image test")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Check image
# =========================
if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )

print("\nImage:")
print(IMAGE_PATH)


# =========================
# Load Grounding DINO
# =========================
print("\nLoading Grounding DINO...")

model = load_model(
    CONFIG_PATH,
    CHECKPOINT_PATH,
    device=device,
)

print("Grounding DINO loaded successfully.")


# =========================
# Load image
# =========================
image_source, image = load_image(IMAGE_PATH)

print("Image loaded.")
print("Original image shape:", image_source.shape)


# =========================
# Detection
# =========================
print("\nRunning detection...")

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption="person.",
    box_threshold=0.35,
    text_threshold=0.25,
    device=device,
)


# =========================
# Print results
# =========================
print("\nDetection results")
print("-" * 60)

print("Number of detections:", len(boxes))

for i, (box, logit, phrase) in enumerate(
    zip(boxes, logits, phrases)
):
    print(
        f"[{i}] "
        f"confidence={float(logit):.4f}, "
        f"label={phrase}, "
        f"box={box.tolist()}"
    )


# =========================
# Visualize
# =========================
print("\nCreating visualization...")

annotated_frame = annotate(
    image_source=image_source,
    boxes=boxes,
    logits=logits,
    phrases=phrases,
)

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

cv2.imwrite(
    OUTPUT_PATH,
    annotated_frame,
)

print("\nSaved result:")
print(OUTPUT_PATH)

print("=" * 60)
print("TEST FINISHED")
print("=" * 60)
