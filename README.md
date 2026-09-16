# Person Detection and Tracking Using Artificial Intelligence

Baseline: **Qwen3-VL** + **Grounding DINO** + **SAM 3.1**, evaluated on **MOT17 / MOT20**.

## Project layout

```
person-detection-tracking/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml            # all paths & hyperparameters live here
├── src/
│   ├── datasets/mot.py        # MOT17/MOT20 loader
│   ├── models/
│   │   ├── qwen3_vl.py        # Qwen3-VL (Transformers)
│   │   ├── grounding_dino.py  # Grounding DINO detector
│   │   └── sam31.py           # SAM 3.1 segmentation/tracking wrapper
│   ├── pipeline/baseline_pipeline.py
│   ├── tracking/tracker.py    # IoU tracker (baseline)
│   ├── evaluation/mot_metrics.py
│   └── utils/
│       ├── visualization.py
│       └── logger.py
├── scripts/
│   ├── test_dataset.py        # Step 1: verify dataset reads correctly
│   ├── check_gpu.py           # Step 2: verify CUDA is visible
│   ├── run_detection.py       # Step 3: Grounding DINO only, 1 frame
│   ├── run_qwen_test.py       # Step 4: Qwen3-VL only, 1 frame
│   ├── run_baseline.py        # Step 6: full pipeline on MOT17
│   ├── run_mot20.py           # full pipeline on MOT20
│   └── evaluate.py            # lists produced prediction files
├── outputs/                   # pipeline results (gitignored, except .gitkeep)
├── weights/                   # checkpoints go here (gitignored)
├── third_party/               # GroundingDINO / sam3 repos go here (gitignored)
└── tests/
    └── test_imports.py
```

## Setup

1. Install PyTorch matching your CUDA version first:
   https://pytorch.org/get-started/locally/

2. Install this project's Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Grounding DINO:
   ```
   git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
   cd third_party/GroundingDINO
   pip install -e .
   cd ../..
   ```
   Download a checkpoint (e.g. `groundingdino_swint_ogc.pth`) into `weights/`.

4. Install SAM 3.1. This is Meta's 2026 `facebookresearch/sam3` repo —
   **not** the older `segment-anything` package, and its API differs
   (it adds video tracking / "Object Multiplex"). Check the repo's
   current README for the required Python/PyTorch/CUDA versions before
   installing, since these can change between releases:
   ```
   git clone https://github.com/facebookresearch/sam3.git third_party/sam3
   cd third_party/sam3
   pip install -e .
   cd ../..
   ```
   Place the checkpoint in `weights/` (Hugging Face access may be
   required to download it).

5. Edit `configs/config.yaml` — set `dataset.mot17.root` /
   `dataset.mot20.root` to your local dataset paths (e.g. `E:/MOT17`
   on Windows, `/data/<you>/MOT17` on a lab GPU server). Nothing else
   in the code should need to change between machines.

## Expected MOT directory layout

```
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
```

If your actual folders look different (extra suffixes, different image
extension, etc.), adjust `src/datasets/mot.py` accordingly.

## Recommended run order

Run these one at a time — each isolates one part of the system, so a
failure points straight at the cause instead of getting lost inside
the full pipeline.

1. **Dataset**
   ```
   python scripts/test_dataset.py
   ```
   Confirms sequences and frame paths are found correctly.

2. **GPU**
   ```
   python scripts/check_gpu.py
   ```
   Confirms `torch.cuda.is_available()` is `True` on the target machine.

3. **Grounding DINO only**
   ```
   python scripts/run_detection.py
   ```
   Produces `outputs/dino_test.jpg` — open it and check the person
   boxes look reasonable before moving on.

4. **Qwen3-VL only**
   ```
   python scripts/run_qwen_test.py
   ```
   Runs on a single frame — don't point it at the whole dataset yet.

5. **SAM 3.1 only** — write and run your own small script that loads
   `SAM31Model` and calls `segment_from_boxes()` on a couple of boxes
   from step 3, once the real sam3 API calls are filled in in
   `src/models/sam31.py` (see the TODO there).

6. **Full pipeline**
   ```
   python scripts/run_baseline.py      # MOT17
   python scripts/run_mot20.py         # MOT20
   ```

## Suggested phases

| Phase | Goal |
|---|---|
| 1 | Dataset loading (MOT17/MOT20 → frames) |
| 2 | Grounding DINO → person boxes, visualized |
| 3 | SAM 3.1 → person masks from those boxes |
| 4 | Qwen3-VL integrated alongside detection |
| 5 | Tracking (IDs consistent across frames) |
| 6 | Evaluation (HOTA / MOTA / IDF1 / ID switches via TrackEval) |

## Evaluation

`run_baseline.py` / `run_mot20.py` write per-sequence prediction files
in standard MOTChallenge format (`frame,id,x,y,w,h,score,-1,-1,-1`)
under `outputs/<sequence_name>/<sequence_name>.txt`.

`scripts/evaluate.py` just lists these files. For real metrics, feed
them into the official toolkit rather than hand-rolling the metrics:
https://github.com/JonathonLuiten/TrackEval

## Notes

- `src/models/sam31.py` deliberately does **not** fake a working SAM
  3.1 call — the `load()` and `segment_from_boxes()` methods raise/
  stub out until you wire in the real `sam3` package API, since
  guessing at an API for a 2026 release risks being wrong and wasting
  GPU-hour debugging time on the lab server.
- The IoU tracker in `src/tracking/tracker.py` is a placeholder
  baseline. SAM 3.1's own video-tracking capability may end up
  replacing it — that's a design decision to make once phases 1–4 are
  working.
