# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repo has two layers:

1. **`lang_sam/` — Library** (`pip install -e .`): Python package that chains GroundingDINO (text → bounding boxes) with SAM 2.1 (bounding boxes → segmentation masks). Exposes a `LangSAM` class, a LitServe API server, and a Gradio UI.

2. **Top-level / pipeline scripts** — Autonomous driving research: use the LangSAM library to segment vehicle wheels in nuScenes/Lyft datasets, then estimate wheel yaw from pairs of wheel masks combined with camera calibration.

## Build / test / lint

```bash
pip install -e .                        # install lang_sam in editable mode
pip install -r requirements.txt         # install dependencies (gradio, litserve, torch, transformers, sam-2, etc.)
ruff check .                            # lint (config in pyproject.toml [tool.ruff])
pre-commit run --all-files              # run all pre-commit hooks (ruff, pyupgrade, docformatter, etc.)
python app.py                           # start Gradio + LitServe server on port 8000
```

There is no test suite.

## Architecture

### Library (`lang_sam/`)

```
LangSAM.predict(images_pil, texts_prompt) → list[dict]
  ├── GDINO.predict()        # text → boxes, scores, labels
  └── SAM.predict_batch()    # boxes → masks, mask_scores
```

- **`lang_sam/lang_sam.py`** — `LangSAM` class. Constructor accepts optional local checkpoint paths for offline use. `predict()` orchestrates the two-stage pipeline: first GroundingDINO finds objects matching the text prompt, then SAM segments each detected box. Supports batch inference.
- **`lang_sam/models/gdino.py`** — Wraps `IDEA-Research/grounding-dino-base` from HuggingFace transformers. Loads from HF Hub by default, or from local `model_ckpt_path` / `processor_ckpt_path`. Appends a period to text prompts automatically (GroundingDINO convention).
- **`lang_sam/models/sam.py`** — Wraps SAM 2.1 (`sam-2` pip package). Supports 4 model sizes defined in `SAM_MODELS` dict: `sam2.1_hiera_tiny/small/base_plus/large`. Uses Hydra/OmegaConf to instantiate model configs. Provides `predict()` (single image), `predict_batch()`, and `generate()` (automatic mask generation).
- **`lang_sam/models/utils.py`** — Device selection (CUDA > MPS > CPU) with TF32 optimizations for Ampere+ GPUs.
- **`lang_sam/utils.py`** — Visualization: draws boxes + masks on images using `supervision` library. Also provides `generate_labelme_json()` and contour extraction utilities for LabelMe-format annotations.
- **`lang_sam/server.py`** — LitServe API. Multipart form-data endpoint: accepts image + text prompt, returns PNG.
- **`app.py`** — Gradio web UI that calls the LitServe endpoint via HTTP. Includes preset examples.

### Wheel segmentation pipeline

The scripts process nuScenes data through these stages (each script is standalone, driven by `main()`):

1. **`seg_wheels/seg_wheels.py`** — Run LangSAM (text prompt `"wheel."`) on every nuScenes camera image. Saves annotated images + JSON with masks, boxes, scores, and unique wheel tokens. Filters small detections (<400px). Uses `sam2.1_hiera_large` with local checkpoints.

2. **`seg_wheels/main.py`** — Same idea but adds LiDAR-projection-based 3D association. Maps LiDAR point clouds to camera images, projects wheel detections to 3D, associates wheels to 3D bounding boxes. Outputs per-sample wheel annotation JSONs.

3. **`process_wheels/association_n_filtering.py`** — Post-processes wheel annotations: filters overlapping false wheels (IoU-based), associates wheels to 3D vehicle boxes using 2D overlap + 3D distance, filters non-vehicle-associated wheels.

4. **`process_wheels/calc_wheel_yaw.py`** — Computes wheel yaw angle from pairs of wheel masks belonging to the same vehicle. Uses camera calibration to lift 2D wheel ground-contact points to 3D ego-frame direction vectors. Validates against annotated box yaw within 45° threshold. Also performs noise sensitivity analysis (pixel noise ±10px, angle noise ±30°).

5. **`process_wheels/calc_wheel_yaw_sensor.py`** — Same yaw calculation but operates in sensor coordinate frame instead of ego frame. Contains additional pitch filtering.

6. **`process_wheels/robust_analysis.py`** — 3D visualization of yaw sensitivity to noise (scatter plots of Δu, Δv vs Δθ).

7. **`process_wheels/stat_dataset_*.py`** — Aggregation scripts for summary statistics across the dataset.

### Supporting scripts

- **`nuscenes_box.py`** — Draws 3D bounding boxes on nuScenes camera images (debugging/visualization).
- **`lyft.py`** — Same but for the Lyft dataset.
- **`stat_radar.py`** — Radar point cloud statistics (incomplete/WIP).
- **`seg_wheels/merge_json.py`** — Utility to merge split JSON wheel annotation files.

## Key conventions

- All pipeline scripts use hardcoded absolute paths (`/home/danc1nc0de/Datasets/nuScenes/`, `/home/danc1nc0de/Datasets/Lyft/`). Expect to update these when running elsewhere.
- Scripts use early-return caching: if output JSONs already exist, `main()` skips processing.
- Model checkpoints live in `checkpoints/` (SAM 2.1 weights) and `grounding-dino-base/` (GroundingDINO model files).
- The default SAM model is `sam2.1_hiera_small` for interactive use, `sam2.1_hiera_large` for batch processing.
- Pre-commit hooks: ruff, pyupgrade (3.11+), docformatter, mdformat, commitizen.
- Line length: 120 (ruff config).
