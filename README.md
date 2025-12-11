# Computer Vision Project (Harris vs. SIFT)

This repo runs a full pipeline comparing Harris and SIFT keypoint detectors (basic detection, robustness tests, distribution analysis, parameter sweeps, and report generation).

## 1) Environment setup
- Use Python 3.10+.
- Create a virtual environment (recommended):
  - `python3 -m venv .venv && source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
  - `opencv-contrib-python-headless` is included so SIFT works without GUI issues.

## 2) Data expected
- Place your building photos (JPG/PNG/JPEG) in `data/original/`.
- Five example images are already present: `building_1/2/3.jpg`, `roma_1/2.jpg`.
- Other folders (`data/transformations`, `data/kaggle_dataset`) are optional for now.

## 3) Running the pipeline
- From the project root: `python run_experiments.py`
- The script will:
  - Create all needed folders under `results/` and `data/` if missing.
  - Run Harris and SIFT on each image.
  - Perform robustness tests (scale, rotation, brightness, blur, noise).
  - Run parameter analysis.
  - Generate plots, metrics, and `results/final_report.md`.

## 4) Outputs
- Key artifacts land in:
  - `results/harris`, `results/sift`, `results/comparison`, `results/distribution`
  - `results/robustness`, `results/plots`, `results/metrics`
  - Final report: `results/final_report.md`

## 5) Troubleshooting
- `ModuleNotFoundError: No module named 'cv2'`: activate your venv and re-run `pip install -r requirements.txt`.
- If matplotlib complains about a GUI backend, it is only saving figures (no `imshow` windows), so the headless OpenCV build avoids Qt issues.
- Ensure images exist in `data/original/`; otherwise the script stops with a helpful message.
