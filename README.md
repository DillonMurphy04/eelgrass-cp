# Eelgrass Conformal Prediction (Thesis Code)

This repository is a cleaned, shareable version of the thesis scripts for eelgrass semantic segmentation uncertainty and conformal prediction. It centralizes repeated utilities into a small Python package and organizes the workflow scripts by purpose (calibration, evaluation, spatial analysis, and visualizations).

**Highlights**
- Shared utilities are found in `src/eelgrass_cp` and are reused by the main scripts.
- Scripts are grouped in `scripts/` by domain (conformal, spatial, sensitivity, visualization, inference).
- Outputs are written to user-specified directories so results are reproducible without changing code structure.

**Repository Layout**
- `src/eelgrass_cp` - Shared utilities (model loading, normalization, CP helpers).
- `scripts/conformal` - Calibration and conformal prediction workflows.
- `scripts/spatial` - Spatial blocking analyses (Morton ordering).
- `scripts/ablation` - Sensitivity analysis experiments.
- `scripts/viz` - Raster and point-based visualization figures.
- `scripts/inference` - Deep-ensemble point inference.
- `docs` - Data/paths guidance.

**Quickstart**
1. Use an environment with Python 3.9+. If running from the ArcGIS Pro Python Command Prompt, most dependencies are already available.
2. If needed, install dependencies: `pip install -r requirements.txt`
3. Run scripts from the repo root so imports like `from src.eelgrass_cp import ...` resolve.
4. Example: `python scripts\conformal\calibrate_pixel_cp_2021.py`

**Data & Paths**
The scripts reference local datasets and model `.emd` files that are not included in this repository. Update the path constants at the top of each script to match the local machine. See `docs/paths_and_data.md` for a concise checklist of required files.

**Suggested Order**
1. Build the 2021 pixel-level calibration cache:
   `scripts/conformal/calibrate_pixel_cp_2021.py`
2. Evaluate 2022 point predictions (vanilla, linear lambda=3, nonparametric):
   `scripts/conformal/eval_2022_points.py`
   `scripts/conformal/eval_2022_compare_lambdas.py`
   `scripts/conformal/nonparametric_eval_2022.py`
   `scripts/conformal/nonparametric_scale_vis.py`
3. Spatial-blocked evaluation (Morton ordering):
   `scripts/spatial/spatial_blocks_morton.py`
4. Sensitivity analysis and bootstrap CIs:
   `scripts/ablation/sensitivity_analysis.py`
5. Visualizations:
   `scripts/viz/raster_viz.py`
   `scripts/viz/raster_viz_transition.py`
   `scripts/viz/raster_viz_single_chip.py`

**Outputs**
Most scripts write to an `OUT_DIR` configured at the top of the file. Typical outputs include:
- CSV summary tables
- JSON metadata/summary artifacts
- Figure PNGs
- Cached NPZ inference tensors

**Notes**
- `arcgis` is required to load `.emd` models via `arcgis.learn`. If using ArcGIS Pro Python, ensure the environment is activated when running scripts.
- GPU is strongly recommended for full-resolution inference.
