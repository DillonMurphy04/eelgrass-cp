# Paths and Data Checklist

This project assumes local data and trained model files. Update each script's `BASE_DIR`, `RASTER_PATH`, `POINTS_SHP`, and `MODEL_EMDS` to match the local environment.

**Required Data**
- 2021 tile dataset with `map.txt` and corresponding images/labels.
- 2022 raster (`.tif`) used for point inference.
- 2022 point shapefile with ground-truth field (usually `GROUND_TRU`).

**Required Model Artifacts**
- ArcGIS `.emd` files for the ensembles to evaluate.

**Common Output Directories**
- Calibration cache (NPZ + JSON) for 2021 pixel CP.
- 2022 point caches (NPZ) for OOD evaluation.
- Figures and CSV/JSON summaries for each analysis script.
