"""Shared utilities for eelgrass conformal prediction experiments."""

from .modeling import load_model_and_config, norm_chip, model_logits, model_probs
from .modeling import softmax_vec, softmax_rows, softmax_3d, softmax_4d
from .raster import safe_read_chip, ensure_uint8_rgb, read_map_pairs, read_points_shp
from .calibration import TemperatureScaler, fit_temperature, stratified_indices, normalize_var
from .calibration import fit_sigma_fn_from_cal
from .cp import qhat_split_conformal, qhat_conformal
from .cp import linear_score_transform, set_composition_binary, per_class_coverage, binned_coverage
from .metrics import corr

__all__ = [
    "load_model_and_config",
    "norm_chip",
    "model_logits",
    "model_probs",
    "softmax_vec",
    "softmax_rows",
    "softmax_3d",
    "softmax_4d",
    "safe_read_chip",
    "ensure_uint8_rgb",
    "read_map_pairs",
    "read_points_shp",
    "TemperatureScaler",
    "fit_temperature",
    "stratified_indices",
    "normalize_var",
    "fit_sigma_fn_from_cal",
    "qhat_split_conformal",
    "qhat_conformal",
    "linear_score_transform",
    "set_composition_binary",
    "per_class_coverage",
    "binned_coverage",
    "corr",
]
