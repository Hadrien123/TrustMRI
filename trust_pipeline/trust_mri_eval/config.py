"""All pipeline parameters in one dataclass.

Metric, uncertainty and model functions receive plain arrays and numbers; only the synthetic
generator, ``pipeline.py`` and ``cli.py`` read the :class:`Config` object. Override fields from the CLI with ``--set name=value``.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Shape3 = tuple[int, int, int]


@dataclass
class Config:
    # ---------------------------------------------------------------- run / paths
    run_id: str = ""                       # empty -> timestamp
    output_dir: Path = Path("outputs")
    data_dir: Path | None = None           # real data (one sub-folder per patient)
    synthetic: bool = True
    seed: int = 0
    regions: tuple[str, ...] = ("ET",)     # evaluated separately; synthetic data only has one region

    # ---------------------------------------------------------------- design
    n_patients: int = 8                    # P (synthetic)
    n_imputations: int = 5                 # N
    n_prompts: int = 5                     # K
    volume_shape: Shape3 = (240, 240, 155)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # ---------------------------------------------------------------- synthetic generator
    syn_brain_semi_axes: tuple[float, float, float] = (0.40, 0.45, 0.40)  # fraction of volume size
    syn_texture_sigma: float = 4.0         # low-pass filter of the brain texture (voxels)
    syn_n_tumours: tuple[int, int] = (1, 3)
    syn_tumour_radius: tuple[float, float] = (5.0, 25.0)
    syn_rim_width: float = 2.0             # enhancing rim thickness (voxels)
    syn_intensity_brain: float = 0.40     # white matter
    syn_intensity_gm: float = 0.28
    syn_intensity_csf: float = 0.10        # ventricles
    syn_gm_fraction: float = 0.35
    syn_intensity_texture: float = 0.03
    syn_intensity_core: float = 0.70
    syn_intensity_rim: float = 1.00
    syn_noise_std: float = 0.03            # smooth imputation noise
    syn_noise_sigma: float = 1.5
    syn_boundary_noise_gain: float = 3.0   # noise std multiplier at the tumour boundary
    syn_boundary_noise_width: float = 3.0
    syn_warp_amplitude: float = 1.0        # local distortions (voxels)
    syn_warp_sigma: float = 6.0
    syn_hallucination_prob: float = 0.3    # per imputed sample
    syn_hallucination_radius: tuple[float, float] = (3.0, 6.0)
    syn_prompt_max_offset: int = 2         # dilate / erode up to this many voxels depending on k
    syn_prompt_jitter: float = 0.7         # boundary jitter amplitude (voxels)
    syn_prompt_miss_prob: float = 0.15     # a prompt ignores a non-largest lesion
    syn_with_flair: bool = True
    syn_oedema_width: float = 6.0

    # ---------------------------------------------------------------- preprocessing
    rescale_percentiles: tuple[float, float] = (0.5, 99.5)
    roi_dilation: int = 5
    crop_margin: int = 2                   # margin around the brain bounding box
    seg_threshold: float = 0.5

    # ---------------------------------------------------------------- image metrics
    ssim_win_size: int = 7
    patch_size: Shape3 = (3, 3, 3)
    coarse_patch_size: Shape3 = (8, 8, 8)
    min_patch_coverage: float = 0.5        # brain (module B) or ROI (model) fraction for a valid patch
    failure_percentile: float = 90.0
    failure_error: str = "l1"              # "l1" or "l2"
    coverage_interval: tuple[float, float] = (5.0, 95.0)

    # ---------------------------------------------------------------- ANOVA
    anova_on: str = "probs"                # "probs" or "masks"
    anova_chunk: int = 200_000

    # ---------------------------------------------------------------- features / labels
    context_sigmas: tuple[float, ...] = (2.0, 5.0)
    flair_abnormality_z: float = 2.0
    label_tolerance: float = 1.0           # voxels within this distance of the GT boundary are correct
    label_rule: str = "auto"               # "auto", "fraction" or "count"
    label_min_fraction: float = 0.9
    label_min_correct: int = 8

    # ---------------------------------------------------------------- confidence model
    split: tuple[int, int, int] = (5, 1, 2)  # train / calibration / test patients
    penalty: str = "l2"                    # "l2" or "l1"
    Cs: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    cv_folds: int = 5
    cv_scoring: str = "roc_auc"
    recalibration: str = "platt"           # "platt", "isotonic" or "none"
    n_bins: int = 10

    # ---------------------------------------------------------------- outputs
    save_maps: bool = True
    save_figures: bool = True

    # ----------------------------------------------------------------
    def override(self, assignments: list[str]) -> "Config":
        """Return a copy with ``name=value`` overrides (values parsed as JSON when possible)."""
        types = {f.name: f.type for f in dataclasses.fields(self)}
        changes: dict[str, Any] = {}
        for item in assignments:
            name, _, raw = item.partition("=")
            name = name.strip()
            if name not in types:
                raise KeyError(f"unknown config field: {name}")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            changes[name] = _coerce(getattr(self, name), value)
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the configuration."""
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(self).items()}


def _coerce(current: Any, value: Any) -> Any:
    """Cast an override to the type of the current value (tuples, paths, numbers)."""
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(current, Path) or (current is None and isinstance(value, str)):
        return Path(value)
    if isinstance(current, bool):
        return bool(value)
    if isinstance(current, float) and isinstance(value, int):
        return float(value)
    return value
