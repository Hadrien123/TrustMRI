"""Patch features for the local confidence model. GROUND-TRUTH FREE BY CONSTRUCTION.

:func:`compute_features` takes only what is available on a new patient (imputed samples,
segmentation probabilities, brain mask, ROI and optionally the real FLAIR), so training and
inference run exactly the same code. SSIM/Dice against the ground truth belong to evaluation and
labels, never here.

Feature groups (used for contribution maps):

- ``imputation``: mean intensity, variance across the N samples, mean pairwise SSIM between samples;
- ``segmentation``: mean probability, entropy, agreement, ANOVA variances (imputation / prompt /
  interaction), signed and absolute distance to the consensus boundary, pooled pairwise Dice;
- ``anatomy`` (if FLAIR is given): fraction of the patch inside the real FLAIR abnormality;
- ``context``: every feature above smoothed with Gaussian sigma in ``context_sigmas`` voxels.

Each feature also has a ``source`` (imputation / segmentation / anatomy), which context features
inherit; ablations drop features by source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from ..metrics.distribution import pairwise_ssim
from ..metrics.image_local import sample_variance
from ..patches import PatchGrid, bounding_box, grow_box
from ..uncertainty.anova import anova_maps
from ..uncertainty.confidence_maps import agreement_map, binary_entropy, signed_distance

GROUPS = ("imputation", "segmentation", "anatomy", "context")


@dataclass
class PatchFeatures:
    """Feature matrix of the valid patches of one patient.

    ``X[i]`` describes patch ``coords[i]`` of ``grid``; ``valid`` marks those patches on the grid.
    """

    X: np.ndarray                 # (n_valid, F) float32
    names: list[str]
    groups: list[str]
    sources: list[str]
    grid: PatchGrid
    valid: np.ndarray             # (nx, ny, nz) bool

    @property
    def coords(self) -> tuple[np.ndarray, ...]:
        return np.nonzero(self.valid)

    def to_grid(self, values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """Scatter per-valid-patch ``values`` back onto the (nx, ny, nz) grid."""
        out = np.full(self.valid.shape, fill, dtype=np.float32)
        out[self.valid] = values
        return out


def feature_grid(roi: np.ndarray, patch_size: tuple[int, int, int]) -> PatchGrid:
    """Patch grid over the ROI bounding box, aligned to the global patch lattice."""
    return PatchGrid.over(roi.shape, patch_size, bounding_box(roi))


def valid_patches(grid: PatchGrid, roi: np.ndarray, min_coverage: float = 0.5) -> np.ndarray:
    """Patches with at least ``min_coverage`` of their voxels in the ROI."""
    return grid.coverage(roi) >= min_coverage


def flair_abnormality(flair: np.ndarray, brain: np.ndarray, z: float = 2.0) -> np.ndarray:
    """FLAIR hyperintensity: robust z-score (median / 1.4826 MAD over the brain) above ``z``."""
    brain = brain.astype(bool)
    vals = flair[brain]
    med = np.median(vals)
    mad = 1.4826 * np.median(np.abs(vals - med)) or 1.0
    return brain & ((flair - med) / mad > z)


def compute_features(imputed: np.ndarray, probs: np.ndarray, brain: np.ndarray, roi: np.ndarray,
                     patch_size: tuple[int, int, int] = (3, 3, 3), threshold: float = 0.5,
                     spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
                     context_sigmas: tuple[float, ...] = (2.0, 5.0), flair: np.ndarray | None = None,
                     flair_z: float = 2.0, min_coverage: float = 0.5, win_size: int = 7,
                     anova_on: str = "probs", anova_chunk: int = 200_000) -> PatchFeatures:
    """Compute the patch features of one patient (ground-truth free).

    Args:
        imputed: (N, *S) imputed images rescaled to [0, 1].
        probs: (N, K, *S) segmentation probabilities.
        brain, roi: boolean masks (*S). At inference the ROI must be built without the reference.
        flair: optional real FLAIR rescaled to [0, 1].
    """
    grid = feature_grid(roi, patch_size)
    valid = valid_patches(grid, roi, min_coverage)
    margin = int(math.ceil(3 * max(context_sigmas, default=0)))
    fbox = grow_box(grid.box, margin, roi.shape)                      # feature box incl. context margin
    inner = tuple(slice(g.start - f.start, g.stop - f.start) for g, f in zip(grid.box, fbox))

    maps, ratios = _voxel_maps(imputed[(..., *fbox)], probs[(..., *fbox)], brain[fbox], roi[fbox],
                               threshold, spacing, win_size, anova_on, anova_chunk)
    sources = {name: src for name, (_, src) in maps.items()}
    sources.update({name: "segmentation" for name in ratios})
    if flair is not None:
        maps["anat_flair_abnormal"] = (flair_abnormality(flair, brain, flair_z)[fbox].astype(np.float32), "anatomy")
        sources["anat_flair_abnormal"] = "anatomy"

    columns: dict[str, np.ndarray] = {}
    for name, (vol, _) in maps.items():
        columns[name] = grid.mean(vol[inner])[valid]
    for name, (num, den) in ratios.items():
        columns[name] = _ratio(grid.sum(num[inner]), grid.sum(den[inner]))[valid]
    base = list(columns)
    for sigma in context_sigmas:
        for name in base:
            key = f"ctx{sigma:g}_{name}"
            if name in ratios:
                num, den = (ndi.gaussian_filter(a, sigma)[inner] for a in ratios[name])
                columns[key] = _ratio(grid.sum(num), grid.sum(den))[valid]
            else:
                columns[key] = grid.mean(ndi.gaussian_filter(maps[name][0], sigma)[inner])[valid]
            sources[key] = sources[name]

    names = list(columns)
    X = np.stack([columns[n] for n in names], axis=1).astype(np.float32) if names else \
        np.zeros((int(valid.sum()), 0), np.float32)
    groups = [("context" if n.startswith("ctx") else sources[n]) for n in names]
    return PatchFeatures(X=np.nan_to_num(X), names=names, groups=groups,
                         sources=[sources[n] for n in names], grid=grid, valid=valid)


def _ratio(num: np.ndarray, den: np.ndarray, empty: float = 1.0) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), empty).astype(np.float32)


def _voxel_maps(imputed, probs, brain, roi, threshold, spacing, win_size, anova_on, anova_chunk):
    """Voxel maps on the feature box: ``{name: (map, source)}`` and ratio maps ``{name: (num, den)}``."""
    n, k = probs.shape[:2]
    masks = probs >= threshold
    mean_prob = probs.mean((0, 1), dtype=np.float32)
    consensus = mean_prob >= threshold
    sd = signed_distance(consensus, spacing)

    maps: dict[str, tuple[np.ndarray, str]] = {
        "imp_mean_intensity": (imputed.mean(0, dtype=np.float32), "imputation"),
        "imp_variance": (sample_variance(imputed), "imputation"),
        "imp_pairwise_ssim": (pairwise_ssim(imputed, brain, 1.0, win_size)[1], "imputation"),
        "seg_mean_prob": (mean_prob, "segmentation"),
        "seg_entropy": (binary_entropy(mean_prob), "segmentation"),
        "seg_agreement": (agreement_map(masks), "segmentation"),
    }
    anova = anova_maps(masks.astype(np.float32) if anova_on == "masks" else probs, roi, anova_chunk,
                       keys=("var_imputation", "var_prompt", "var_interaction"))
    for comp in ("imputation", "prompt", "interaction"):
        maps[f"seg_var_{comp}"] = (anova[f"var_{comp}"], "segmentation")
    maps["seg_signed_distance"] = (sd, "segmentation")
    maps["seg_abs_distance"] = (np.abs(sd), "segmentation")

    # pooled pairwise Dice of the M = N*K masks: sum_v c(c-1) / ((M-1) sum_v c), c = #masks at voxel v
    m_total = n * k
    count = masks.reshape(m_total, *masks.shape[2:]).sum(0, dtype=np.float32)
    ratios = {"seg_pairwise_dice": (count * (count - 1), (m_total - 1) * count)} if m_total > 1 else {}
    return maps, ratios
