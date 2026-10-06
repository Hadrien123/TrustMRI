"""B. Local, patch-wise maps: SSIM vs ground truth, L1/L2 error, variance across samples, failures.

Voxel-wise maps are averaged over the N samples, then aggregated per patch (mean over the brain
voxels of each patch) with a :class:`~trust_mri_eval.patches.PatchGrid`. Patch maps are reported
separately in the tumour region and the healthy brain (majority vote of each patch's brain voxels).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..patches import PatchGrid

#: maps for which a *low* value is bad (all others: high is bad)
LOW_IS_BAD = frozenset({"ssim", "pairwise_ssim"})


def voxel_error_maps(gt: np.ndarray, samples: np.ndarray) -> dict[str, np.ndarray]:
    """Absolute (L1) and squared (L2) error vs ``gt``, averaged over the N samples."""
    l1 = np.zeros(gt.shape, dtype=np.float32)
    l2 = np.zeros(gt.shape, dtype=np.float32)
    for s in samples:
        d = s - gt
        l1 += np.abs(d)
        l2 += d * d
    return {"l1": l1 / len(samples), "l2": l2 / len(samples)}


def sample_variance(samples: np.ndarray) -> np.ndarray:
    """Voxel-wise unbiased variance across samples (zeros when N = 1)."""
    if len(samples) < 2:
        return np.zeros(samples.shape[1:], dtype=np.float32)
    # float64 accumulation: identical samples give exactly 0
    return samples.var(axis=0, ddof=1, dtype=np.float64).astype(np.float32)


def patch_maps(voxel_maps: dict[str, np.ndarray], grid: PatchGrid,
               weights: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Aggregate each voxel map per patch (weighted mean, e.g. over brain voxels)."""
    return {name: grid.mean(vol, weights) for name, vol in voxel_maps.items()}


def patch_regions(grid: PatchGrid, brain: np.ndarray, tumour: np.ndarray | None = None,
                  min_coverage: float = 0.5) -> dict[str, np.ndarray]:
    """Boolean patch masks: ``brain`` (valid patches), and if ``tumour`` is given, ``tumour``
    (majority of the patch's brain voxels are tumour) and ``healthy`` (the other valid patches)."""
    brain = brain.astype(bool)
    valid = grid.coverage(brain) >= min_coverage
    out = {"brain": valid}
    if tumour is not None:
        n_brain = grid.sum(brain)
        n_tumour = grid.sum(tumour.astype(bool) & brain)
        is_tumour = valid & (n_tumour > 0.5 * np.maximum(n_brain, 1))
        out["tumour"], out["healthy"] = is_tumour, valid & ~is_tumour
    return out


def flag_threshold(patch_values: Sequence[np.ndarray], percentile: float = 90.0,
                   low_is_bad: bool = False) -> float:
    """Threshold from training-set patch values: the ``percentile``-th percentile of the pooled
    values (or the (100 - percentile)-th when low values are bad, e.g. SSIM)."""
    pooled = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in patch_values])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return float("nan")
    return float(np.percentile(pooled, 100 - percentile if low_is_bad else percentile))


def failure_map(patch_error: np.ndarray, threshold: float, low_is_bad: bool = False) -> np.ndarray:
    """Binary (uint8) patch map of patches beyond ``threshold``; NaN patches are 0."""
    v = np.nan_to_num(patch_error, nan=-np.inf if not low_is_bad else np.inf)
    flagged = v < threshold if low_is_bad else v > threshold
    return flagged.astype(np.uint8)


def summarize_patch_map(values: np.ndarray, regions: dict[str, np.ndarray],
                        threshold: float | None = None, low_is_bad: bool = False,
                        prefix: str = "") -> dict[str, float]:
    """Mean, worst-tail percentile (p95, or p05 when low is bad) and fraction of patches beyond
    ``threshold`` of a patch map, per region."""
    out: dict[str, float] = {}
    tail = 5 if low_is_bad else 95
    for region, mask in regions.items():
        v = values[mask & np.isfinite(values)]
        key = f"{prefix}{region}"
        out[f"{key}_mean"] = float(v.mean()) if v.size else float("nan")
        out[f"{key}_p{tail:02d}"] = float(np.percentile(v, tail)) if v.size else float("nan")
        if threshold is not None:
            beyond = (v < threshold) if low_is_bad else (v > threshold)
            out[f"{key}_frac_{'below' if low_is_bad else 'above'}_thr"] = \
                float(beyond.mean()) if v.size else float("nan")
    return out
