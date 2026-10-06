"""A. Ground truth vs generated image, global scores (per imputed sample, then mean +- std over N).

Inputs are expected rescaled to [0, 1] (``preprocessing.robust_rescale``); all scores are
restricted to the brain mask.
"""
from __future__ import annotations

import numpy as np

from .ssim import masked_mean, ssim_map


def mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute error (L1) inside ``mask``."""
    return masked_mean(np.abs(a - b), mask)


def mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean squared error (L2) inside ``mask``."""
    d = a - b
    return masked_mean(d * d, mask)


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio (dB) inside ``mask``; +inf for identical images."""
    m = mse(a, b, mask)
    if m == 0:
        return float("inf")
    return float(10 * np.log10(data_range ** 2 / m))


def ssim_by_region(ssim: np.ndarray, regions: dict[str, np.ndarray | None]) -> dict[str, float]:
    """Average a voxel-wise SSIM map inside each named region (NaN for empty/missing regions)."""
    return {name: (masked_mean(ssim, m) if m is not None else float("nan")) for name, m in regions.items()}


def image_global_metrics(gt: np.ndarray, samples: np.ndarray, brain: np.ndarray, tumour: np.ndarray,
                         ssim_maps: list[np.ndarray] | None = None, data_range: float = 1.0,
                         win_size: int = 7) -> dict[str, list[float]]:
    """Per-sample L1, L2, PSNR and SSIM (whole brain, tumour, healthy brain).

    Args:
        gt: ground-truth image (*S), in [0, 1].
        samples: imputed images (N, *S), in [0, 1].
        brain, tumour: boolean masks; healthy = brain & ~tumour.
        ssim_maps: optional precomputed SSIM maps (one per sample) to avoid recomputation.

    Returns:
        ``{metric: [value for each sample]}``; summarise with :func:`mean_std`.
    """
    brain, tumour = brain.astype(bool), tumour.astype(bool)
    regions = {"brain": brain, "tumour": tumour & brain, "healthy": brain & ~tumour}
    out: dict[str, list[float]] = {k: [] for k in
                                   ("l1", "l2", "psnr", "ssim_brain", "ssim_tumour", "ssim_healthy")}
    for n, s in enumerate(samples):
        out["l1"].append(mae(gt, s, brain))
        out["l2"].append(mse(gt, s, brain))
        out["psnr"].append(psnr(gt, s, brain, data_range))
        smap = ssim_maps[n] if ssim_maps is not None else ssim_map(gt, s, data_range, win_size)
        for region, value in ssim_by_region(smap, regions).items():
            out[f"ssim_{region}"].append(value)
    return out


def mean_std(per_sample: dict[str, list[float]], prefix: str = "") -> dict[str, float]:
    """``{prefix+metric+'_mean': ..., prefix+metric+'_std': ...}`` over samples.

    NaN entries are ignored; infinite entries (PSNR of identical images) are ignored unless all
    entries are infinite, in which case the mean is inf and the std 0.
    """
    out = {}
    for name, values in per_sample.items():
        v = np.asarray(values, dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            m, s = float("nan"), float("nan")
        elif np.isinf(v).all():
            m, s = float(v[0]), 0.0
        else:
            v = v[np.isfinite(v)]
            m, s = float(v.mean()), float(v.std())
        out[f"{prefix}{name}_mean"], out[f"{prefix}{name}_std"] = m, s
    return out
