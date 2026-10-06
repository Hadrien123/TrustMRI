"""C. Distribution of the N generated images.

Items 1-3 need no ground truth (usable on new patients); item 4 (calibration of the spread)
compares the spread with the real image.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from .image_global import psnr
from .image_local import sample_variance
from .ssim import masked_mean, pairwise_ssim_maps


def mean_variance_maps(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-wise mean and unbiased variance across the N samples."""
    return samples.mean(axis=0, dtype=np.float32), sample_variance(samples)


def pairwise_ssim(samples: np.ndarray, mask: np.ndarray, data_range: float = 1.0,
                  win_size: int = 7) -> tuple[list[float], np.ndarray]:
    """SSIM between all N(N-1)/2 pairs of samples.

    Returns:
        (per-pair SSIM inside ``mask``, voxel-wise mean pairwise-SSIM map). With N < 2 the list is
        empty and the map is all ones.
    """
    values: list[float] = []
    acc = np.zeros(samples.shape[1:], dtype=np.float32)
    for _, smap in pairwise_ssim_maps(samples, data_range, win_size):
        values.append(masked_mean(smap, mask))
        acc += smap
    return values, (acc / len(values) if values else np.ones_like(acc))


def pairwise_psnr(samples: np.ndarray, mask: np.ndarray, data_range: float = 1.0) -> list[float]:
    """PSNR (dB) between all pairs of samples inside ``mask``."""
    return [psnr(samples[i], samples[j], mask, data_range) for i, j in combinations(range(len(samples)), 2)]


def interval_coverage(samples: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                      interval: tuple[float, float] = (5.0, 95.0)) -> float:
    """Fraction of voxels in ``mask`` whose true intensity lies within the [lo, hi] percentile
    interval of the N samples (inclusive)."""
    mask = mask.astype(bool)
    if not mask.any():
        return float("nan")
    s = samples[:, mask]
    lo, hi = np.percentile(s, interval, axis=0)
    t = gt[mask]
    return float(((t >= lo) & (t <= hi)).mean())


def spread_error_correlation(local_std: np.ndarray, local_error: np.ndarray) -> float:
    """Spearman correlation between local spread (std) and local error, over finite entries
    (typically patch maps). NaN if fewer than 3 points or a constant input."""
    a, b = np.asarray(local_std, float).ravel(), np.asarray(local_error, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)

