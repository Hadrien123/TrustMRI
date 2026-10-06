"""3D SSIM maps (float32), numerically identical to ``skimage.metrics.structural_similarity``
with a uniform window and sample covariance, but returning the voxel-wise map so that it can be
averaged inside any region (brain, tumour, healthy) or patch, and reusing per-image local
statistics across pairs.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import ndimage as ndi

K1, K2 = 0.01, 0.03


@dataclass
class LocalStats:
    """Local mean and (sample-normalised) variance of one image over a cubic window."""

    image: np.ndarray
    mean: np.ndarray
    var: np.ndarray
    win_size: int


def local_stats(image: np.ndarray, win_size: int = 7) -> LocalStats:
    """Window mean and unbiased window variance of ``image``."""
    x = image.astype(np.float32, copy=False)
    mu = ndi.uniform_filter(x, size=win_size)
    n = win_size ** x.ndim
    var = (ndi.uniform_filter(x * x, size=win_size) - mu * mu) * (n / (n - 1))
    return LocalStats(x, mu, var, win_size)


def ssim_map(a: np.ndarray | LocalStats, b: np.ndarray | LocalStats, data_range: float = 1.0,
             win_size: int = 7) -> np.ndarray:
    """Voxel-wise SSIM map between two images (or their precomputed :class:`LocalStats`)."""
    sa = a if isinstance(a, LocalStats) else local_stats(a, win_size)
    sb = b if isinstance(b, LocalStats) else local_stats(b, win_size)
    n = sa.win_size ** sa.image.ndim
    cov = (ndi.uniform_filter(sa.image * sb.image, size=sa.win_size) - sa.mean * sb.mean) * (n / (n - 1))
    c1, c2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    num = (2 * sa.mean * sb.mean + c1) * (2 * cov + c2)
    den = (sa.mean ** 2 + sb.mean ** 2 + c1) * (sa.var + sb.var + c2)
    return (num / den).astype(np.float32)


def pairwise_ssim_maps(samples: np.ndarray, data_range: float = 1.0,
                       win_size: int = 7) -> Iterator[tuple[tuple[int, int], np.ndarray]]:
    """Yield ``((i, j), ssim_map)`` for all N(N-1)/2 pairs of ``samples`` (N, *S)."""
    stats = [local_stats(s, win_size) for s in samples]
    for i, j in combinations(range(len(samples)), 2):
        yield (i, j), ssim_map(stats[i], stats[j], data_range)


def masked_mean(values: np.ndarray, mask: np.ndarray | None) -> float:
    """Mean of ``values`` inside ``mask`` (NaN if the mask is empty)."""
    if mask is None:
        return float(np.mean(values))
    mask = mask.astype(bool)
    return float(values[mask].mean()) if mask.any() else float("nan")
