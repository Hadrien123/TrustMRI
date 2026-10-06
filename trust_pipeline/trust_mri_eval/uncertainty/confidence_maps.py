"""Voxel-wise uncertainty maps from the N x K segmentations (no ground truth needed)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def agreement_map(masks: np.ndarray) -> np.ndarray:
    """Fraction of the N x K binary masks that predict tumour at each voxel (float32 in [0, 1])."""
    flat = masks.reshape(-1, *masks.shape[-3:])
    return flat.mean(axis=0, dtype=np.float32)


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Entropy in bits of a Bernoulli(p) variable (0 at p in {0, 1}, 1 at p = 0.5)."""
    p = np.clip(p.astype(np.float32), 1e-7, 1 - 1e-7)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return np.where((p <= 1e-7) | (p >= 1 - 1e-7), 0.0, h).astype(np.float32)


def entropy_map(mean_prob: np.ndarray) -> np.ndarray:
    """Entropy (bits) of the mean probability over the N x K segmentations."""
    return binary_entropy(mean_prob)


def signed_distance(mask: np.ndarray, spacing: tuple[float, ...] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Signed distance (mm) to the boundary of ``mask``: negative inside, positive outside.

    Inside voxels get -(distance to the nearest outside voxel), outside voxels +(distance to the
    nearest inside voxel). An empty mask gives the volume diagonal everywhere.
    """
    mask = mask.astype(bool)
    if not mask.any():
        diag = float(np.linalg.norm(np.array(mask.shape) * np.array(spacing)))
        return np.full(mask.shape, diag, dtype=np.float32)
    if mask.all():
        diag = float(np.linalg.norm(np.array(mask.shape) * np.array(spacing)))
        return np.full(mask.shape, -diag, dtype=np.float32)
    outside = ndi.distance_transform_edt(~mask, sampling=spacing)
    inside = ndi.distance_transform_edt(mask, sampling=spacing)
    return (outside - inside).astype(np.float32)
