"""Patch labels of the confidence model (training and evaluation only: they need the reference).

A patch is correct (1) when at least ``min_fraction`` of its voxels are correctly segmented; voxels
within ``tolerance`` voxels of the reference boundary always count as correct.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..patches import PatchGrid


def patch_labels(pred: np.ndarray, gt: np.ndarray, grid: PatchGrid, valid: np.ndarray,
                 tolerance: float = 1.0, min_fraction: float = 0.9) -> np.ndarray:
    """1 / 0 (uint8) for each valid patch of ``grid``, in ``np.nonzero(valid)`` order."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    correct = pred == gt
    if tolerance > 0 and gt.any():
        near_boundary = np.where(gt, ndi.distance_transform_edt(gt), ndi.distance_transform_edt(~gt)) <= tolerance
        correct |= near_boundary
    return (grid.mean(correct.astype(np.float32)) >= min_fraction)[valid].astype(np.uint8)
