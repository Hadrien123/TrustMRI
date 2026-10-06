"""Patch labels for the confidence model (needs the reference: training/evaluation only).

A patch is *correct* (label 1) when the consensus agrees with the reference on enough of its
voxels; voxels within ``tolerance`` voxels of the reference boundary always count as correct
(boundary ambiguity is not a segmentation failure).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..patches import PatchGrid, grow_box


def boundary_band(gt: np.ndarray, tolerance: float = 1.0) -> np.ndarray:
    """Voxels within ``tolerance`` voxels of the reference boundary, on either side.

    Inside voxels whose distance to the nearest background voxel is <= tolerance, and outside
    voxels whose distance to the nearest reference voxel is <= tolerance.
    """
    gt = gt.astype(bool)
    if tolerance <= 0 or not gt.any() or gt.all():
        return np.zeros_like(gt)
    d_in = ndi.distance_transform_edt(gt)
    d_out = ndi.distance_transform_edt(~gt)
    return np.where(gt, d_in <= tolerance, d_out <= tolerance)


def correct_voxels(pred: np.ndarray, gt: np.ndarray, tolerance: float = 1.0) -> np.ndarray:
    """Voxel-wise correctness: agreement with the reference, or inside the tolerance band."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    return (pred == gt) | boundary_band(gt, tolerance)


def patch_labels(pred: np.ndarray, gt: np.ndarray, grid: PatchGrid, valid: np.ndarray,
                 tolerance: float = 1.0, rule: str = "auto", min_fraction: float = 0.9,
                 min_correct: int = 8) -> np.ndarray:
    """Label (uint8, 1 = correct) of each valid patch of ``grid``, in ``np.nonzero(valid)`` order.

    Rules: ``fraction`` (>= ``min_fraction`` of the patch's voxels correct), ``count``
    (>= ``min_correct`` voxels correct), or ``auto`` (``count`` for 9-voxel in-plane 3x3x1
    patches, i.e. 8 of 9, else ``fraction``). The tolerance band is computed on the grid box grown
    by a margin, which is exact because the grid covers the whole ROI (and so the reference).
    """
    box = grow_box(grid.box, int(np.ceil(tolerance)) + 2, pred.shape)
    inner = tuple(slice(g.start - b.start, g.stop - b.start) for g, b in zip(grid.box, box))
    correct = correct_voxels(pred[box], gt[box], tolerance)[inner]
    n_correct = grid.sum(correct)
    n_inside = grid.sum(np.ones(grid.box_shape, dtype=bool))
    if rule == "auto":
        rule = "count" if sorted(grid.patch) == [1, 3, 3] else "fraction"
    if rule == "count":
        ok = n_correct >= min_correct
    elif rule == "fraction":
        ok = n_correct >= min_fraction * n_inside - 1e-9
    else:
        raise ValueError(f"unknown label rule {rule!r}")
    return ok[valid].astype(np.uint8)


def patch_confusion(pred: np.ndarray, gt: np.ndarray, grid: PatchGrid, valid: np.ndarray) -> np.ndarray:
    """Per valid patch voxel counts (TP, FP, FN, TN) of ``pred`` vs ``gt``; shape (n_valid, 4).

    Used for the QU-BraTS-style filtering curve (Dice of the voxels that are kept)."""
    p, g = grid.crop(pred.astype(bool)), grid.crop(gt.astype(bool))
    counts = [grid.sum(p & g), grid.sum(p & ~g), grid.sum(~p & g), grid.sum(~p & ~g)]
    return np.stack([c[valid] for c in counts], axis=1).astype(np.int64)
