"""Connected-component (lesion) utilities, BraTS 2023 conventions.

Lesions are 3D connected components with 26-connectivity (3x3x3 structure). Components smaller
than ``min_size`` voxels are removed. Reference lesions are defined after dilating the reference
mask by ``dilation`` voxels (BraTS 2023: 18-connected structure, iterated), so that nearby
fragments of one lesion count as a single lesion; labels are then restricted to the original
reference voxels.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

try:  # optional, faster
    import cc3d  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on the environment
    cc3d = None

STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)
DILATION_STRUCTURE = ndi.generate_binary_structure(3, 2)


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """26-connected components of a binary mask: (int32 labels, number of components)."""
    mask = mask.astype(bool)
    if cc3d is not None:
        labels, n = cc3d.connected_components(mask, connectivity=26, return_N=True)
        return labels.astype(np.int32), int(n)
    labels, n = ndi.label(mask, structure=STRUCTURE_26)
    return labels.astype(np.int32), int(n)


def remove_small(labels: np.ndarray, n: int, min_size: int) -> tuple[np.ndarray, int]:
    """Drop components with fewer than ``min_size`` voxels and relabel 1..n' (order preserved)."""
    if n == 0 or min_size <= 1:
        return labels, n
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = sizes >= min_size
    keep[0] = False
    new_ids = np.zeros(n + 1, dtype=np.int32)
    new_ids[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return new_ids[labels], int(keep.sum())


def label_lesions(mask: np.ndarray, min_size: int = 10) -> tuple[np.ndarray, int]:
    """Lesions of a predicted mask: 26-connected components with at least ``min_size`` voxels."""
    return remove_small(*connected_components(mask), min_size)


def reference_lesions(gt_mask: np.ndarray, dilation: int = 3, min_size: int = 10) -> tuple[np.ndarray, int]:
    """Reference lesions, BraTS 2023 style: components of the dilated reference, restricted to the
    original reference voxels, small ones removed."""
    gt = gt_mask.astype(bool)
    if dilation > 0 and gt.any():
        dilated = ndi.binary_dilation(gt, structure=DILATION_STRUCTURE, iterations=dilation)
        labels, n = connected_components(dilated)
        labels = np.where(gt, labels, 0).astype(np.int32)
    else:
        labels, n = connected_components(gt)
    return remove_small(labels, n, min_size)


def dilate_lesion(lesion: np.ndarray, dilation: int) -> np.ndarray:
    """Dilate one lesion mask with the BraTS structure, computed in its bounding box."""
    if dilation <= 0 or not lesion.any():
        return lesion.copy()
    idx = np.argwhere(lesion)
    lo = np.maximum(idx.min(0) - dilation - 1, 0)
    hi = np.minimum(idx.max(0) + dilation + 2, lesion.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    out = np.zeros_like(lesion, dtype=bool)
    out[box] = ndi.binary_dilation(lesion[box], structure=DILATION_STRUCTURE, iterations=dilation)
    return out


def overlap_matrix(pred_labels: np.ndarray, n_pred: int, ref_labels: np.ndarray,
                   n_ref: int) -> np.ndarray:
    """(n_pred + 1, n_ref + 1) voxel counts of each (pred label, ref label) pair (row/col 0 = background)."""
    pairs = pred_labels.astype(np.int64).ravel() * (n_ref + 1) + ref_labels.ravel()
    return np.bincount(pairs, minlength=(n_pred + 1) * (n_ref + 1)).reshape(n_pred + 1, n_ref + 1)
