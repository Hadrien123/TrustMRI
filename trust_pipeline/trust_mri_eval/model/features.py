"""Patch features of the confidence model, computed WITHOUT ground truth from the N x K outputs.

- ``variance``: variance across the N imputed images (imputation);
- ``pairwise_ssim``: mean SSIM between the N imputed images (imputation);
- ``pairwise_dice``: Dice between the N x K segmentations (segmentation), pooled per patch as
  sum c(c-1) / ((M-1) sum c), where c is the number of the M masks that are positive at a voxel
  (1 when no mask is positive).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..patches import PatchGrid, bounding_box

NAMES = ("variance", "pairwise_ssim", "pairwise_dice")


@dataclass
class PatchFeatures:
    """``X[i]`` describes the i-th valid patch of ``grid`` (``valid`` marks them on the grid)."""

    X: np.ndarray        # (n_valid, 3) float32
    grid: PatchGrid
    valid: np.ndarray    # (nx, ny, nz) bool

    def to_grid(self, values: np.ndarray) -> np.ndarray:
        """Scatter per-valid-patch values back onto the grid (NaN elsewhere)."""
        out = np.full(self.valid.shape, np.nan, dtype=np.float32)
        out[self.valid] = values
        return out


def compute_features(variance: np.ndarray, pairwise_ssim: np.ndarray, masks: np.ndarray, roi: np.ndarray,
                     patch_size: tuple[int, int, int] = (3, 3, 3), min_coverage: float = 0.5) -> PatchFeatures:
    """Features of the patches covering the ROI (at least ``min_coverage`` of the patch inside it).

    Args:
        variance, pairwise_ssim: voxel maps across the N imputed images.
        masks: (N, K, *S) binary segmentations.
        roi: region of interest; at inference build it without the reference.
    """
    grid = PatchGrid.over(roi.shape, patch_size, bounding_box(roi))
    valid = grid.coverage(roi) >= min_coverage
    count = masks.reshape(-1, *masks.shape[-3:]).sum(0, dtype=np.float32)
    m = masks.shape[0] * masks.shape[1]
    num, den = grid.sum(count * (count - 1)), grid.sum((m - 1) * count)
    dice = np.where(den > 0, num / np.where(den > 0, den, 1), 1.0)
    X = np.stack([grid.mean(variance)[valid], grid.mean(pairwise_ssim)[valid], dice[valid]], axis=1)
    return PatchFeatures(np.nan_to_num(X).astype(np.float32), grid, valid)
