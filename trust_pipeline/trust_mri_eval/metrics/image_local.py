"""Local (patch-wise) quality of one imputed image against the real one: SSIM, L1 and L2 per patch.

Each voxel map is averaged over the brain voxels of every patch of a :class:`PatchGrid`.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from ..patches import PatchGrid


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Voxel-wise SSIM map (skimage), with the Gaussian window of BraSyn's ``util.ssim`` (sigma 1.5)."""
    return structural_similarity(a, b, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False, full=True)[1].astype(np.float32)


def local_maps(gt: np.ndarray, image: np.ndarray, brain: np.ndarray, grid: PatchGrid) -> dict[str, np.ndarray]:
    """Patch maps (nx, ny, nz) of SSIM, L1 and L2 between ``image`` and ``gt`` (NaN outside the brain)."""
    diff = image - gt
    voxel_maps = {"ssim": ssim_map(gt, image), "l1": np.abs(diff), "l2": diff * diff}
    return {name: grid.mean(values, brain) for name, values in voxel_maps.items()}
