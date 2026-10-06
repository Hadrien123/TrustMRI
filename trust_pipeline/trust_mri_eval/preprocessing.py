"""Preprocessing: brain crop, robust intensity rescaling, binary masks, consensus and ROI.

All downstream computations run on the brain bounding box (``Prepared``), which roughly halves
memory and time on BraTS volumes. ``Prepared.uncrop`` maps results back to the full volume.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .data.types import PatientData
from .patches import bounding_box


# ---------------------------------------------------------------------------- pure functions
def robust_rescale(volume: np.ndarray, mask: np.ndarray,
                   percentiles: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """Map the [p_lo, p_hi] intensity percentiles inside ``mask`` to [0, 1], clip, zero outside.

    Applied per volume before SSIM/PSNR: SSIM is biased on z-scored images with negative values.
    """
    mask = mask.astype(bool)
    out = np.zeros(volume.shape, dtype=np.float32)
    if not mask.any():
        return out
    lo, hi = np.percentile(volume[mask], percentiles)
    scale = hi - lo if hi > lo else 1.0
    out[mask] = np.clip((volume[mask] - lo) / scale, 0.0, 1.0)
    return out


def binarize(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binary masks ``probs >= threshold`` (bool)."""
    return probs >= threshold


def consensus_mask(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Consensus of the N x K probability maps: their mean thresholded at ``threshold``."""
    return probs.reshape(-1, *probs.shape[-3:]).mean(0) >= threshold


def dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    """Dilation by a Euclidean ball of ``radius`` voxels (distance transform: fast for large radii)."""
    mask = mask.astype(bool)
    if radius <= 0 or not mask.any():
        return mask.copy()
    return ndi.distance_transform_edt(~mask) <= radius


def compute_roi(pred_masks: np.ndarray, gt_mask: np.ndarray | None = None, dilation: int = 5,
                brain_mask: np.ndarray | None = None) -> np.ndarray:
    """Region of interest: union of all predicted masks (and the reference, if given), dilated.

    ``pred_masks`` is (..., X, Y, Z). At inference there is no reference: pass ``gt_mask=None``.
    The result is intersected with ``brain_mask`` when given.
    """
    union = pred_masks.reshape(-1, *pred_masks.shape[-3:]).any(0)
    if gt_mask is not None:
        union |= gt_mask.astype(bool)
    roi = dilate(union, dilation)
    if brain_mask is not None:
        roi &= brain_mask.astype(bool)
    return roi


# ---------------------------------------------------------------------------- patient container
@dataclass
class Prepared:
    """One patient, cropped to the brain bounding box, intensities rescaled to [0, 1]."""

    patient_id: str
    region: str
    spacing: tuple[float, float, float]
    affine: np.ndarray
    full_shape: tuple[int, int, int]
    crop: tuple[slice, slice, slice]
    brain: np.ndarray                 # bool
    imputed: np.ndarray               # (N, *s) float32 in [0, 1]
    probs: np.ndarray                 # (N, K, *s) float32
    masks: np.ndarray                 # (N, K, *s) bool
    mean_prob: np.ndarray             # float32
    consensus: np.ndarray             # bool
    roi: np.ndarray                   # bool, includes the reference when available
    gt_image: np.ndarray | None = None  # float32 in [0, 1]
    gt_mask: np.ndarray | None = None   # bool
    flair: np.ndarray | None = None     # float32 in [0, 1]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.brain.shape)  # type: ignore[return-value]

    @property
    def has_ground_truth(self) -> bool:
        return self.gt_image is not None and self.gt_mask is not None

    @property
    def tumour(self) -> np.ndarray | None:
        """Reference tumour region inside the brain."""
        return None if self.gt_mask is None else self.gt_mask & self.brain

    @property
    def healthy(self) -> np.ndarray | None:
        """Brain outside the reference tumour."""
        return None if self.gt_mask is None else self.brain & ~self.gt_mask

    def uncrop(self, volume: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """Embed a cropped (*s) volume into the full (*S) volume."""
        out = np.full(self.full_shape, fill, dtype=volume.dtype)
        out[self.crop] = volume
        return out


def prepare(data: PatientData, threshold: float = 0.5,
            percentiles: tuple[float, float] = (0.5, 99.5), roi_dilation: int = 5,
            crop_margin: int = 2) -> Prepared:
    """Crop to the brain, rescale every image volume, binarise and build consensus and ROI."""
    brain_full = data.brain_mask.astype(bool)
    crop = bounding_box(brain_full, margin=crop_margin)
    brain = brain_full[crop]

    imputed = np.stack([robust_rescale(v[crop], brain, percentiles) for v in data.imputed])
    probs = np.ascontiguousarray(data.seg_probs[(..., *crop)], dtype=np.float32)
    masks = binarize(probs, threshold)
    mean_prob = probs.mean((0, 1), dtype=np.float32)
    gt_mask = None if data.gt_mask is None else data.gt_mask[crop].astype(bool) & brain
    gt_image = None if data.gt_image is None else robust_rescale(data.gt_image[crop], brain, percentiles)
    flair = None if data.real_flair is None else robust_rescale(data.real_flair[crop], brain, percentiles)

    return Prepared(
        patient_id=data.patient_id, region=data.region, spacing=tuple(data.spacing),
        affine=data.affine, full_shape=data.shape, crop=crop, brain=brain,
        imputed=imputed, probs=probs, masks=masks, mean_prob=mean_prob,
        consensus=mean_prob >= threshold,
        roi=compute_roi(masks, gt_mask, roi_dilation, brain),
        gt_image=gt_image, gt_mask=gt_mask, flair=flair,
    )
