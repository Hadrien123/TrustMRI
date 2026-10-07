"""Preprocessing shared by every metric. Volumes arrive in BraSyn space (data/io.py): oriented, cropped,
images in [0, 1].

1. binarise the segmentation probabilities: Dice, NSD and F1 compare masks;
2. region of interest (all predicted masks + reference, dilated): where the ANOVA and the confidence
   model look, since far from the tumour every segmentation is trivially right.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .data.types import PatientData


def compute_roi(masks: np.ndarray, gt_mask: np.ndarray | None = None, dilation: int = 5,
                brain_mask: np.ndarray | None = None) -> np.ndarray:
    """Union of all predicted masks (..., X, Y, Z) and the reference if given, dilated by ``dilation``
    voxels, inside the brain. At inference there is no reference: pass ``gt_mask=None``."""
    union = masks.reshape(-1, *masks.shape[-3:]).any(0)
    if gt_mask is not None:
        union |= gt_mask.astype(bool)
    roi = ndi.distance_transform_edt(~union) <= dilation if union.any() else union
    return roi & brain_mask.astype(bool) if brain_mask is not None else roi


@dataclass
class Prepared:
    """One patient in BraSyn space."""

    patient_id: str
    region: str
    affine: np.ndarray
    brain: np.ndarray                   # bool
    imputed: np.ndarray                 # (N, *s) float32 in [0, 1]
    probs: np.ndarray                   # (N, K, *s) float32
    masks: np.ndarray                   # (N, K, *s) bool
    roi: np.ndarray                     # bool
    gt_image: np.ndarray | None = None  # float32 in [0, 1]
    gt_mask: np.ndarray | None = None   # bool

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.brain.shape)  # type: ignore[return-value]

    @property
    def has_ground_truth(self) -> bool:
        return self.gt_image is not None and self.gt_mask is not None


def prepare(data: PatientData, threshold: float = 0.5, roi_dilation: int = 5) -> Prepared:
    """Run the two preprocessing steps of the module docstring."""
    brain = data.brain_mask.astype(bool)
    masks = data.seg_probs >= threshold
    gt_mask = None if data.gt_mask is None else data.gt_mask.astype(bool) & brain
    return Prepared(
        patient_id=data.patient_id, region=data.region, affine=data.affine,
        brain=brain, imputed=data.imputed, probs=data.seg_probs, masks=masks,
        roi=compute_roi(masks, gt_mask, roi_dilation, brain), gt_image=data.gt_image, gt_mask=gt_mask,
    )
