"""The single data contract between the NIfTI loader and the rest of the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PatientData:
    """Everything the evaluation needs for one patient and one tumour region.

    Shapes use ``S = (X, Y, Z)`` for the volume (240 x 240 x 155 for BraTS).

    Attributes:
        patient_id: unique identifier, used for grouping (never split a patient across sets).
        imputed: (N, *S) float32, N stochastic imputations of the missing sequence.
        seg_probs: (N, K, *S) float32 in [0, 1]; ``seg_probs[n, k]`` segments ``imputed[n]`` with
            prompt k. The same K prompts are used for every n (crossed design).
        brain_mask: (*S) uint8 binary.
        gt_image: (*S) float32, the real held-out sequence. Evaluation only.
        gt_mask: (*S) uint8 binary reference for ``region``. Evaluation and labels only.
        region: tumour region evaluated by ``seg_probs`` and ``gt_mask`` ("ET", "TC" or "WT").
        affine: 4x4 voxel-to-world matrix used when saving maps as NIfTI.
    """

    patient_id: str
    imputed: np.ndarray
    seg_probs: np.ndarray
    brain_mask: np.ndarray
    gt_image: np.ndarray | None = None
    gt_mask: np.ndarray | None = None
    region: str = "ET"
    affine: np.ndarray = field(default_factory=lambda: np.eye(4))

    @property
    def shape(self) -> tuple[int, int, int]:
        """Spatial shape S."""
        return tuple(self.brain_mask.shape)  # type: ignore[return-value]

    @property
    def n_imputations(self) -> int:
        return int(self.imputed.shape[0])

    @property
    def n_prompts(self) -> int:
        return int(self.seg_probs.shape[1])

    @property
    def has_ground_truth(self) -> bool:
        return self.gt_image is not None and self.gt_mask is not None

    def validate(self) -> "PatientData":
        """Check shapes, dtypes and value ranges; cast to the canonical dtypes. Returns self."""
        s = self.shape
        if len(s) != 3:
            raise ValueError(f"brain_mask must be 3D, got {s}")
        if self.imputed.ndim != 4 or self.imputed.shape[1:] != s:
            raise ValueError(f"imputed must be (N, *{s}), got {self.imputed.shape}")
        if self.seg_probs.ndim != 5 or self.seg_probs.shape[2:] != s:
            raise ValueError(f"seg_probs must be (N, K, *{s}), got {self.seg_probs.shape}")
        if self.seg_probs.shape[0] != self.imputed.shape[0]:
            raise ValueError("seg_probs and imputed disagree on N")
        for name in ("gt_image", "gt_mask"):
            arr = getattr(self, name)
            if arr is not None and arr.shape != s:
                raise ValueError(f"{name} must have shape {s}, got {arr.shape}")
        self.imputed = self.imputed.astype(np.float32, copy=False)
        self.seg_probs = self.seg_probs.astype(np.float32, copy=False)
        lo, hi = float(self.seg_probs.min()), float(self.seg_probs.max())
        if lo < -1e-6 or hi > 1 + 1e-6:
            raise ValueError(f"seg_probs must lie in [0, 1], got [{lo}, {hi}]")
        self.brain_mask = (self.brain_mask > 0).astype(np.uint8)
        if self.gt_mask is not None:
            self.gt_mask = (self.gt_mask > 0).astype(np.uint8)
        if self.gt_image is not None:
            self.gt_image = self.gt_image.astype(np.float32, copy=False)
        return self
