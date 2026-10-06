"""D. Segmentation evaluation of the consensus mask against the reference.

Conventions (BraTS-compatible):
- both masks empty: Dice = NSD = F1 = 1, HD95 = 0;
- exactly one empty: Dice = NSD = 0, HD95 = ``HD95_MISS`` (374 mm, the BraTS 2023 penalty);
- lesion-wise Dice/HD95 (BraTS 2023): each reference lesion is scored against the union of the
  predicted lesions touching its dilated footprint; missed reference lesions and invented
  (false-positive) predicted lesions score Dice 0 / HD95 374; the mean is taken over
  (#reference lesions + #false positives).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .lesions import dilate_lesion, label_lesions, overlap_matrix, reference_lesions

HD95_MISS = 374.0


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice similarity coefficient (1 if both are empty)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = int(pred.sum()) + int(gt.sum())
    return 1.0 if denom == 0 else float(2 * np.logical_and(pred, gt).sum() / denom)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, 1), border_value=0)


def surface_distances(pred: np.ndarray, gt: np.ndarray,
                      spacing: tuple[float, ...] = (1.0, 1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Distances (mm) from each predicted surface voxel to the reference surface, and vice versa.

    Computed in the bounding box of ``pred | gt`` (exact, much smaller distance transforms).
    Both masks must be non-empty.
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    idx = np.argwhere(pred | gt)
    lo, hi = np.maximum(idx.min(0) - 1, 0), np.minimum(idx.max(0) + 2, pred.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    sp, sg = _surface(pred[box]), _surface(gt[box])
    d_to_gt = ndi.distance_transform_edt(~sg, sampling=spacing)
    d_to_pred = ndi.distance_transform_edt(~sp, sampling=spacing)
    return d_to_gt[sp], d_to_pred[sg]


def nsd(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
        tolerance_mm: float = 1.0) -> float:
    """Normalized surface distance: fraction of both surfaces within ``tolerance_mm`` of the other."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    a, b = surface_distances(pred, gt, spacing)
    return float(((a <= tolerance_mm).sum() + (b <= tolerance_mm).sum()) / (a.size + b.size))


def hd95(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, ...] = (1.0, 1.0, 1.0)) -> float:
    """95th-percentile symmetric Hausdorff distance (mm)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return HD95_MISS
    a, b = surface_distances(pred, gt, spacing)
    return float(max(np.percentile(a, 95), np.percentile(b, 95)))


def lesionwise_scores(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
                      dilation: int = 3, min_size: int = 10, compute_hd95: bool = True) -> dict[str, float]:
    """BraTS 2023 lesion-wise Dice (and HD95)."""
    ref_lab, n_ref = reference_lesions(gt, dilation, min_size)
    pred_lab, n_pred = label_lesions(pred, min_size)
    matched: set[int] = set()
    dices, hds = [], []
    for r in range(1, n_ref + 1):
        lesion = ref_lab == r
        hits = np.unique(pred_lab[dilate_lesion(lesion, dilation)])
        hits = hits[hits > 0]
        matched.update(hits.tolist())
        p = np.isin(pred_lab, hits) if hits.size else np.zeros_like(lesion)
        dices.append(dice(p, lesion))
        if compute_hd95:
            hds.append(hd95(p, lesion, spacing) if hits.size else HD95_MISS)
    n_fp = n_pred - len(matched)
    denom = n_ref + n_fp
    out = {"lesion_dice": float(sum(dices) / denom) if denom else 1.0,
           "lesion_n_ref": float(n_ref), "lesion_n_pred": float(n_pred), "lesion_n_fp_brats": float(n_fp)}
    if compute_hd95:
        out["lesion_hd95"] = float((sum(hds) + n_fp * HD95_MISS) / denom) if denom else 0.0
    return out


def lesion_detection(pred: np.ndarray, gt: np.ndarray, rule: str = "any_overlap",
                     iou_threshold: float = 0.1, dilation: int = 3, min_size: int = 10) -> dict[str, float]:
    """Lesion-level detection: TP / FP (invented) / FN (missed), precision, recall, F1.

    Matching rules between a predicted and a reference lesion:
    ``any_overlap`` (share >= 1 voxel), ``dilated_overlap`` (touch the reference lesion dilated by
    ``dilation``, as in BraTS 2023) or ``iou`` (IoU >= ``iou_threshold``).
    Precision (recall) is 1 when there are no predicted (reference) lesions and none of the other
    kind either, 0 otherwise.
    """
    ref_lab, n_ref = reference_lesions(gt, dilation, min_size)
    pred_lab, n_pred = label_lesions(pred, min_size)
    if rule == "dilated_overlap":
        match = np.zeros((n_pred, n_ref), dtype=bool)
        for r in range(1, n_ref + 1):
            hits = np.unique(pred_lab[dilate_lesion(ref_lab == r, dilation)])
            match[hits[hits > 0] - 1, r - 1] = True
    else:
        ov = overlap_matrix(pred_lab, n_pred, ref_lab, n_ref)
        inter = ov[1:, 1:]
        if rule == "any_overlap":
            match = inter > 0
        elif rule == "iou":
            union = ov[1:, :].sum(1)[:, None] + ov[:, 1:].sum(0)[None, :] - inter
            match = inter / np.maximum(union, 1) >= iou_threshold
        else:
            raise ValueError(f"unknown lesion matching rule {rule!r}")
    tp_ref = int(match.any(0).sum()) if n_ref else 0
    tp_pred = int(match.any(1).sum()) if n_pred else 0
    precision = tp_pred / n_pred if n_pred else float(n_ref == 0)
    recall = tp_ref / n_ref if n_ref else float(n_pred == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"lesion_tp": float(tp_ref), "lesion_fp": float(n_pred - tp_pred), "lesion_fn": float(n_ref - tp_ref),
            "lesion_precision": float(precision), "lesion_recall": float(recall), "lesion_f1": float(f1)}


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
                         nsd_tolerance_mm: float = 1.0, lesion_dilation: int = 3, min_size: int = 10,
                         match_rule: str = "any_overlap", match_iou: float = 0.1,
                         compute_hd95: bool = True) -> dict[str, float]:
    """All segmentation scores of ``pred`` (e.g. the consensus mask) against ``gt``."""
    out = {"dice": dice(pred, gt), "nsd": nsd(pred, gt, spacing, nsd_tolerance_mm)}
    if compute_hd95:
        out["hd95"] = hd95(pred, gt, spacing)
    out.update(lesionwise_scores(pred, gt, spacing, lesion_dilation, min_size, compute_hd95))
    out.update(lesion_detection(pred, gt, match_rule, match_iou, lesion_dilation, min_size))
    return out
