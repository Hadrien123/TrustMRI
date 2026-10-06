"""D. Segmentation evaluation with the official BraTS evaluation package (BraTS_evaluation + Panoptica).

The BraTS glioma config ("gli") computes global and lesion-wise Dice, NSD and HD95, and lesion
detection (TP / FP / FN, precision, recall, F1) after matching connected components. A binary mask
is written with the ET label (3), which the ET, TC and WT groups all contain, so the "et" group
evaluates exactly the given binary mask whatever the region. Infinite HD95 (an empty mask) is
replaced by 373 mm, as in ``brats_evaluation.metrics_parser``.

Caveat (panoptica 2.1.7, so also the official BraTS numbers): the global NSD and HD95 ignore the voxel
spacing (NSD tolerance 0.5 voxel, HD95 in voxels); the lesion-wise ones use it (NSD tolerance =
smallest spacing, i.e. 1 mm on BraTS). Values are kept as the official evaluation reports them.
"""
from __future__ import annotations

import contextlib
import io
from functools import lru_cache

import numpy as np
from brats_evaluation import config_path
from panoptica import Panoptica_Evaluator, disable_citation_reminder

disable_citation_reminder()

HD95_MISS = 373.0
_GROUP = "et"
_KEYS = {  # our name -> Panoptica result key (same metrics as the BraTS leaderboard parser)
    "dice": "global_bin_dsc", "nsd": "global_bin_nsd", "hd95": "global_bin_hd95",
    "lesion_dice": "sq_dsc", "lesion_nsd": "sq_nsd", "lesion_hd95": "sq_hd95",
    "lesion_tp": "tp", "lesion_fp": "fp", "lesion_fn": "fn",
    "lesion_precision": "prec", "lesion_recall": "rec", "lesion_f1": "rq",
}


@lru_cache(maxsize=1)
def _evaluator() -> Panoptica_Evaluator:
    with contextlib.redirect_stdout(io.StringIO()):  # the config prints a notice about shared labels
        return Panoptica_Evaluator.load_from_config(str(config_path("gli")))


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray,
                         spacing: tuple[float, ...] = (1.0, 1.0, 1.0)) -> dict[str, float]:
    """Official BraTS metrics of a binary prediction against a binary reference."""
    evaluator = _evaluator()
    skip = [g for g in evaluator.segmentation_class_groups_names if g != _GROUP]
    result = evaluator.evaluate((pred > 0).astype(np.uint8) * 3, (gt > 0).astype(np.uint8) * 3,
                                voxelspacing=tuple(spacing), skip_groups=skip)[_GROUP].to_dict(True)
    out = {name: float("nan") if result.get(key) is None else float(result[key]) for name, key in _KEYS.items()}
    for name in ("hd95", "lesion_hd95"):
        if np.isinf(out[name]):
            out[name] = HD95_MISS
    return out
