import numpy as np
import pytest

from trust_mri_eval.metrics.lesions import label_lesions, reference_lesions
from trust_mri_eval.metrics.segmentation import (HD95_MISS, dice, hd95, lesion_detection, lesionwise_scores,
                                                 nsd, segmentation_metrics)


def ball(shape, center, r):
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum((x - c) ** 2 for x, c in zip(g, center)) <= r * r


@pytest.fixture
def two_lesions():
    shape = (40, 40, 40)
    return ball(shape, (10, 10, 10), 4) | ball(shape, (28, 28, 28), 5)


def test_identical_masks(two_lesions):
    m = segmentation_metrics(two_lesions, two_lesions)
    for key in ("dice", "nsd", "lesion_dice", "lesion_f1", "lesion_precision", "lesion_recall"):
        assert m[key] == 1.0, key
    assert m["hd95"] == 0 and m["lesion_hd95"] == 0 and m["lesion_n_ref"] == 2


def test_empty_conventions():
    e = np.zeros((10, 10, 10), bool)
    f = e.copy()
    f[4:7, 4:7, 4:7] = True
    assert dice(e, e) == 1 and nsd(e, e) == 1 and hd95(e, e) == 0
    assert dice(f, e) == 0 and nsd(f, e) == 0 and hd95(f, e) == HD95_MISS
    assert lesion_detection(e, e)["lesion_f1"] == 1


def test_invented_lesion_lowers_precision(two_lesions):
    pred = two_lesions | ball(two_lesions.shape, (10, 30, 30), 3)
    m = segmentation_metrics(pred, two_lesions)
    assert m["lesion_precision"] < 1 and m["lesion_recall"] == 1 and m["lesion_fp"] == 1
    assert m["lesion_dice"] == pytest.approx(2 / 3)       # (1 + 1 + 0) / (2 ref + 1 FP)


def test_missed_lesion_lowers_recall(two_lesions):
    pred = ball(two_lesions.shape, (28, 28, 28), 5)
    m = segmentation_metrics(pred, two_lesions)
    assert m["lesion_recall"] == 0.5 and m["lesion_precision"] == 1 and m["lesion_fn"] == 1
    assert m["lesion_dice"] == pytest.approx(0.5)
    assert m["lesion_hd95"] == pytest.approx(HD95_MISS / 2)


def test_nsd_tolerance():
    shape = (30, 30, 30)
    gt, pred = ball(shape, (15, 15, 15), 8), ball(shape, (15, 15, 15), 10)   # surfaces 1.4-2.45 voxels apart
    assert nsd(pred, gt, tolerance_mm=1.0) == 0.0
    assert 0 < nsd(pred, gt, tolerance_mm=2.0) < 1
    assert nsd(pred, gt, tolerance_mm=2.5) == 1.0
    assert nsd(pred, gt, spacing=(0.5, 0.5, 0.5), tolerance_mm=1.25) == 1.0   # distances halved in mm


def test_reference_lesions_merge_nearby_fragments():
    gt = np.zeros((30, 30, 30), bool)
    gt[10:15, 10:15, 10:15] = True
    gt[10:15, 10:15, 17:22] = True        # 2-voxel gap: separate components, merged after dilation
    assert label_lesions(gt)[1] == 2
    assert reference_lesions(gt, dilation=3)[1] == 1


def test_small_components_removed():
    m = np.zeros((20, 20, 20), bool)
    m[2, 2, 2] = True
    m[10:13, 10:13, 10:13] = True
    labels, n = label_lesions(m, min_size=10)
    assert n == 1 and labels[2, 2, 2] == 0


def test_iou_rule(two_lesions):
    pred = two_lesions.copy()
    pred[10, 10, 10:15] = True
    shifted = np.roll(two_lesions, 7, axis=0)
    assert lesion_detection(shifted, two_lesions, rule="any_overlap")["lesion_f1"] == 1
    assert lesion_detection(shifted, two_lesions, rule="iou", iou_threshold=0.5)["lesion_f1"] == 0
