import numpy as np

from trust_mri_eval.metrics.segmentation import segmentation_metrics


def ball(shape, center, r):
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum((x - c) ** 2 for x, c in zip(g, center)) <= r * r


REF = ball((40, 40, 40), (10, 10, 10), 4) | ball((40, 40, 40), (28, 28, 28), 5)


def test_identical_masks():
    m = segmentation_metrics(REF, REF)
    assert m["dice"] == m["nsd"] == m["lesion_f1"] == 1 and m["hd95"] == 0


def test_invented_lesion_lowers_precision():
    m = segmentation_metrics(REF | ball(REF.shape, (10, 30, 30), 3), REF)
    assert m["lesion_precision"] < 1 and m["lesion_recall"] == 1
