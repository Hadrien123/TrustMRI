import inspect

import numpy as np
import pytest

from trust_mri_eval.model.features import compute_features
from trust_mri_eval.model.labels import boundary_band, patch_confusion, patch_labels
from trust_mri_eval.model.logistic import (Recalibrator, evaluate, fit_confidence_model, qubrats_curve,
                                           reliability, select_columns, split_patients)
from trust_mri_eval.patches import PatchGrid
from trust_mri_eval.preprocessing import compute_roi, prepare


@pytest.fixture(scope="module")
def prep(small_patient):
    return prepare(small_patient)


@pytest.fixture(scope="module")
def feats(prep):
    return compute_features(prep.imputed, prep.probs, prep.brain, prep.roi, flair=prep.flair)


def test_features_and_labels_shapes_match(prep, feats):
    y = patch_labels(prep.consensus, prep.gt_mask, feats.grid, feats.valid)
    assert feats.X.shape[0] == y.shape[0] == feats.valid.sum() > 0
    assert feats.X.shape[1] == len(feats.names) == len(feats.groups) == len(feats.sources)
    assert np.isfinite(feats.X).all()
    assert set(feats.groups) == {"imputation", "segmentation", "anatomy", "context"}
    assert patch_confusion(prep.consensus, prep.gt_mask, feats.grid, feats.valid).shape == (y.size, 4)
    assert 0 < y.mean() < 1                    # both classes present on synthetic data


def test_features_do_not_depend_on_ground_truth(prep):
    """Same inputs, same ROI, different references -> identical features."""
    roi = compute_roi(prep.masks, None, 5, prep.brain)          # inference ROI: no reference
    a = compute_features(prep.imputed, prep.probs, prep.brain, roi)
    assert not any(p.startswith("gt") for p in inspect.signature(compute_features).parameters)
    b = compute_features(prep.imputed.copy(), prep.probs.copy(), prep.brain, roi)
    np.testing.assert_array_equal(a.X, b.X)


def test_labels_identical_masks_all_correct(prep, feats):
    y = patch_labels(prep.gt_mask, prep.gt_mask, feats.grid, feats.valid)
    assert y.all()


def test_label_rules():
    gt = np.zeros((3, 3, 1), bool)
    pred = gt.copy()
    pred[0, 0, 0] = True                       # 1 wrong voxel of 9
    grid = PatchGrid.over(gt.shape, (3, 3, 1))
    valid = np.ones(grid.n, bool)
    assert patch_labels(pred, gt, grid, valid, tolerance=0, rule="auto")[0] == 1      # 8 of 9
    pred[0, 1, 0] = True
    assert patch_labels(pred, gt, grid, valid, tolerance=0, rule="auto")[0] == 0      # 7 of 9


def test_boundary_band():
    gt = np.zeros((9, 9, 9), bool)
    gt[3:6, 3:6, 3:6] = True
    band = boundary_band(gt, 1.0)
    assert band[3, 4, 4] and band[2, 4, 4] and not band[1, 4, 4] and not band[4, 4, 4]


def test_split_patients_disjoint():
    ids = [f"p{i}" for i in range(8)]
    s = split_patients(ids, 5, 1, 2, seed=3)
    sets = [set(v) for v in s.values()]
    assert sum(map(len, sets)) == 8 and set.union(*sets) == set(ids)
    assert all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:])
    with pytest.raises(ValueError):
        split_patients(ids, 6, 2, 2)


def _toy(n_pat=4, n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_pat * n, 3)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.standard_normal(len(X)) > -1).astype(np.uint8)
    patient = np.repeat([f"p{i}" for i in range(n_pat)], n)
    return X, y, patient


def test_logistic_fit_contributions_and_evaluation():
    X, y, patient = _toy()
    m = fit_confidence_model(X, y, patient, ["a", "b", "c"], ["imputation", "segmentation", "context"],
                             Cs=(0.01, 1.0), cv_folds=4)
    assert m.C in (0.01, 1.0) and len(m.cv_scores) == 2
    p = m.predict(X)
    assert ((0 <= p) & (p <= 1)).all()
    logit = np.log(m.predict_raw(X) / (1 - m.predict_raw(X)))
    np.testing.assert_allclose(m.contributions(X).sum(1) + m.pipeline[-1].intercept_[0], logit, rtol=1e-4, atol=1e-4)
    assert set(m.group_contributions(X)) == {"imputation", "segmentation", "context"}
    ev = evaluate(p, y)
    assert ev["auroc_incorrect"] > 0.85
    m.recalibrator = Recalibrator("platt").fit(m.predict_raw(X), y)
    assert evaluate(m.predict(X), y)["ece"] < ev["ece"] + 1e-9 or ev["ece"] < 0.02
    only = fit_confidence_model(X, y, patient, ["a", "b", "c"], ["i", "s", "c"],
                                columns=select_columns(["a", "b", "c"], ["i", "s", "c"], include=["b"]), Cs=(1.0,))
    assert only.feature_names == ["b"]


def test_reliability_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    y = (rng.random(p.size) < p).astype(int)
    assert reliability(p, y)["ece"] < 0.01


def test_qubrats_curve_endpoints():
    conf = np.array([0.9, 0.2, 0.6])
    cm = np.array([[10, 0, 0, 5], [0, 4, 4, 1], [3, 1, 1, 0]])
    q = qubrats_curve(conf, cm, np.array(["a", "a", "a"]), thresholds=np.array([0.0, 0.5, 1.01]))
    assert q["filtered"][0] == 0 and q["filtered"][-1] == 1
    assert q["dice"][0] == pytest.approx(26 / (26 + 5 + 5))     # all kept
    assert q["dice"][1] == pytest.approx(26 / (26 + 1 + 1))     # uncertain patch filtered
    assert q["dice"][-1] == 1.0                                 # nothing kept
