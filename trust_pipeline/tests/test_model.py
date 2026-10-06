import numpy as np

from trust_mri_eval.model.features import compute_features
from trust_mri_eval.model.labels import patch_labels
from trust_mri_eval.model.logistic import evaluate, fit_confidence_model, split_patients
from trust_mri_eval.preprocessing import compute_roi, prepare


def test_features_and_labels_match(small_patient):
    prep = prepare(small_patient)
    # inference-style ROI and no ground-truth argument: features never see the reference
    roi = compute_roi(prep.masks, None, 5, prep.brain)
    feats = compute_features(prep.imputed, prep.probs, prep.brain, roi)
    y = patch_labels(prep.consensus, prep.gt_mask, feats.grid, feats.valid)
    assert feats.X.shape == (y.size, len(feats.names)) and np.isfinite(feats.X).all()


def test_split_patients_disjoint():
    s = split_patients([f"p{i}" for i in range(8)], 5, 1, 2)
    sets = [set(v) for v in s.values()]
    assert sum(map(len, sets)) == len(set.union(*sets)) == 8


def test_logistic_detects_errors():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1600, 3)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.standard_normal(len(X)) > -1).astype(np.uint8)
    patient = np.repeat([f"p{i}" for i in range(4)], 400)
    model = fit_confidence_model(X, y, patient, ["a", "b", "c"], ["imputation", "segmentation", "context"],
                                 Cs=(0.01, 1.0), cv_folds=4)
    assert evaluate(model.predict(X), y)["auroc_incorrect"] > 0.85
