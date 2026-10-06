"""Confidence model: logistic regression giving, per patch, the probability that the segmentation is correct.

Patients are the unit of splitting (train / test), and C is chosen by GroupKFold cross-validation
grouped by patient. class_weight="balanced" because incorrect patches are rare.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def split_patients(patient_ids: list[str], test_fraction: float, seed: int = 0) -> dict[str, list[str]]:
    """Shuffle patients into disjoint train / test lists; at least one of each when there are two patients
    or more and ``test_fraction`` > 0 (a single patient goes to train)."""
    n = len(patient_ids)
    n_test = min(max(round(test_fraction * n), 1), n - 1) if n > 1 and test_fraction > 0 else 0
    shuffled = [patient_ids[i] for i in np.random.default_rng(seed).permutation(n)]
    return {"train": sorted(shuffled[n_test:]), "test": sorted(shuffled[:n_test])}


def _model(C: float, penalty: str, seed: int) -> Pipeline:
    solver = {"l1": dict(l1_ratio=1.0, solver="liblinear"), "l2": dict(l1_ratio=0.0)}[penalty]
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                                                              random_state=seed, **solver))


def auroc_incorrect(confidence: np.ndarray, y: np.ndarray) -> float:
    """AUROC for detecting incorrect patches from 1 - confidence (NaN if only one class)."""
    return float(roc_auc_score(1 - y, 1 - confidence)) if len(np.unique(y)) == 2 else float("nan")


def fit_confidence_model(X: np.ndarray, y: np.ndarray, patient: np.ndarray, penalty: str = "l2",
                         Cs: tuple[float, ...] = (0.01, 0.1, 1.0), cv_folds: int = 5,
                         seed: int = 0) -> tuple[Pipeline, float]:
    """Fit on all rows with the C that maximises the patient-grouped CV AUROC; returns (model, C)."""
    n_folds = min(cv_folds, len(np.unique(patient)))
    scores = {}
    for C in Cs if n_folds >= 2 else ():
        fold_scores = [auroc_incorrect(_model(C, penalty, seed).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1], y[va])
                       for tr, va in GroupKFold(n_folds).split(X, y, patient) if len(np.unique(y[tr])) == 2]
        scores[C] = np.nanmean(fold_scores) if fold_scores else np.nan
    finite = {c: s for c, s in scores.items() if np.isfinite(s)}
    best = max(finite, key=finite.get) if finite else Cs[len(Cs) // 2]
    return _model(best, penalty, seed).fit(X, y), best
