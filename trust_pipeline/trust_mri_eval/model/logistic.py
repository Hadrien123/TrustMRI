"""Local confidence model: logistic regression predicting P(segmentation correct | patch features).

- Patients are the unit of splitting: train / calibration / test sets are disjoint sets of
  patients, and C is chosen by GroupKFold cross-validation grouped by patient.
- class_weight="balanced" handles the rarity of incorrect patches; it biases probabilities, which
  is why an optional Platt or isotonic recalibration is fitted on the calibration patients.
- Contributions are coefficient x standardised feature value (additive on the logit scale).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------- splitting
def split_patients(patient_ids: list[str], n_train: int, n_cal: int, n_test: int,
                   seed: int = 0) -> dict[str, list[str]]:
    """Shuffle patients and split them into disjoint train / calibration / test lists.

    Patients beyond ``n_train + n_cal + n_test`` go to train. Raises if there are too few.
    """
    ids = list(dict.fromkeys(patient_ids))
    if n_train + n_cal + n_test > len(ids):
        raise ValueError(f"split {n_train}/{n_cal}/{n_test} needs more than {len(ids)} patients")
    order = np.random.default_rng(seed).permutation(len(ids))
    shuffled = [ids[i] for i in order]
    test = shuffled[:n_test]
    cal = shuffled[n_test:n_test + n_cal]
    train = shuffled[n_test + n_cal:]
    return {"train": sorted(train), "calibration": sorted(cal), "test": sorted(test)}


# ---------------------------------------------------------------------------- model
def _make_classifier(C: float, penalty: str, seed: int) -> Pipeline:
    if penalty == "l1":
        clf = LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear", class_weight="balanced",
                                 random_state=seed, max_iter=2000)
    elif penalty == "l2":
        clf = LogisticRegression(C=C, l1_ratio=0.0, class_weight="balanced", random_state=seed, max_iter=2000)
    else:
        raise ValueError(f"unknown penalty {penalty!r}")
    return make_pipeline(StandardScaler(), clf)


class Recalibrator:
    """Monotone map from raw model probabilities to calibrated ones (Platt or isotonic)."""

    def __init__(self, method: str = "platt"):
        if method not in ("platt", "isotonic"):
            raise ValueError(f"unknown recalibration {method!r}")
        self.method = method
        self._model: LogisticRegression | IsotonicRegression | None = None

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "Recalibrator":
        if self.method == "platt":
            self._model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_raw)[:, None], y)
        else:
            self._model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p_raw, y)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        if isinstance(self._model, LogisticRegression):
            return self._model.predict_proba(_logit(p_raw)[:, None])[:, 1]
        return self._model.predict(p_raw)  # type: ignore[union-attr]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class ConfidenceModel:
    """Fitted confidence model on a subset of features."""

    pipeline: Pipeline
    feature_names: list[str]
    feature_groups: list[str]
    columns: np.ndarray                       # indices of the used features in the full matrix
    C: float
    cv_scores: dict[float, float] = field(default_factory=dict)
    recalibrator: Recalibrator | None = None

    def _x(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.columns]

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Uncalibrated P(correct) from the balanced logistic regression."""
        return self.pipeline.predict_proba(self._x(X))[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Confidence = (recalibrated) probability that the segmentation is correct, in [0, 1]."""
        p = self.predict_raw(X)
        return self.recalibrator.transform(p) if self.recalibrator is not None else p

    @property
    def coefficients(self) -> dict[str, float]:
        """Logistic coefficients on standardised features."""
        coef = self.pipeline[-1].coef_[0]
        return dict(zip(self.feature_names, coef.tolist()))

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Per-feature contribution to the logit: coefficient x standardised value, (n, F_used)."""
        z = self.pipeline[0].transform(self._x(X))
        return z * self.pipeline[-1].coef_[0]

    def group_contributions(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Contributions summed by feature group."""
        contrib = self.contributions(X)
        groups = np.array(self.feature_groups)
        return {g: contrib[:, groups == g].sum(1) for g in dict.fromkeys(self.feature_groups)}


def select_columns(names: list[str], sources: list[str], include: list[str] | None = None,
                   exclude_sources: tuple[str, ...] = ()) -> np.ndarray:
    """Indices of features to use: an explicit name list, or all except some sources."""
    if include is not None:
        missing = set(include) - set(names)
        if missing:
            raise KeyError(f"unknown features {sorted(missing)}")
        return np.array([names.index(n) for n in include])
    return np.array([i for i, s in enumerate(sources) if s not in exclude_sources])


def fit_confidence_model(X: np.ndarray, y: np.ndarray, patient: np.ndarray, names: list[str],
                         groups: list[str], columns: np.ndarray | None = None, penalty: str = "l2",
                         Cs: tuple[float, ...] = (0.01, 0.1, 1.0), cv_folds: int = 5,
                         scoring: str = "roc_auc", seed: int = 0) -> ConfidenceModel:
    """Fit the logistic model, choosing C by GroupKFold CV over patients (never over patches).

    Args:
        X: (n_patches, F) features of the training patients; y: labels (1 = correct).
        patient: (n_patches,) patient id of each row, used as CV group.
        columns: feature indices to use (default: all).
        scoring: "roc_auc" (higher is better) or "neg_log_loss".
    """
    columns = np.arange(X.shape[1]) if columns is None else np.asarray(columns)
    if len(np.unique(y)) < 2:
        raise ValueError("training labels contain a single class; cannot fit a confidence model")
    Xc = X[:, columns]
    n_groups = len(np.unique(patient))
    scores: dict[float, float] = {}
    if n_groups >= 2 and len(Cs) > 1:
        cv = GroupKFold(n_splits=min(cv_folds, n_groups))
        for C in Cs:
            fold_scores = []
            for tr, va in cv.split(Xc, y, groups=patient):
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                    continue
                p = _make_classifier(C, penalty, seed).fit(Xc[tr], y[tr]).predict_proba(Xc[va])[:, 1]
                fold_scores.append(roc_auc_score(y[va], p) if scoring == "roc_auc" else -log_loss(y[va], p))
            scores[C] = float(np.mean(fold_scores)) if fold_scores else float("nan")
    finite = {c: s for c, s in scores.items() if np.isfinite(s)}
    best_C = max(finite, key=finite.get) if finite else (Cs[len(Cs) // 2] if Cs else 1.0)
    pipe = _make_classifier(best_C, penalty, seed).fit(Xc, y)
    return ConfidenceModel(pipe, [names[i] for i in columns], [groups[i] for i in columns],
                           columns, best_C, scores)


# ---------------------------------------------------------------------------- evaluation
def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict[str, np.ndarray | float]:
    """Reliability diagram (equal-width bins of predicted P(correct)) and expected calibration error."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    conf = np.bincount(idx, weights=p, minlength=n_bins) / np.maximum(count, 1)
    acc = np.bincount(idx, weights=y, minlength=n_bins) / np.maximum(count, 1)
    ece = float(np.sum(count / max(len(p), 1) * np.abs(acc - conf)))
    return {"edges": edges, "confidence": conf, "accuracy": acc, "count": count, "ece": ece}


def risk_coverage(confidence: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float]:
    """Selective risk (error rate of kept patches) vs coverage, keeping the most confident first.

    AURC is the area under the risk-coverage curve (lower is better)."""
    order = np.argsort(-np.asarray(confidence), kind="stable")
    errors = 1 - np.asarray(y, float)[order]
    n = np.arange(1, len(order) + 1)
    coverage, risk = n / len(order), np.cumsum(errors) / n
    return {"coverage": coverage, "risk": risk, "aurc": float(np.mean(risk))}


def qubrats_curve(confidence: np.ndarray, confusion: np.ndarray, patient: np.ndarray,
                  rest: dict[str, np.ndarray] | None = None,
                  thresholds: np.ndarray | None = None) -> dict[str, np.ndarray | float]:
    """QU-BraTS-style filtering: at each threshold, patches with confidence < threshold are
    filtered out; Dice is computed on the kept voxels (per patient, then averaged).

    Args:
        confidence: (n,) per-patch confidence; confusion: (n, 4) TP, FP, FN, TN voxel counts.
        patient: (n,) patient id per patch.
        rest: optional {patient: (4,) counts} of voxels outside the scored patches (always kept).

    Returns thresholds, mean Dice of kept voxels, mean fraction of filtered voxels, mean ratio of
    filtered TP and TN (as in QU-BraTS, filtering correct voxels is penalised) and their AUCs.
    """
    thresholds = np.linspace(0, 1, 21) if thresholds is None else thresholds
    per_patient = {"dice": [], "filtered": [], "ftp": [], "ftn": []}
    for pid in np.unique(patient):
        sel = patient == pid
        conf, cm = confidence[sel], confusion[sel].astype(float)
        extra = np.zeros(4) if rest is None else np.asarray(rest.get(pid, np.zeros(4)), float)
        total = cm.sum(0) + extra
        curves = {k: [] for k in per_patient}
        for t in thresholds:
            kept = cm[conf >= t].sum(0) + extra
            tp, fp, fn, _ = kept
            curves["dice"].append(1.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
            curves["filtered"].append(1 - kept.sum() / max(total.sum(), 1))
            curves["ftp"].append(1 - kept[0] / total[0] if total[0] else 0.0)
            curves["ftn"].append(1 - kept[3] / total[3] if total[3] else 0.0)
        for k in per_patient:
            per_patient[k].append(curves[k])
    out: dict[str, np.ndarray | float] = {"thresholds": thresholds}
    for k, v in per_patient.items():
        out[k] = np.mean(np.array(v), axis=0)
    span = thresholds[-1] - thresholds[0] or 1.0
    for k in ("dice", "ftp", "ftn"):
        y_vals = np.asarray(out[k])
        out[f"auc_{k}"] = float(np.sum((y_vals[1:] + y_vals[:-1]) / 2 * np.diff(thresholds)) / span)
    out["score"] = float((out["auc_dice"] + (1 - out["auc_ftp"]) + (1 - out["auc_ftn"])) / 3)
    return out


def evaluate(confidence: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict[str, float | dict]:
    """AUROC for detecting incorrect patches, ECE, Brier score, AURC (+ curves for plotting)."""
    y = np.asarray(y)
    out: dict[str, float | dict] = {"n_patches": float(len(y)), "frac_incorrect": float(1 - y.mean())}
    if len(np.unique(y)) == 2:
        out["auroc_incorrect"] = float(roc_auc_score(1 - y, 1 - confidence))
        fpr, tpr, _ = roc_curve(1 - y, 1 - confidence)
        out["roc"] = {"fpr": fpr, "tpr": tpr}
    else:
        out["auroc_incorrect"] = float("nan")
    rel = reliability(confidence, y, n_bins)
    out["ece"] = rel["ece"]
    out["reliability"] = rel
    out["brier"] = float(np.mean((confidence - y) ** 2))
    rc = risk_coverage(confidence, y)
    out["aurc"] = rc["aurc"]
    out["risk_coverage"] = rc
    return out
