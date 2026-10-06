"""End-to-end orchestration.

Phase 1, one patient at a time (only one patient's volumes are ever in memory):
    preprocessing -> A (global image) -> B (local maps) -> C (distribution) -> D (segmentation)
    -> confidence maps -> ANOVA -> features (ground-truth free) and labels.
Phase 2, across patients:
    training-set thresholds -> failure maps and local summaries -> logistic confidence model
    (full / agreement only / without imputation features) -> confidence and contribution maps
    -> figures, CSV, JSON and Markdown report.

The patient split (train / calibration / test) is fixed before phase 1, so that every
threshold and model parameter is learned on training patients only.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data.io import list_patients, load_patient, save_nifti
from .data.types import PatientData
from .metrics.distribution import interval_coverage, mean_variance_maps, pairwise_psnr, pairwise_ssim, \
    spread_error_correlation
from .metrics.image_global import image_global_metrics, mean_std
from .metrics.image_local import LOW_IS_BAD, failure_map, flag_threshold, patch_maps, patch_regions, \
    summarize_patch_map, voxel_error_maps
from .metrics.lesions import label_lesions
from .metrics.segmentation import dice, segmentation_metrics
from .metrics.ssim import local_stats, masked_mean, ssim_map
from .model.features import GROUPS, PatchFeatures, compute_features
from .model.labels import patch_confusion, patch_labels
from .model.logistic import ConfidenceModel, Recalibrator, evaluate, fit_confidence_model, qubrats_curve, \
    select_columns, split_patients
from .patches import PatchGrid
from .preprocessing import Prepared, prepare
from .report import write_report
from .uncertainty.anova import COMPONENTS, anova_maps, anova_scalar
from .uncertainty.confidence_maps import agreement_map, entropy_map, signed_distance
from .viz import plots

log = logging.getLogger("trust_mri_eval")

#: local patch maps summarised and thresholded (module B + local pairwise SSIM from C)
LOCAL_MAPS = ("ssim", "l1", "l2", "variance", "pairwise_ssim")
MODEL_VARIANTS = ("full", "agreement only", "no imputation")


# ---------------------------------------------------------------------------- containers
@dataclass
class MapWriter:
    """Writes cropped maps back into the full volume as NIfTI under ``directory``."""

    directory: Path | None
    crop: tuple[slice, slice, slice]
    full_shape: tuple[int, int, int]
    affine: np.ndarray

    def __call__(self, name: str, volume: np.ndarray, fill: float = 0.0) -> None:
        if self.directory is None:
            return
        out = np.full(self.full_shape, fill, dtype=np.uint8 if volume.dtype == bool else np.float32)
        out[self.crop] = volume
        save_nifti(out, self.directory / f"{name}.nii.gz", self.affine)


@dataclass
class LocalMaps:
    grid: PatchGrid
    regions: dict[str, np.ndarray]
    maps: dict[str, np.ndarray]


@dataclass
class PatientResult:
    """What is kept of a patient after phase 1 (small: patch-level arrays and 2D slices)."""

    patient_id: str
    row: dict[str, float | str]
    local: dict[str, LocalMaps]
    features: PatchFeatures
    labels: np.ndarray | None
    confusion: np.ndarray | None
    rest: np.ndarray | None
    slices: dict[str, np.ndarray]
    z: int
    writer: MapWriter


# ---------------------------------------------------------------------------- data sources
def data_source(cfg: Config) -> tuple[list[str], Callable[[str, str], PatientData]]:
    """Patient ids and a ``load(patient_id, region)`` function (synthetic or NIfTI)."""
    if cfg.synthetic:
        from .data.synthetic import generate_patient, patient_ids
        ids = patient_ids(cfg)

        def load(pid: str, region: str) -> PatientData:
            data = generate_patient(ids.index(pid), cfg)
            data.region = region
            return data
        return ids, load
    if cfg.data_dir is None:
        raise ValueError("real data requires data_dir")
    paths = {p.name: p for p in list_patients(cfg.data_dir)}
    if not paths:
        raise FileNotFoundError(f"no patient folders found in {cfg.data_dir}")
    return sorted(paths), lambda pid, region: load_patient(paths[pid], region)


def local_tags(cfg: Config) -> dict[str, tuple[int, int, int]]:
    """Patch sizes of module B, keyed by a short tag (e.g. ``p3``, ``p8``)."""
    sizes = [tuple(s) for s in (cfg.patch_size, cfg.coarse_patch_size) if s]
    return {(f"p{s[0]}" if len(set(s)) == 1 else "p" + "x".join(map(str, s))): s for s in sizes}


# ---------------------------------------------------------------------------- phase 1
def process_patient(prep: Prepared, cfg: Config, maps_dir: Path | None) -> PatientResult:
    """Every per-patient computation; returns only small arrays."""
    t0 = time.time()
    writer = MapWriter(maps_dir, prep.crop, prep.full_shape, prep.affine)
    brain, has_gt = prep.brain, prep.has_ground_truth
    tumour = prep.tumour if has_gt else prep.consensus & brain
    n, k = prep.probs.shape[:2]
    row: dict[str, float | str] = {
        "patient_id": prep.patient_id, "region": prep.region, "N": n, "K": k,
        "brain_voxels": int(brain.sum()), "roi_voxels": int(prep.roi.sum()),
        "consensus_voxels": int(prep.consensus.sum()),
        "reference_voxels": int(prep.gt_mask.sum()) if has_gt else np.nan,
    }

    # ---- C. distribution of the samples (no ground truth) -------------------------------
    mean_img, var_img = mean_variance_maps(prep.imputed)
    row["dist_mean_variance_brain"] = masked_mean(var_img, brain)
    row["dist_mean_variance_tumour"] = masked_mean(var_img, tumour)
    pair_values, pair_map = pairwise_ssim(prep.imputed, brain, 1.0, cfg.ssim_win_size)
    row.update(mean_std({"dist_pairwise_ssim": pair_values}))
    row.update(mean_std({"dist_pairwise_psnr": pairwise_psnr(prep.imputed, brain)}))
    voxel_maps = {"variance": var_img, "pairwise_ssim": pair_map}
    writer("imputed_mean", mean_img)
    writer("imputed_variance", var_img)
    writer("pairwise_ssim_voxel", pair_map)

    # ---- A. ground truth vs samples ------------------------------------------------------
    if has_gt:
        gt_stats = local_stats(prep.gt_image, cfg.ssim_win_size)
        ssim_mean = np.zeros(prep.shape, np.float32)
        ssim_maps = []
        for s in prep.imputed:
            smap = ssim_map(gt_stats, local_stats(s, cfg.ssim_win_size))
            ssim_maps.append(smap)
            ssim_mean += smap / n
        row.update(mean_std(image_global_metrics(prep.gt_image, prep.imputed, brain, prep.tumour, ssim_maps),
                            prefix="img_"))
        del ssim_maps, gt_stats
        voxel_maps["ssim"] = ssim_mean
        voxel_maps.update(voxel_error_maps(prep.gt_image, prep.imputed))
        row["dist_coverage"] = interval_coverage(prep.imputed, prep.gt_image, brain, cfg.coverage_interval)

    # ---- B. local patch maps (thresholds come in phase 2) ---------------------------------
    std_img = np.sqrt(var_img)
    local: dict[str, LocalMaps] = {}
    for tag, size in local_tags(cfg).items():
        grid = PatchGrid.over(prep.shape, size)
        pm = patch_maps(voxel_maps, grid, brain)
        pm["std"] = grid.mean(std_img, brain)
        local[tag] = LocalMaps(grid, patch_regions(grid, brain, prep.tumour, cfg.min_patch_coverage), pm)
        for name in LOCAL_MAPS:
            if name in pm:
                writer(f"local_{name}_{tag}", grid.expand(pm[name]))
    main = next(iter(local.values()))
    if has_gt:
        sel = main.regions["brain"]
        row["dist_spread_error_spearman"] = spread_error_correlation(main.maps["std"][sel], main.maps["l1"][sel])

    # ---- D. segmentation ----------------------------------------------------------------
    if has_gt:
        seg = segmentation_metrics(prep.consensus, prep.gt_mask, prep.spacing, cfg.nsd_tolerance_mm,
                                   cfg.lesion_gt_dilation, cfg.lesion_min_size, cfg.lesion_match_rule,
                                   cfg.lesion_match_iou, cfg.compute_hd95)
        row.update({f"seg_{key}": v for key, v in seg.items()})
        row.update(mean_std({"seg_dice_per_mask": [dice(m, prep.gt_mask) for m in prep.masks.reshape(-1, *prep.shape)]}))

    # ---- confidence maps ------------------------------------------------------------------
    agreement = agreement_map(prep.masks)
    entropy = entropy_map(prep.mean_prob)
    sdist = signed_distance(prep.consensus, prep.spacing)
    roi = prep.roi
    row["unc_mean_entropy_roi"] = masked_mean(entropy, roi)
    row["unc_disagreement_fraction_roi"] = float(((agreement > 0) & (agreement < 1))[roi].mean()) if roi.any() else np.nan
    writer("consensus", prep.consensus)
    writer("agreement", agreement)
    writer("entropy", entropy)
    writer("signed_distance", sdist)

    # ---- ANOVA (inside the ROI) ------------------------------------------------------------
    values = prep.masks.astype(np.float32) if cfg.anova_on == "masks" else prep.probs
    anova = anova_maps(values, roi, cfg.anova_chunk)
    total = float(anova["var_total"][roi].sum())
    for comp in COMPONENTS:
        # share of the total segmentation variance in the ROI (variance-weighted fraction)
        row[f"anova_share_{comp}"] = float(anova[f"var_{comp}"][roi].sum() / total) if total > 0 else np.nan
        writer(f"anova_var_{comp}", anova[f"var_{comp}"])
        writer(f"anova_frac_{comp}", anova[f"frac_{comp}"])
    writer("anova_var_total", anova["var_total"])
    voxel_mm3 = float(np.prod(prep.spacing))
    volumes = prep.masks.sum(axis=(2, 3, 4)) * voxel_mm3
    n_lesions = np.array([[label_lesions(prep.masks[i, j], cfg.lesion_min_size)[1] for j in range(k)]
                          for i in range(n)], dtype=float)
    for label, scalars in (("volume", volumes), ("n_lesions", n_lesions)):
        res = anova_scalar(scalars)
        row[f"anova_{label}_mean"] = float(scalars.mean())
        row[f"anova_{label}_var_total"] = res["var_total"]
        for comp in COMPONENTS:
            row[f"anova_{label}_frac_{comp}"] = res[f"frac_{comp}"]

    # ---- features (ground-truth free) and labels -----------------------------------------
    feats = compute_features(prep.imputed, prep.probs, brain, roi, cfg.patch_size, cfg.seg_threshold,
                             prep.spacing, cfg.context_sigmas, prep.flair, cfg.flair_abnormality_z,
                             cfg.min_patch_coverage, cfg.ssim_win_size, cfg.anova_on, cfg.anova_chunk)
    labels = confusion = rest = None
    if has_gt:
        labels = patch_labels(prep.consensus, prep.gt_mask, feats.grid, feats.valid, cfg.label_tolerance,
                              cfg.label_rule, cfg.label_min_fraction, cfg.label_min_correct)
        confusion = patch_confusion(prep.consensus, prep.gt_mask, feats.grid, feats.valid)
        p, g = prep.consensus, prep.gt_mask
        whole = np.array([(p & g).sum(), (p & ~g).sum(), (~p & g).sum(), (~p & ~g).sum()], dtype=np.int64)
        rest = whole - confusion.sum(0)
        row["model_patches"] = int(labels.size)
        row["model_frac_incorrect"] = float(1 - labels.mean()) if labels.size else np.nan

    # ---- slices for the figure -------------------------------------------------------------
    z = plots.largest_tumour_slice(prep.gt_mask if has_gt else prep.consensus)
    sl = (slice(None), slice(None), z)
    slices = {
        "brain": brain[sl], "imputed": prep.imputed[0][sl], "variance": var_img[sl],
        "consensus": prep.consensus[sl], "agreement": agreement[sl],
        "gt_image": (prep.gt_image if has_gt else mean_img)[sl],
        "gt_mask": (prep.gt_mask if has_gt else np.zeros_like(brain))[sl],
        "local_ssim": main.grid.expand(main.maps["ssim"], fill=np.nan)[sl] if "ssim" in main.maps
        else np.full(brain[sl].shape, np.nan, np.float32),
    }
    for comp in COMPONENTS:
        slices[f"frac_{comp}"] = anova[f"frac_{comp}"][sl]
    log.info("  %s processed in %.1fs (%d patches)", prep.patient_id, time.time() - t0, feats.X.shape[0])
    return PatientResult(prep.patient_id, row, local, feats, labels, confusion, rest, slices, z, writer)


# ---------------------------------------------------------------------------- phase 2
def local_thresholds(results: dict[str, PatientResult], train_ids: list[str],
                     percentile: float) -> dict[str, dict[str, float]]:
    """Per patch size and per map: percentile of training-patient brain-patch values."""
    out: dict[str, dict[str, float]] = {}
    tags = next(iter(results.values())).local.keys()
    for tag in tags:
        out[tag] = {}
        for name in LOCAL_MAPS:
            vals = [r.local[tag].maps[name][r.local[tag].regions["brain"]]
                    for pid, r in results.items() if pid in train_ids and name in r.local[tag].maps]
            if vals:
                out[tag][name] = flag_threshold(vals, percentile, name in LOW_IS_BAD)
    return out


def finalize_local(res: PatientResult, thresholds: dict[str, dict[str, float]], cfg: Config) -> None:
    """Local summaries (mean, tail percentile, fraction beyond threshold) and failure maps."""
    for tag, lm in res.local.items():
        for name in LOCAL_MAPS:
            if name in lm.maps and name in thresholds[tag]:
                res.row.update(summarize_patch_map(lm.maps[name], lm.regions, thresholds[tag][name],
                                                   name in LOW_IS_BAD, prefix=f"local_{tag}_{name}_"))
        err = cfg.failure_error
        if err in lm.maps and err in thresholds[tag]:
            fmap = failure_map(lm.maps[err], thresholds[tag][err])
            fmap = np.where(lm.regions["brain"], fmap, 0).astype(np.uint8)
            res.row[f"local_{tag}_failure_fraction"] = float(fmap[lm.regions["brain"]].mean())
            volume = lm.grid.expand(fmap.astype(np.float32)).astype(np.uint8)
            res.writer(f"failure_{tag}", volume)
            if "failure" not in res.slices:
                res.slices["failure"] = volume[:, :, res.z]
    res.slices.setdefault("failure", np.zeros_like(res.slices["brain"], dtype=np.uint8))


def _stack(results: dict[str, PatientResult], ids: list[str], names: list[str]):
    """Concatenate features (columns in ``names`` order), labels and patient ids."""
    X, y, pid = [], [], []
    for p in ids:
        r = results[p]
        if r.labels is None:
            continue
        idx = [r.features.names.index(nm) for nm in names]
        X.append(r.features.X[:, idx])
        y.append(r.labels)
        pid.append(np.full(r.labels.size, p, dtype=object))
    if not X:
        return np.zeros((0, len(names)), np.float32), np.zeros(0, np.uint8), np.zeros(0, object)
    return np.concatenate(X), np.concatenate(y), np.concatenate(pid)


def train_models(results: dict[str, PatientResult], split: dict[str, list[str]],
                 cfg: Config) -> tuple[dict[str, ConfidenceModel], list[str]]:
    """Fit the three model variants on training patients; recalibrate on calibration patients.

    Returns the models and the feature names (shared by all patients) they index into."""
    common = [nm for nm in next(iter(results.values())).features.names
              if all(nm in r.features.names for r in results.values())]
    ref = next(iter(results.values())).features
    groups = [ref.groups[ref.names.index(nm)] for nm in common]
    sources = [ref.sources[ref.names.index(nm)] for nm in common]
    X, y, pid = _stack(results, split["train"], common)
    Xc, yc, _ = _stack(results, split["calibration"], common)
    columns = {
        "full": select_columns(common, sources),
        "agreement only": select_columns(common, sources, include=["seg_agreement"]),
        "no imputation": select_columns(common, sources, exclude_sources=("imputation",)),
    }
    models = {}
    for name in MODEL_VARIANTS:
        m = fit_confidence_model(X, y, pid, common, groups, columns[name], cfg.penalty, tuple(cfg.Cs),
                                 cfg.cv_folds, cfg.cv_scoring, cfg.seed)
        if cfg.recalibration != "none" and yc.size and len(np.unique(yc)) == 2:
            m.recalibrator = Recalibrator(cfg.recalibration).fit(m.predict_raw(Xc), yc)
        models[name] = m
        log.info("  model %-15s C=%g, %d features", name, m.C, len(m.feature_names))
    return models, common


def apply_model(res: PatientResult, model: ConfidenceModel, names: list[str]) -> np.ndarray:
    """Confidence per valid patch; writes confidence, group-contribution and dominant-source maps."""
    feats = res.features
    X = feats.X[:, [feats.names.index(nm) for nm in names]]
    conf = model.predict(X)
    grid = feats.grid
    res.writer("confidence", grid.expand(feats.to_grid(conf), fill=np.nan), fill=np.nan)
    contrib = model.group_contributions(X)
    stack = np.zeros((len(GROUPS), conf.size), np.float32)
    for gi, g in enumerate(GROUPS):
        if g in contrib:
            stack[gi] = contrib[g]
            res.writer(f"contribution_{g}", grid.expand(feats.to_grid(contrib[g]), fill=0.0))
    dominant = np.argmax(np.abs(stack), axis=0) + 1
    dom_volume = grid.expand(feats.to_grid(dominant.astype(np.float32), fill=0), fill=0.0)
    res.writer("dominant_source", dom_volume.astype(np.uint8))
    res.slices["confidence"] = grid.expand(feats.to_grid(conf), fill=np.nan)[:, :, res.z]
    res.slices["dominant"] = dom_volume[:, :, res.z]
    res.row["conf_mean"] = float(conf.mean()) if conf.size else np.nan
    for gi, g in enumerate(GROUPS):
        res.row[f"conf_dominant_{g}"] = float((dominant == gi + 1).mean()) if conf.size else np.nan
    return conf


# ---------------------------------------------------------------------------- driver
def run(cfg: Config) -> Path:
    """Run the full pipeline; returns the output directory."""
    run_id = cfg.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(cfg.output_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    ids, load = data_source(cfg)
    split = split_patients(ids, *cfg.split, seed=cfg.seed)
    log.info("run %s: %d patients, split %s", run_id, len(ids), {k: len(v) for k, v in split.items()})

    all_rows, summary = [], {"run_id": run_id, "config": cfg.to_dict(), "split": split, "regions": {}}
    for region in cfg.regions:
        rows, region_summary = run_region(cfg, region, ids, load, split, out)
        all_rows.extend(rows)
        summary["regions"][region] = region_summary

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "metrics_per_patient.csv", index=False)
    with open(out / "metrics_summary.json", "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    write_report(out / "report.md", summary, df)
    log.info("outputs in %s", out)
    return out


def run_region(cfg: Config, region: str, ids: list[str], load: Callable[[str, str], PatientData],
               split: dict[str, list[str]], out: Path) -> tuple[list[dict], dict]:
    """Phases 1 and 2 for one tumour region."""
    maps_root = out / "maps" / region if cfg.save_maps else None
    fig_dir = out / "figures" / region
    results: dict[str, PatientResult] = {}
    for pid in ids:
        log.info("[%s] %s", region, pid)
        prep = prepare(load(pid, region), cfg.seg_threshold, tuple(cfg.rescale_percentiles),
                       cfg.roi_dilation, cfg.crop_margin)
        results[pid] = process_patient(prep, cfg, maps_root / pid if maps_root else None)
        del prep

    set_of = {pid: name for name, members in split.items() for pid in members}
    thresholds = local_thresholds(results, split["train"], cfg.failure_percentile)
    for r in results.values():
        r.row["split"] = set_of.get(r.patient_id, "unused")
        finalize_local(r, thresholds, cfg)

    summary: dict = {"local_thresholds": thresholds}
    trainable = [p for p in split["train"] if results[p].labels is not None]
    if trainable and len(np.unique(np.concatenate([results[p].labels for p in trainable]))) == 2:
        models, names = train_models(results, split, cfg)
        evaluations, curves = {}, {}
        for name, model in models.items():
            X, y, pid = _stack(results, split["test"], names)
            if not y.size:
                continue
            conf = model.predict(X)
            ev = evaluate(conf, y, cfg.n_bins)
            test = [results[p] for p in split["test"] if results[p].labels is not None]
            confusion = np.concatenate([r.confusion for r in test])
            rest = {r.patient_id: r.rest for r in test}
            curves[name] = qubrats_curve(conf, confusion, pid, rest)
            evaluations[name] = ev
        full = models["full"]
        for r in results.values():
            conf = apply_model(r, full, names)
            if r.labels is not None and r.patient_id in split["test"] and len(np.unique(r.labels)) == 2:
                r.row["conf_auroc_incorrect"] = evaluate(conf, r.labels, cfg.n_bins)["auroc_incorrect"]
        summary["model"] = {
            name: {
                "C": m.C, "cv_scores": {str(c): s for c, s in m.cv_scores.items()}, "n_features": len(m.feature_names),
                "recalibration": cfg.recalibration if m.recalibrator else "none",
                **{key: evaluations[name][key] for key in ("auroc_incorrect", "ece", "brier", "aurc",
                                                          "n_patches", "frac_incorrect")
                   if name in evaluations},
                **({f"qubrats_{key}": curves[name][key] for key in ("auc_dice", "auc_ftp", "auc_ftn", "score")}
                   if name in curves else {}),
            } for name, m in models.items()}
        summary["coefficients"] = full.coefficients
        summary["qubrats_curves"] = {name: {"thresholds": q["thresholds"], "dice": q["dice"],
                                            "filtered": q["filtered"]} for name, q in curves.items()}
        if cfg.save_figures and evaluations:
            plots.reliability_plot(evaluations, fig_dir / "reliability.png")
            plots.roc_plot(evaluations, fig_dir / "roc.png")
            plots.risk_coverage_plot(evaluations, fig_dir / "risk_coverage.png")
            plots.qubrats_plot(curves, fig_dir / "qubrats_filtering.png")
            plots.coefficient_plot(full.coefficients, fig_dir / "coefficients.png")
    else:
        log.warning("no labelled training patients with both classes: confidence model skipped")

    if cfg.save_figures:
        for r in results.values():
            r.slices.setdefault("confidence", np.full_like(r.slices["agreement"], np.nan))
            r.slices.setdefault("dominant", np.zeros_like(r.slices["agreement"]))
            plots.patient_panel(r.slices, fig_dir / f"{r.patient_id}.png",
                                f"{r.patient_id} - {region} - axial slice {r.z} (cropped) - split: {r.row['split']}")

    rows = [r.row for r in results.values()]
    summary["cohort"] = _cohort_summary(rows)
    return rows, summary


def _cohort_summary(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Mean, std and count over patients of every numeric column."""
    df = pd.DataFrame(rows)
    out = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            v = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            out[col] = {"mean": float(v.mean()) if len(v) else math.nan,
                        "std": float(v.std(ddof=0)) if len(v) else math.nan, "n": int(len(v))}
    return out


def _jsonable(obj):
    """Recursively convert numpy types; NaN/inf -> None (strict JSON)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj
