"""End-to-end orchestration.

Per patient (one patient in memory at a time):
    preprocessing
    quality(): one output (imputed image n = eval_imputation, segmentation (n, eval_prompt))
        image vs real image: L1, L2, SSIM, PSNR, and local patch maps (SSIM, L1, L2)
        segmentation vs reference: official BraTS DSC, NSD, lesion-wise F1
    trust(): all N x K outputs
        distribution of the N images (mean variance, pairwise SSIM and PSNR), agreement of the
        N x K segmentations, ANOVA, patch features (variance, pairwise SSIM, pairwise Dice)
    patch labels (is the evaluated segmentation correct on the patch?), if there is a ground truth
Across patients:
    logistic confidence model (train patients) -> AUROC (test patients) -> confidence heatmaps,
    CSV, JSON summary and one figure per patient.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from brats_evaluation import config_path, evaluate_single_exam   # official BraTS evaluation
from panoptica import Panoptica_Evaluator
from skimage.metrics import peak_signal_noise_ratio as psnr       # what BraSyn's test.py imports
from util.ssim import ssim                                        # BraSyn's 3D SSIM

from .config import Config
from .data.io import index_cases, load_case, save_nifti
from .data.types import PatientData
from .metrics.image_local import local_maps, ssim_map
from .model.features import PatchFeatures, compute_features
from .model.labels import patch_labels
from .model.logistic import auroc_incorrect, fit_confidence_model, split_patients
from .patches import PatchGrid
from .preprocessing import Prepared, prepare
from .uncertainty.anova import COMPONENTS, anova_maps
from .viz import plots

log = logging.getLogger("trust_mri_eval")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVALUATOR = Panoptica_Evaluator.load_from_config(str(config_path("gli")))   # BraTS glioma config


def t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, np.float32))[None, None].to(DEVICE)   # shape BraSyn's ssim expects


@dataclass
class MapWriter:
    """Writes maps as NIfTI in BraSyn space (does nothing if ``directory`` is None)."""

    directory: Path | None
    affine: np.ndarray

    def __call__(self, name: str, volume: np.ndarray) -> None:
        if self.directory is not None:
            save_nifti(volume.astype(np.float32), self.directory / f"{name}.nii.gz", self.affine)


@dataclass
class PatientResult:
    """What is kept of a patient (small arrays only)."""

    row: dict
    features: PatchFeatures
    labels: np.ndarray | None
    slices: dict[str, np.ndarray]
    z: int
    write: MapWriter


# ---------------------------------------------------------------------------- one patient
def quality(prep: Prepared, cfg: Config, grid: PatchGrid, write: MapWriter) -> tuple[dict, dict]:
    """Evaluated output vs ground truth; returns (row, local maps painted on voxels). Image: L1, L2 in the brain,
    SSIM and PSNR as in BraSyn's test.py. Segmentation: official BraTS evaluation; binary masks are written as
    label 3, which every gli group contains, so the "et" group scores them as they are."""
    gt, image = prep.gt_image, prep.imputed[cfg.eval_imputation]
    seg = prep.masks[cfg.eval_imputation, cfg.eval_prompt]
    diff = (image - gt)[prep.brain]
    brats = evaluate_single_exam(seg.astype(np.uint8) * 3, prep.gt_mask.astype(np.uint8) * 3, prep.patient_id,
                                 EVALUATOR)["et"]
    row = {"img_l1": float(np.abs(diff).mean()), "img_l2": float((diff ** 2).mean()),
           "img_ssim": ssim(t(gt), t(image)).item(), "img_psnr": psnr(gt, image, data_range=image.max() - image.min()),
           "seg_dice": brats["global_bin_dsc"], "seg_nsd": brats["global_bin_nsd"], "seg_lesion_f1": brats["rq"]}
    maps = {}
    for name, values in local_maps(prep.gt_image, image, prep.brain, grid).items():
        maps[f"local_{name}"] = grid.expand(values, np.nan)
        write(f"local_{name}", maps[f"local_{name}"])
    return row, maps


def trust(prep: Prepared, cfg: Config, grid: PatchGrid, write: MapWriter) -> tuple[dict, dict, PatchFeatures]:
    """Distribution of the N images, ANOVA and patch features; returns (row, maps, features)."""
    pairs = list(combinations(prep.imputed, 2))
    variance = prep.imputed.var(0, ddof=1) if pairs else np.zeros(prep.shape, np.float32)
    ssim_values = [ssim(t(a), t(b)).item() for a, b in pairs]          # BraSyn, as imputation_eval.ipynb
    psnr_values = [psnr(a, b, data_range=b.max() - b.min()) for a, b in pairs]
    pair_ssim = sum(ssim_map(a, b) for a, b in pairs) / len(pairs) if pairs else np.ones(prep.shape, np.float32)
    row = {"dist_mean_variance": float(variance[prep.brain].mean()),
           "dist_pairwise_ssim": float(np.mean(ssim_values)) if ssim_values else np.nan,
           "dist_pairwise_psnr": float(np.mean(psnr_values)) if psnr_values else np.nan}
    maps = {"variance": grid.expand(grid.mean(variance, prep.brain), np.nan),
            "agreement": prep.masks.mean((0, 1), dtype=np.float32)}   # fraction of the N x K masks saying tumour
    write("local_variance", maps["variance"])
    write("seg_agreement", maps["agreement"])

    anova = anova_maps(prep.probs, prep.roi, cfg.anova_chunk)
    total = float(anova["var_total"][prep.roi].sum())
    for comp in COMPONENTS:
        row[f"anova_share_{comp}"] = float(anova[f"var_{comp}"][prep.roi].sum() / total) if total > 0 else np.nan
        maps[f"frac_{comp}"] = anova[f"frac_{comp}"]
        write(f"anova_var_{comp}", anova[f"var_{comp}"])
        write(f"anova_frac_{comp}", anova[f"frac_{comp}"])

    feats = compute_features(variance, pair_ssim, prep.masks, prep.roi, cfg.patch_size, cfg.min_patch_coverage)
    return row, maps, feats


def process_patient(data: PatientData, cfg: Config, maps_dir: Path | None) -> PatientResult:
    """Preprocessing, quality (if there is a ground truth) and trust for one patient."""
    prep = prepare(data, cfg.seg_threshold, cfg.roi_dilation)
    write = MapWriter(maps_dir, prep.affine)
    grid = PatchGrid.over(prep.shape, cfg.patch_size)
    image = prep.imputed[cfg.eval_imputation]
    seg = prep.masks[cfg.eval_imputation, cfg.eval_prompt]
    gt = prep.has_ground_truth
    N, K = prep.probs.shape[:2]
    nan = np.full(prep.shape, np.nan, np.float32)

    row = {"patient_id": prep.patient_id, "region": prep.region, "n_imputations": N, "n_prompts": K}
    labels = None
    maps = {"local_ssim": nan, "local_l1": nan, "patch_label": nan}
    if gt:
        quality_row, local = quality(prep, cfg, grid, write)
        row.update(quality_row)
        maps.update(local)
    trust_row, trust_maps, feats = trust(prep, cfg, grid, write)
    row.update(trust_row)
    maps.update(trust_maps)
    if gt:
        labels = patch_labels(seg, prep.gt_mask, feats.grid, feats.valid, cfg.label_tolerance, cfg.label_min_fraction)
        maps["patch_label"] = feats.grid.expand(feats.to_grid(labels), np.nan)
        write("patch_label", maps["patch_label"])

    z = plots.largest_tumour_slice(prep.gt_mask if gt else seg)
    volumes = {"image": prep.gt_image if gt else image, "imputed": image, "seg": seg,
               "gt_mask": prep.gt_mask if gt else np.zeros_like(seg), **maps}
    log.info("  %s: %d patches", prep.patient_id, feats.X.shape[0])
    return PatientResult(row, feats, labels, {k: v[z] for k, v in volumes.items()}, z, write)


# ---------------------------------------------------------------------------- confidence model
def confidence_model(results: dict[str, PatientResult], split: dict[str, list[str]], cfg: Config) -> dict:
    """Fit on train patients, write every patient's confidence heatmap, return C and the test AUROC."""
    train = [results[p] for p in split["train"] if results[p].labels is not None]
    if not train or len(np.unique(np.concatenate([r.labels for r in train]))) < 2:
        log.warning("no labelled training patients with both classes: confidence model skipped")
        return {}
    X = np.concatenate([r.features.X for r in train])
    y = np.concatenate([r.labels for r in train])
    patient = np.concatenate([np.full(r.labels.size, r.row["patient_id"]) for r in train])
    model, C = fit_confidence_model(X, y, patient, cfg.penalty, tuple(cfg.Cs), cfg.cv_folds, cfg.seed)

    test_conf, test_y = [], []
    for pid, r in results.items():
        conf = model.predict_proba(r.features.X)[:, 1] if len(r.features.X) else np.zeros(0)
        volume = r.features.grid.expand(r.features.to_grid(conf), fill=np.nan)
        r.write("confidence", volume)
        r.slices["confidence"] = volume[r.z]
        if pid in split["test"] and r.labels is not None:
            test_conf.append(conf)
            test_y.append(r.labels)
    auroc = auroc_incorrect(np.concatenate(test_conf), np.concatenate(test_y)) if test_conf else np.nan
    return {"C": C, "test_auroc_incorrect": auroc}


# ---------------------------------------------------------------------------- driver
def run(cfg: Config) -> Path:
    """Run the full pipeline; returns the output directory."""
    cases = index_cases(cfg.data_dir)
    if not cases:
        raise FileNotFoundError(f"no imputed images <case>-<modality>-runNN.nii[.gz] under {cfg.data_dir}")
    out = Path(cfg.output_dir) / (cfg.run_id or datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    split = split_patients(sorted(cases), cfg.test_fraction, seed=cfg.seed)

    results = {}
    for pid in sorted(cases):
        log.info("%s", pid)
        data = load_case(pid, cases[pid], cfg.modality, cfg.region)
        results[pid] = process_patient(data, cfg, out / "maps" / pid if cfg.save_maps else None)
        results[pid].row["split"] = "test" if pid in split["test"] else "train"
    model = confidence_model(results, split, cfg)
    if cfg.save_figures:
        for pid, r in results.items():
            r.slices.setdefault("confidence", np.full(r.slices["image"].shape, np.nan, np.float32))
            plots.patient_panel(r.slices, out / "figures" / f"{pid}.png",
                                f"{pid} - {r.row['region']} - axial slice {r.z} (BraSyn space) - {r.row['split']}")

    df = pd.DataFrame([r.row for r in results.values()])
    summary = {"config": cfg.to_dict(), "split": split, "model": model, "cohort": _cohort(df)}
    df.to_csv(out / "metrics_per_patient.csv", index=False)
    (out / "metrics_summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    log.info("outputs in %s", out)
    return out


def _cohort(df: pd.DataFrame) -> dict:
    """Mean, std and count over patients of every numeric column (inf ignored)."""
    df = df.select_dtypes("number").replace([np.inf, -np.inf], np.nan)
    return {c: {"mean": df[c].mean(), "std": df[c].std(ddof=0), "n": int(df[c].notna().sum())} for c in df.columns}


def _jsonable(obj):
    """Recursively convert numpy types; NaN/inf -> None (strict JSON)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return str(obj) if isinstance(obj, Path) else obj
