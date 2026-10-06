"""Short Markdown report summarising a run (cohort mean +- std of every metric, model results)."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

_SECTIONS = [
    ("A. Imputed vs real image (brain mask, intensities rescaled to [0, 1])", "img_", [
        ("img_l1_mean", "L1 (MAE)"), ("img_l2_mean", "L2 (MSE)"), ("img_psnr_mean", "PSNR (dB)"),
        ("img_ssim_brain_mean", "SSIM whole brain"), ("img_ssim_tumour_mean", "SSIM tumour"),
        ("img_ssim_healthy_mean", "SSIM healthy brain")]),
    ("C. Distribution of the N imputed samples", "dist_", [
        ("dist_mean_variance_brain", "mean variance, brain"), ("dist_mean_variance_tumour", "mean variance, tumour"),
        ("dist_pairwise_ssim_mean", "pairwise SSIM"), ("dist_pairwise_psnr_mean", "pairwise PSNR (dB)"),
        ("dist_coverage", "coverage of [5, 95] % interval"),
        ("dist_spread_error_spearman", "Spearman(local std, local error)")]),
    ("D. Segmentation (consensus of N x K masks vs reference)", "seg_", [
        ("seg_dice", "Dice"), ("seg_nsd", "NSD"), ("seg_hd95", "HD95 (mm)"),
        ("seg_lesion_dice", "lesion-wise Dice (BraTS 2023)"), ("seg_lesion_hd95", "lesion-wise HD95 (mm)"),
        ("seg_lesion_precision", "lesion precision"), ("seg_lesion_recall", "lesion recall"),
        ("seg_lesion_f1", "lesion F1"), ("seg_dice_per_mask_mean", "Dice of individual masks"),
        ("seg_dice_per_mask_std", "std of Dice across masks")]),
    ("Uncertainty and ANOVA (inside the ROI)", "anova_", [
        ("unc_mean_entropy_roi", "mean entropy (bits)"),
        ("unc_disagreement_fraction_roi", "fraction of ROI voxels where masks disagree"),
        ("anova_share_imputation", "variance share: imputation"), ("anova_share_prompt", "variance share: prompt"),
        ("anova_share_interaction", "variance share: interaction"),
        ("anova_volume_frac_imputation", "lesion volume: imputation fraction"),
        ("anova_volume_frac_prompt", "lesion volume: prompt fraction"),
        ("anova_volume_frac_interaction", "lesion volume: interaction fraction"),
        ("anova_n_lesions_frac_imputation", "lesion count: imputation fraction"),
        ("anova_n_lesions_frac_prompt", "lesion count: prompt fraction")]),
]


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _metric_table(cohort: dict, items: list[tuple[str, str]]) -> list[str]:
    lines = ["| metric | mean ± std over patients | n |", "|---|---|---|"]
    for key, label in items:
        if key in cohort and cohort[key]["n"]:
            c = cohort[key]
            lines.append(f"| {label} | {_fmt(c['mean'])} ± {_fmt(c['std'])} | {c['n']} |")
    return lines if len(lines) > 2 else []


def _local_table(cohort: dict) -> list[str]:
    keys = sorted(k for k in cohort if k.startswith("local_") and ("_mean" in k or "frac" in k or "fraction" in k))
    if not keys:
        return []
    lines = ["| local summary | mean ± std over patients |", "|---|---|"]
    for key in keys:
        if key.endswith(("_brain_mean", "_tumour_mean", "_healthy_mean", "_thr", "failure_fraction")):
            lines.append(f"| `{key[len('local_'):]}` | {_fmt(cohort[key]['mean'], 4)} ± {_fmt(cohort[key]['std'], 4)} |")
    return lines


def write_report(path: str | Path, summary: dict, df: pd.DataFrame) -> None:
    """Write ``report.md`` from the run summary and the per-patient table."""
    lines = [f"# TRUST-MRI evaluation report — run `{summary['run_id']}`", ""]
    cfg = summary["config"]
    source = "synthetic data" if cfg["synthetic"] else f"real data from `{cfg['data_dir']}`"
    lines += [f"{len(df['patient_id'].unique())} patients ({source}), N = {cfg['n_imputations']} imputations, "
              f"K = {cfg['n_prompts']} prompts, patch size {tuple(cfg['patch_size'])}. "
              f"Split: train {summary['split']['train']}, calibration {summary['split']['calibration']}, "
              f"test {summary['split']['test']}.", ""]
    if cfg["synthetic"]:
        lines += ["> Synthetic data: these numbers check that the pipeline works end to end; they say nothing "
                  "about a real imputation or segmentation model.", ""]

    for region, rs in summary["regions"].items():
        cohort = rs["cohort"]
        lines += [f"## Region {region}", ""]
        for title, _, items in _SECTIONS:
            table = _metric_table(cohort, items)
            if table:
                lines += [f"### {title}", ""] + table + [""]
        local = _local_table(cohort)
        if local:
            lines += ["### B. Local patch maps",
                      "", "Thresholds: percentile "
                      f"{cfg['failure_percentile']:g} of training-patient brain patches (SSIM: lower tail). "
                      "`frac_above_thr` / `frac_below_thr` = fraction of patches beyond the threshold.", ""] + local + [""]
        if "model" in rs:
            lines += ["### Local confidence model (test patients)", "",
                      "| model | features | C | AUROC (incorrect) | ECE | Brier | AURC | QU-BraTS score |",
                      "|---|---|---|---|---|---|---|---|"]
            for name, m in rs["model"].items():
                lines.append(f"| {name} | {m['n_features']} | {m['C']:g} | {_fmt(m.get('auroc_incorrect'))} | "
                             f"{_fmt(m.get('ece'))} | {_fmt(m.get('brier'))} | {_fmt(m.get('aurc'))} | "
                             f"{_fmt(m.get('qubrats_score'))} |")
            any_m = next(iter(rs["model"].values()))
            lines += ["", f"{_fmt(any_m.get('frac_incorrect'))} of the {int(any_m.get('n_patches') or 0)} test "
                      "patches are incorrect. AUROC: detection of incorrect patches from 1 - confidence. "
                      "AURC: area under the risk-coverage curve (lower is better). QU-BraTS score: mean of "
                      "AUC(Dice of kept voxels), 1 - AUC(filtered TP), 1 - AUC(filtered TN).", ""]
            coefs = sorted(rs.get("coefficients", {}).items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
            if coefs:
                lines += ["Largest standardised coefficients of the full model (> 0: more likely correct): "
                          + ", ".join(f"`{k}` {v:+.2f}" for k, v in coefs) + ".", ""]
        lines += [f"Figures: `figures/{region}/` (one panel per patient, reliability, ROC, risk-coverage, "
                  f"QU-BraTS filtering, coefficients). Maps: `maps/{region}/<patient>/*.nii.gz`.", ""]

    lines += ["## Files", "",
              "- `metrics_per_patient.csv`: one row per patient and region, every scalar metric.",
              "- `metrics_summary.json`: configuration, split, thresholds, cohort mean/std, model results.",
              "- `maps/`: voxel and patch maps as NIfTI, in the original volume geometry.",
              "- `figures/`: PNG figures.", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
