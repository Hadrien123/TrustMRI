import json

import dataclasses
import pandas as pd

from trust_mri_eval.cli import build_config
from trust_mri_eval.pipeline import run


def test_end_to_end_synthetic(small_cfg, tmp_path):
    cfg = dataclasses.replace(small_cfg, output_dir=tmp_path, run_id="t", save_maps=True, save_figures=True,
                              coarse_patch_size=(8, 8, 8))
    out = run(cfg)
    df = pd.read_csv(out / "metrics_per_patient.csv")
    assert len(df) == cfg.n_patients and set(df["split"]) == {"train", "calibration", "test"}
    for col in ("img_ssim_brain_mean", "seg_dice", "seg_lesion_f1", "anova_share_prompt",
                "local_p3_failure_fraction", "conf_mean"):
        assert df[col].notna().all(), col
    summary = json.loads((out / "metrics_summary.json").read_text())
    assert set(summary["regions"]["ET"]["model"]) == {"full", "agreement only", "no imputation"}
    maps = out / "maps" / "ET" / "syn000"
    for name in ("confidence", "failure_p3", "anova_frac_prompt", "agreement", "dominant_source"):
        assert (maps / f"{name}.nii.gz").exists(), name
    figs = out / "figures" / "ET"
    for name in ("syn000.png", "reliability.png", "roc.png", "risk_coverage.png", "coefficients.png"):
        assert (figs / name).exists(), name
    assert (out / "report.md").read_text(encoding="utf-8").startswith("# TRUST-MRI")


def test_cli_overrides():
    cfg = build_config(["--synthetic", "--shape", "32", "32", "24", "--set", "Cs=[0.1,1]", "penalty=l1", "n_patients=5"])
    assert cfg.volume_shape == (32, 32, 24) and cfg.Cs == (0.1, 1) and cfg.penalty == "l1" and cfg.n_patients == 5
