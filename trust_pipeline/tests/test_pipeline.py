import dataclasses

import pandas as pd

from trust_mri_eval.pipeline import run


def test_end_to_end_synthetic(small_cfg, tmp_path):
    out = run(dataclasses.replace(small_cfg, output_dir=tmp_path, run_id="t", save_maps=True, save_figures=True))
    df = pd.read_csv(out / "metrics_per_patient.csv")
    assert len(df) == small_cfg.n_patients
    assert df[["img_ssim_brain_mean", "seg_dice", "seg_lesion_f1", "anova_share_prompt", "conf_mean"]].notna().all().all()
    for f in ("report.md", "metrics_summary.json", "maps/ET/syn000/confidence.nii.gz", "figures/ET/syn000.png"):
        assert (out / f).exists(), f
