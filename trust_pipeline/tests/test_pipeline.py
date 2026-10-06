import dataclasses

import pandas as pd

from trust_mri_eval.pipeline import run


def test_end_to_end(small_cfg, tmp_path):
    out = run(dataclasses.replace(small_cfg, output_dir=tmp_path, run_id="t", save_maps=True, save_figures=True))
    df = pd.read_csv(out / "metrics_per_patient.csv")
    assert len(df) == 4
    assert df[["img_ssim", "seg_dice", "seg_lesion_f1", "dist_mean_variance", "anova_share_prompt"]].notna().all().all()
    for f in ("metrics_summary.json", "maps/case0/confidence.nii.gz", "figures/case0.png"):
        assert (out / f).exists(), f
