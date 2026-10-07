#!/usr/bin/env python3
"""Bar chart: pretrained vs fine-tuned BraSyn on BraTS 2023 and BraTS-Africa.

Input: the per_case_metrics.csv files written by compare_imputation_models.py
(one row per subject x modality, columns pretrained_<METRIC> / finetuned_<METRIC>).

  Bars        mean over all subject x modality pairs
  Error bars  95% confidence interval, bootstrap over patients (a patient's
              4 modalities are averaged first, because they are not independent)
  Brackets    paired Wilcoxon signed-rank test, pretrained vs fine-tuned
              (same test as compare_imputation_models.py)

Usage (on the workstation, inside an environment with pandas, numpy, scipy, matplotlib):
    python3 plot_finetuning_metrics.py
    python3 plot_finetuning_metrics.py --out slides/metrics.png --dpi 300
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

RESULTS = Path.home() / "software/TRUSTMRI/results"

# (column name, panel title, hint, value format)
METRICS = [
    ("SSIM", "MSSIM", "↑ higher is better", "{:.3f}"),
    ("PSNR", "PSNR (dB)", "↑ higher is better", "{:.2f}"),
    ("CORR", "Correlation", "↑ higher is better", "{:.3f}"),
    ("RMSE", "RMSE", "↓ lower is better", "{:.3f}"),
    ("MAE", "MAE", "↓ lower is better", "{:.3f}"),
]
MODELS = [("pretrained", "Pretrained"), ("finetuned", "Fine-tuned")]
COLORS = {
    ("BraTS 2023", "pretrained"): "#E8710A",
    ("BraTS 2023", "finetuned"): "#F8C38A",
    ("BraTS-Africa", "pretrained"): "#1A73E8",
    ("BraTS-Africa", "finetuned"): "#A8C7FA",
}
X = {  # bar positions: two groups of two bars
    ("BraTS 2023", "pretrained"): 0.0,
    ("BraTS 2023", "finetuned"): 1.0,
    ("BraTS-Africa", "pretrained"): 2.6,
    ("BraTS-Africa", "finetuned"): 3.6,
}


def bootstrap_ci(df, col, rng, n_boot=5000):
    """95% CI of the mean, resampling patients rather than single scans."""
    per_subject = df.groupby("subject")[col].mean().to_numpy()
    idx = rng.integers(0, len(per_subject), size=(n_boot, len(per_subject)))
    boots = per_subject[idx].mean(axis=1)
    return np.percentile(boots, [2.5, 97.5])


def stars(p):
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brats", type=Path,
                    default=RESULTS / "brats2023_model_comparison_preproc/per_case_metrics.csv")
    ap.add_argument("--africa", type=Path,
                    default=RESULTS / "africa_model_comparison/per_case_metrics.csv")
    ap.add_argument("--out", type=Path, default=Path("brasyn_finetuning_metrics.png"))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed, for reproducible error bars")
    args = ap.parse_args()

    data = {"BraTS 2023": pd.read_csv(args.brats), "BraTS-Africa": pd.read_csv(args.africa)}
    rng = np.random.default_rng(args.seed)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), dpi=args.dpi)
    axes = axes.ravel()

    print(f"{'dataset':<13}{'metric':<7}{'model':<11}{'mean':>9}{'CI low':>9}{'CI high':>9}{'p':>11}")
    for ax, (metric, title, hint, fmt) in zip(axes, METRICS):
        top = 0.0
        brackets = []
        for ds, df in data.items():
            highest = 0.0
            for model, _ in MODELS:
                col = f"{model}_{metric}"
                mean = df[col].mean()
                lo, hi = bootstrap_ci(df, col, rng)
                x = X[(ds, model)]
                ax.bar(x, mean, 0.85, color=COLORS[(ds, model)], zorder=2)
                ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
                            ecolor="#333", capsize=4, lw=1.2, zorder=3)
                ax.text(x, hi, fmt.format(mean), ha="center", va="bottom", fontsize=9.5, zorder=4,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
                highest = max(highest, hi)
                p_txt = ""
                if model == "finetuned":
                    p = wilcoxon(df[f"finetuned_{metric}"], df[f"pretrained_{metric}"]).pvalue
                    brackets.append((X[(ds, "pretrained")], x, highest, stars(p)))
                    p_txt = f"{p:.2e}"
                print(f"{ds:<13}{metric:<7}{model:<11}{mean:9.4f}{lo:9.4f}{hi:9.4f}{p_txt:>11}")
            top = max(top, highest)

        for x0, x1, h, s in brackets:
            y = h + top * 0.09
            ax.plot([x0, x0, x1, x1], [y, y + top * 0.02, y + top * 0.02, y], color="#333", lw=1)
            ax.text((x0 + x1) / 2, y + top * 0.025, s, ha="center", va="bottom", fontsize=10.5,
                    fontweight="normal" if s == "n.s." else "bold")

        ax.set_ylim(0, top * 1.25)
        ax.set_title(f"{title}\n", fontsize=13, fontweight="bold", pad=2)
        ax.text(0.5, 1.01, hint, transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color="#555")
        ax.set_xticks(list(X.values()))
        ax.set_xticklabels([label for _, label in MODELS] * 2, fontsize=9.5)
        for xc, ds in [(0.5, "BraTS 2023"), (3.1, "BraTS-Africa")]:
            ax.text(xc, -0.13, ds, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=10.5, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", zorder=0)
        ax.set_axisbelow(True)
        if metric == "PSNR":
            ax.set_ylabel("dB")

    # Last panel: legend and caption
    leg = axes[5]
    leg.axis("off")
    handles = [Patch(color=c, label=f"{ds} – {dict(MODELS)[m]}") for (ds, m), c in COLORS.items()]
    leg.legend(handles=handles, loc="upper left", frameon=False, fontsize=11.5, bbox_to_anchor=(0.02, 0.95))
    n = {ds: df["subject"].nunique() for ds, df in data.items()}
    leg.text(0.02, 0.42,
             f"Bars: mean over all scans (BraTS 2023: {n['BraTS 2023']} patients × 4 modalities;\n"
             f"BraTS-Africa: {n['BraTS-Africa']} held-out patients × 4 modalities).\n"
             "Error bars: 95% confidence interval\n(bootstrap over patients).\n"
             "Brackets: paired Wilcoxon test, pretrained vs fine-tuned\n"
             "*** p < 0.001,  n.s. = not significant.\n"
             "NRMSE omitted: identical to RMSE on [0, 1]-normalized data.",
             transform=leg.transAxes, va="top", fontsize=9.8, color="#333", linespacing=1.45)

    fig.suptitle("Missing-modality reconstruction: pretrained vs fine-tuned BraSyn",
                 fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3.2, w_pad=2.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor="white", bbox_inches="tight")
    print(f"\nSaved {args.out.resolve()}")


if __name__ == "__main__":
    main()
