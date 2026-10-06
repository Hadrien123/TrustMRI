"""Figures: per-patient slice panels and model-evaluation charts (PNG, matplotlib).

Colour roles: magnitude maps use one-hue sequential ramps; confidence uses a diverging
orange (likely wrong) - grey (0.5) - blue (likely correct) map; compared models use the first
three categorical slots in a fixed order; text stays in ink colours.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#008300"]          # categorical slots 1, 2, 3, 6
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fd", "#9cc0ee", "#2a78d6", "#0d3a73"])
SEQ_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ["#fdf4ef", "#f5b393", "#eb6834", "#7a2a0b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#a63d12", "#eb6834", "#d9d8d4", "#2a78d6", "#0d3a73"])


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": INK_3, "ytick.color": INK_3, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "lines.linewidth": 2.0, "legend.frameon": False,
    })


# ---------------------------------------------------------------------------- slices
def largest_tumour_slice(mask: np.ndarray) -> int:
    """Axial index (last axis) with the largest mask area; middle slice if the mask is empty."""
    area = mask.reshape(-1, mask.shape[-1]).sum(0)
    return int(np.argmax(area)) if area.any() else mask.shape[-1] // 2


def _show(ax, img, title, cmap="gray", vmin=None, vmax=None, brain=None, colorbar=True):
    data = np.ma.masked_where(~brain, img) if brain is not None else img
    im = ax.imshow(np.ma.transpose(data) if np.ma.isMaskedArray(data) else data.T, cmap=cmap,
                   vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
    ax.set_title(title)
    ax.set_axis_off()
    if colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, colors=INK_3)
    return im


def patient_panel(s: dict[str, np.ndarray], path: str | Path, title: str,
                  group_names: tuple[str, ...] = ("imputation", "segmentation", "anatomy", "context")) -> None:
    """3 x 4 panel of one axial slice. ``s`` holds 2D slices: gt_image, imputed, variance,
    local_ssim, failure, consensus, gt_mask, agreement, frac_imputation, frac_prompt,
    frac_interaction, confidence, dominant (0 = none, 1.. = group index + 1), brain."""
    _style()
    brain = s["brain"].astype(bool)
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    ax = axes.ravel()
    _show(ax[0], s["gt_image"], "Ground-truth image", vmin=0, vmax=1)
    _show(ax[1], s["imputed"], "Imputed sample (n = 0)", vmin=0, vmax=1)
    _show(ax[2], s["variance"], "Variance across samples", SEQ, vmin=0, brain=brain)
    _show(ax[3], s["local_ssim"], "Local SSIM vs ground truth", SEQ.reversed(), vmin=0, vmax=1, brain=brain)

    _show(ax[4], s["gt_image"], "Failure map (patch error > thr)", vmin=0, vmax=1, colorbar=False)
    fail = np.ma.masked_where(s["failure"] == 0, s["failure"])
    ax[4].imshow(fail.T, cmap=ListedColormap([SERIES[1]]), origin="lower", alpha=0.75, interpolation="nearest")

    _show(ax[5], s["gt_image"], "Consensus vs reference", vmin=0, vmax=1, colorbar=False)
    for mask, color in ((s["gt_mask"], SERIES[1]), (s["consensus"], SERIES[0])):
        if mask.any():
            ax[5].contour(mask.T.astype(float), levels=[0.5], colors=[color], linewidths=1.5, origin="lower")
    ax[5].legend(handles=[Patch(color=SERIES[0], label="consensus"), Patch(color=SERIES[1], label="reference")],
                 loc="lower right", fontsize=7, labelcolor=INK_2, frameon=True, facecolor=SURFACE,
                 edgecolor=SURFACE, framealpha=0.9)

    _show(ax[6], s["agreement"], "Agreement (fraction of masks)", SEQ, vmin=0, vmax=1, brain=brain)
    for i, comp in enumerate(("imputation", "prompt", "interaction")):
        _show(ax[7 + i], s[f"frac_{comp}"], f"ANOVA fraction: {comp}", SEQ_ORANGE if i == 0 else SEQ,
              vmin=0, vmax=1, brain=brain)

    _show(ax[10], s["gt_image"], "Confidence P(correct)", vmin=0, vmax=1, colorbar=False)
    conf = np.ma.masked_invalid(np.where(s["confidence"] >= 0, s["confidence"], np.nan))
    im = ax[10].imshow(conf.T, cmap=DIVERGING, vmin=0, vmax=1, origin="lower", alpha=0.8, interpolation="nearest")
    cb = plt.colorbar(im, ax=ax[10], fraction=0.046, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK_3)

    _show(ax[11], s["gt_image"], "Dominant contribution", vmin=0, vmax=1, colorbar=False)
    dom = np.ma.masked_where(s["dominant"] <= 0, s["dominant"])
    cmap = ListedColormap(SERIES[:len(group_names)])
    ax[11].imshow(dom.T, cmap=cmap, vmin=0.5, vmax=len(group_names) + 0.5, origin="lower", alpha=0.85,
                  interpolation="nearest")
    ax[11].legend(handles=[Patch(color=SERIES[i], label=g) for i, g in enumerate(group_names)],
                  loc="lower right", fontsize=7, labelcolor=INK_2, frameon=True, facecolor=SURFACE,
                 edgecolor=SURFACE, framealpha=0.9)

    fig.suptitle(title, color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------------------- model charts
def _models_axes(title: str, xlabel: str, ylabel: str):
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def reliability_plot(models: dict[str, dict], path: str | Path) -> None:
    """Reliability diagram: observed fraction correct vs mean predicted confidence per bin."""
    fig, ax = _models_axes("Reliability (test patients)", "Predicted P(correct)", "Observed fraction correct")
    ax.plot([0, 1], [0, 1], color=INK_3, lw=1, ls="--", label="perfect calibration")
    for i, (name, ev) in enumerate(models.items()):
        rel = ev["reliability"]
        ok = rel["count"] > 0
        ax.plot(rel["confidence"][ok], rel["accuracy"][ok], color=SERIES[i], marker="o", ms=4,
                label=f"{name} (ECE {ev['ece']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", fontsize=8, labelcolor=INK_2)
    _save(fig, path)


def roc_plot(models: dict[str, dict], path: str | Path) -> None:
    """ROC curves for detecting incorrect patches (score = 1 - confidence)."""
    fig, ax = _models_axes("Detecting incorrect patches", "False-positive rate", "True-positive rate")
    ax.plot([0, 1], [0, 1], color=INK_3, lw=1, ls="--")
    for i, (name, ev) in enumerate(models.items()):
        if "roc" in ev:
            ax.plot(ev["roc"]["fpr"], ev["roc"]["tpr"], color=SERIES[i],
                    label=f"{name} (AUROC {ev['auroc_incorrect']:.3f})")
    ax.legend(loc="lower right", fontsize=8, labelcolor=INK_2)
    _save(fig, path)


def risk_coverage_plot(models: dict[str, dict], path: str | Path) -> None:
    """Error rate of the kept patches vs fraction kept (most confident first)."""
    fig, ax = _models_axes("Risk-coverage", "Coverage (fraction of patches kept)", "Risk (error rate of kept)")
    for i, (name, ev) in enumerate(models.items()):
        rc = ev["risk_coverage"]
        ax.plot(rc["coverage"], rc["risk"], color=SERIES[i], label=f"{name} (AURC {ev['aurc']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8, labelcolor=INK_2)
    _save(fig, path)


def qubrats_plot(curves: dict[str, dict], path: str | Path) -> None:
    """QU-BraTS-style curve: Dice of kept voxels vs fraction of voxels filtered as the
    confidence threshold rises."""
    fig, ax = _models_axes("Filtering uncertain patches", "Fraction of ROI voxels filtered", "Dice of kept voxels")
    for i, (name, q) in enumerate(curves.items()):
        ax.plot(q["filtered"], q["dice"], color=SERIES[i], marker="o", ms=3,
                label=f"{name} (QU score {q['score']:.3f})")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=8, labelcolor=INK_2)
    _save(fig, path)


def coefficient_plot(coefs: dict[str, float], path: str | Path, top: int = 20) -> None:
    """Horizontal bars of the largest standardised coefficients (sign shown by direction)."""
    _style()
    items = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(items) + 1.2))
    names, vals = [k for k, _ in items], [v for _, v in items]
    ax.barh(names, vals, color=SERIES[0], height=0.7)
    ax.axvline(0, color=INK_3, lw=1)
    ax.set_title(f"Top {len(items)} coefficients (standardised features)", loc="left")
    ax.set_xlabel("Coefficient  (> 0: more likely correct)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7, colors=INK_2)
    _save(fig, path)


def _save(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
