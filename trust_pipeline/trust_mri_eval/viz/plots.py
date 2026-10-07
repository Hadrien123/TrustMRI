"""One PNG panel per patient: an axial slice of the quality maps, the trust maps and the confidence heatmap
next to the true patch labels.

Magnitude maps use a one-hue sequential ramp; confidence uses a diverging orange (likely wrong) -
grey (0.5) - blue (likely correct) map.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SURFACE, INK, INK_3 = "#fcfcfb", "#0b0b0b", "#8a8984"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f3f7fd", "#9cc0ee", BLUE, "#0d3a73"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#a63d12", ORANGE, "#d9d8d4", BLUE, "#0d3a73"])


def largest_tumour_slice(mask: np.ndarray) -> int:
    """Axial index (axis 0, inferior-superior in BraSyn's IPL space) with the largest mask area; middle slice
    if the mask is empty."""
    area = mask.reshape(mask.shape[0], -1).sum(1)
    return int(np.argmax(area)) if area.any() else mask.shape[0] // 2


def _show(ax, image, title, cmap="gray", vmin=None, vmax=None, overlay=None, overlay_cmap=None):
    ax.imshow(image, cmap=cmap if overlay is None else "gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    im = None
    if overlay is not None:
        im = ax.imshow(np.ma.masked_invalid(overlay), cmap=overlay_cmap, vmin=0, vmax=1, alpha=0.85,
                       interpolation="nearest")
    ax.set_title(title, color=INK, fontsize=10)
    ax.set_axis_off()
    mappable = im or ax.images[0]
    if overlay is not None or cmap != "gray":
        cb = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, colors=INK_3)


def patient_panel(s: dict[str, np.ndarray], path: str | Path, title: str) -> None:
    """2 x 6 panel: local quality on the first row, trust on the second. ``s`` holds 2D slices: image,
    imputed, seg, gt_mask, and the maps local_ssim, local_l1, variance, agreement, frac_imputation,
    frac_prompt, frac_interaction, confidence, patch_label (NaN outside where they are defined)."""
    fig, axes = plt.subplots(2, 6, figsize=(23, 8), facecolor=SURFACE)
    ax = axes.ravel()
    _show(ax[0], s["image"], "Real image", vmin=0, vmax=1)
    _show(ax[1], s["imputed"], "Evaluated imputed image", vmin=0, vmax=1)
    _show(ax[2], np.nan_to_num(s["local_ssim"], nan=1.0), "Local SSIM vs real", SEQ.reversed(), 0, 1)
    _show(ax[3], np.nan_to_num(s["local_l1"]), "Local L1 error vs real", SEQ, 0)
    _show(ax[4], np.nan_to_num(s["variance"]), "Local variance across images", SEQ, 0)
    _show(ax[5], s["image"], "Evaluated segmentation vs reference", vmin=0, vmax=1)
    for mask, color in ((s["gt_mask"], ORANGE), (s["seg"], BLUE)):
        if mask.any():
            ax[5].contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.5)
    ax[5].legend(handles=[Patch(color=BLUE, label="segmentation"), Patch(color=ORANGE, label="reference")],
                 loc="lower right", fontsize=7, facecolor=SURFACE, edgecolor=SURFACE, framealpha=0.9)
    _show(ax[6], s["agreement"], "Agreement of the N x K segmentations", SEQ, 0, 1)
    for i, comp in enumerate(("imputation", "prompt", "interaction")):
        _show(ax[7 + i], s[f"frac_{comp}"], f"ANOVA share: {comp}", SEQ, 0, 1)
    _show(ax[10], s["image"], "Confidence P(correct)", vmin=0, vmax=1, overlay=s["confidence"],
          overlay_cmap=DIVERGING)
    _show(ax[11], s["image"], "Patch correct (reference)", vmin=0, vmax=1, overlay=s["patch_label"],
          overlay_cmap=DIVERGING)
    fig.suptitle(title, color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
