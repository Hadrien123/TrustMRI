#!/usr/bin/env python3
"""
Batch-generate PNGs of ORIGINAL vs IMPUTED MRI + error map for BraTS 2023.
Fully non-interactive (Agg backend).

Expected layout (group folder level is optional):

    data_root/
        GLI/                      <- or "Other", etc.
            BraTS-SSA-00000001/
                t1n/
                    original/xxx.nii.gz
                    imputed/xxx.nii.gz      <- present only for imputed modalities
                t1c/ ...
                t2w/ ...
                t2f/ ...
                seg/ (optional)             <- only used to choose tumor-centered slices

For every subject and every modality that has BOTH original/ and imputed/ files,
one PNG is written:

    <out_dir>/<group>/<subject>_<modality>.png
    columns: Original | Imputed | |Original - Imputed|   (rows = axial slices)

Also writes <out_dir>/metrics.csv (MAE, RMSE, PSNR, optional SSIM; brain voxels only).
Intensities of both volumes are normalized with the ORIGINAL volume's 1-99th
percentile (brain voxels), so error values are in comparable normalized units.

Install:  pip install nibabel numpy matplotlib      (optional for --ssim: scikit-image)

Examples:
    python brats2023_imputation_png.py /data/root out_pngs
    python brats2023_imputation_png.py /data/root out_pngs --groups GLI --modalities t1c t2w
    python brats2023_imputation_png.py /data/root out_pngs --limit 10 --n-slices 4 --err-max 0.3 --ssim
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

MODALITY_NAMES = {"t1n": "T1 (native)", "t1c": "T1c", "t2w": "T2", "t2f": "T2-FLAIR"}


# ----------------------------------------------------------------------------
# Discovery / IO
# ----------------------------------------------------------------------------
def first_nifti(d: Path):
    if not d.is_dir():
        return None
    hits = sorted(d.glob("*.nii*"))
    return hits[0] if hits else None

def is_subject_dir(d: Path) -> bool:
    return any((d / m / s).is_dir() for m in MODALITY_NAMES for s in ("original", "imputed"))

def discover_subjects(root, groups=None, subjects=None, limit=None):
    """Return list of (group_name, subject_dir). Works with or without a group level."""
    root = Path(root)
    found = []
    for a in sorted(p for p in root.iterdir() if p.is_dir()):
        if is_subject_dir(a):
            found.append(("", a))
        elif groups is None or a.name in groups:
            for b in sorted(p for p in a.iterdir() if p.is_dir()):
                if is_subject_dir(b):
                    found.append((a.name, b))
    if subjects:
        found = [(g, d) for g, d in found if d.name in set(subjects)]
    return found[:limit] if limit else found

def load_nifti(path):
    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.asarray(img.dataobj, dtype=np.float32)

def load_seg(subject_dir: Path):
    """Optional segmentation (used only to pick slices). Looks in seg/original, seg/, then *seg*.nii*."""
    for d in (subject_dir / "seg" / "original", subject_dir / "seg"):
        f = first_nifti(d)
        if f:
            return load_nifti(f).round().astype(np.uint8)
    hits = sorted(subject_dir.glob("*seg*.nii*"))
    return load_nifti(hits[0]).round().astype(np.uint8) if hits else None

# ----------------------------------------------------------------------------
# Processing
# ----------------------------------------------------------------------------
def norm_params(vol, p_low=1, p_high=99):
    brain = vol[vol > 0]
    if brain.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(brain, [p_low, p_high])
    return float(lo), float(hi)


def apply_norm(vol, params):
    lo, hi = params
    return np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1)


def pick_slices(seg, ref, n):
    """Axial slice indices: spread over the tumor if a seg exists, else over the central brain."""
    if seg is not None and seg.shape == ref.shape and seg.any():
        zs = np.where((seg > 0).any(axis=(0, 1)))[0]
        if n == 1:
            return [int(np.argmax((seg > 0).sum(axis=(0, 1))))]
        return sorted({int(z) for z in np.linspace(zs[0], zs[-1], n + 2)[1:-1]})
    zs = np.where((ref > 0).any(axis=(0, 1)))[0]
    span = zs[-1] - zs[0]
    return sorted({int(z) for z in np.linspace(zs[0] + 0.2 * span, zs[-1] - 0.2 * span, n)})


def compute_metrics(orig_n, imp_n, mask, use_ssim=False):
    err = np.abs(orig_n - imp_n)[mask]
    mae = float(err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    out = {
        "MAE": mae,
        "RMSE": rmse,
        "PSNR": float(20 * np.log10(1.0 / rmse)) if rmse > 0 else float("inf"),
    }
    if use_ssim:
        try:
            from skimage.metrics import structural_similarity

            _, smap = structural_similarity(orig_n, imp_n, data_range=1.0, full=True)
            out["SSIM"] = float(smap[mask].mean())
        except ImportError:
            print("  [warn] scikit-image not installed; skipping SSIM (pip install scikit-image)")
    return out


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def render(group, subject_dir, mod, args):
    f_orig = first_nifti(subject_dir / mod / "original")
    f_imp = first_nifti(subject_dir / mod / "imputed")
    if f_orig is None or f_imp is None:
        return None  # nothing to compare for this modality

    orig, imp = load_nifti(f_orig), load_nifti(f_imp)
    if orig.shape != imp.shape:
        raise ValueError(f"shape mismatch {orig.shape} vs {imp.shape}")

    params = norm_params(orig)
    o_n, i_n = apply_norm(orig, params), apply_norm(imp, params)
    mask = orig > 0
    metrics = compute_metrics(o_n, i_n, mask, args.ssim)

    slices = pick_slices(load_seg(subject_dir), orig, args.n_slices)
    nrows = len(slices)
    fig, axes = plt.subplots(nrows, 3, figsize=(3.3 * 3 + 0.8, 3.3 * nrows + 0.9),
                             facecolor="black", squeeze=False)
    err_im = None
    for r, z in enumerate(slices):
        err = np.abs(o_n[:, :, z] - i_n[:, :, z]) * mask[:, :, z]
        panels = [
            ("Original", o_n[:, :, z], dict(cmap="gray", vmin=0, vmax=1)),
            ("Imputed", i_n[:, :, z], dict(cmap="gray", vmin=0, vmax=1)),
            ("|Original - Imputed|", err, dict(cmap="inferno", vmin=0, vmax=args.err_max)),
        ]
        for c, (title, img, kw) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(np.rot90(img), **kw)
            ax.axis("off")
            if c == 2:
                err_im = im
            if r == 0:
                ax.set_title(title, color="white", fontsize=11)
            if c == 0:
                ax.text(-0.02, 0.5, f"z={z}", transform=ax.transAxes, color="white",
                        rotation=90, va="center", ha="right", fontsize=9)

    head = f"{group + ' / ' if group else ''}{subject_dir.name}  -  {MODALITY_NAMES[mod]}"
    stats = f"MAE={metrics['MAE']:.4f}   RMSE={metrics['RMSE']:.4f}   PSNR={metrics['PSNR']:.2f} dB"
    if "SSIM" in metrics:
        stats += f"   SSIM={metrics['SSIM']:.3f}"
    fig.suptitle(f"{head}\n{stats}", color="white", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.92, 0.93))
    cax = fig.add_axes([0.935, 0.15, 0.015, 0.6])
    cb = fig.colorbar(err_im, cax=cax)
    cb.ax.tick_params(colors="white", labelsize=8)
    cb.set_label("abs. error (normalized)", color="white", fontsize=8)

    out_dir = Path(args.out_dir) / group
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{subject_dir.name}_{mod}.png", dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"group": group, "subject": subject_dir.name, "modality": mod, **metrics}


def main():
    p = argparse.ArgumentParser(description="Original vs imputed BraTS2023 PNGs with error maps")
    p.add_argument("data_root")
    p.add_argument("out_dir")
    p.add_argument("--groups", nargs="+", help="Only these group folders (e.g. GLI Other)")
    p.add_argument("--subjects", nargs="+", help="Only these subject folder names")
    p.add_argument("--modalities", nargs="+", default=list(MODALITY_NAMES), choices=list(MODALITY_NAMES))
    p.add_argument("--limit", type=int, help="Process at most N subjects")
    p.add_argument("--n-slices", type=int, default=3, help="Axial slices per PNG (rows)")
    p.add_argument("--err-max", type=float, default=0.5, help="Color-scale max for the error map")
    p.add_argument("--ssim", action="store_true", help="Also compute SSIM (needs scikit-image)")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    cases = discover_subjects(args.data_root, args.groups, args.subjects, args.limit)
    print(f"Found {len(cases)} subject(s)")

    rows = []
    for i, (group, sdir) in enumerate(cases, 1):
        for mod in args.modalities:
            try:
                res = render(group, sdir, mod, args)
            except Exception as e:  # keep going on bad cases
                print(f"[{i}/{len(cases)}] {sdir.name} {mod} FAILED: {e}")
                continue
            if res:
                rows.append(res)
                print(f"[{i}/{len(cases)}] {group}/{sdir.name} {mod}  MAE={res['MAE']:.4f}")

    if not rows:
        print("Nothing written - no modality had both original/ and imputed/ files.")
        return

    keys = ["group", "subject", "modality"] + [k for k in ("MAE", "RMSE", "PSNR", "SSIM") if k in rows[0]]
    with open(Path(args.out_dir) / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    by_mod = defaultdict(list)
    for r in rows:
        by_mod[r["modality"]].append(r["MAE"])
    print(f"\nWrote {len(rows)} PNG(s) + metrics.csv to {args.out_dir}")
    for m, v in by_mod.items():
        print(f"  {m}: mean MAE = {np.mean(v):.4f} over {len(v)} case(s)")


if __name__ == "__main__":
    main()