#!/usr/bin/env python3
"""
Batch-generate PNGs comparing ORIGINAL (ground truth) vs IMPUTED MRI for BraTS 2023,
with error maps, shown in the three anatomical planes (sagittal / coronal / axial).
Optionally compares TWO models against each other. Non-interactive (Agg backend).

Expected layout (group folder level is optional):

    data_root/
        GLI/                      <- or "Other", etc.
            BraTS-SSA-00000001/
                t1n/
                    original/xxx.nii.gz     <- ground truth
                    imputed/xxx.nii.gz      <- model 1 (folder name configurable)
                t1c/ ...  t2w/ ...  t2f/ ...
                seg/ (optional)             <- only used to center the views on the tumor

Single model  (default):
    columns:  Ground truth | Model | |GT - Model|
Two models (--model2-dir and/or --model2-root):
    columns:  Ground truth | Model 1 | Model 2 | |GT - M1| | |GT - M2| | Comparison
    Comparison (--compare):
        delta   (default)  signed error difference |GT-M1| - |GT-M2|
                           blue = model 1 better, red = model 2 better
        absdiff            |M1 - M2|  (where the two models disagree)

Rows are the planes (sagittal, coronal, axial), each at the tumor centroid
(or the brain centroid if there is no segmentation).

Output: <out_dir>/<group>/<subject>_<modality>[_compare].png  and  <out_dir>/metrics.csv
All volumes are normalized with the ground truth's 1-99th percentile (brain voxels); errors and
metrics are computed on brain voxels only (GT > 0).

Install:  pip install nibabel numpy matplotlib      (optional for --ssim: scikit-image)

Examples:
    # one model
    python brats2023_imputation_png.py /data/root out_pngs
    # two models stored side by side inside each modality folder (imputed/ and imputed_b/)
    python brats2023_imputation_png.py /data/root out_cmp --model2-dir imputed_b --labels "UNet" "Diffusion"
    # model 2 stored in a different root with the same layout
    python brats2023_imputation_png.py /data/root out_cmp --model2-root /data/root_model2
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
PLANE_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}


# ----------------------------------------------------------------------------
# Discovery / IO
# ----------------------------------------------------------------------------
def first_nifti(d: Path):
    if not d.is_dir():
        return None
    hits = sorted(d.glob("*.nii*"))
    return hits[0] if hits else None


def is_subject_dir(d: Path, model_dir: str) -> bool:
    return any((d / m / s).is_dir() for m in MODALITY_NAMES for s in ("original", model_dir))


def discover_subjects(root, model_dir, groups=None, subjects=None, limit=None):
    """Return list of (group_name, subject_dir). Works with or without a group level."""
    root = Path(root)
    found = []
    for a in sorted(p for p in root.iterdir() if p.is_dir()):
        if is_subject_dir(a, model_dir):
            found.append(("", a))
        elif groups is None or a.name in groups:
            for b in sorted(p for p in a.iterdir() if p.is_dir()):
                if is_subject_dir(b, model_dir):
                    found.append((a.name, b))
    if subjects:
        found = [(g, d) for g, d in found if d.name in set(subjects)]
    return found[:limit] if limit else found


def load_nifti(path):
    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.asarray(img.dataobj, dtype=np.float32)


def load_seg(subject_dir: Path):
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


def center_point(ref, seg):
    """(x, y, z) voxel at the tumor centroid, else the brain centroid."""
    if seg is not None and seg.shape == ref.shape and seg.any():
        pts = np.argwhere(seg > 0)
    else:
        pts = np.argwhere(ref > 0)
    return tuple(int(round(c)) for c in pts.mean(axis=0))


def take(vol, plane, idx):
    """2D slice in the requested plane, rotated for display (superior/anterior up)."""
    return np.rot90(np.take(vol, idx, axis=PLANE_AXIS[plane]))


def smooth2d(img, sigma):
    """Separable Gaussian blur (numpy only); used to hide through-plane blockiness in reformats."""
    if sigma <= 0:
        return img
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.asarray(img, dtype=np.float32)
    for a in (0, 1):
        out = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), a, out)
    return out


def compute_metrics(gt_n, pr_n, mask, use_ssim=False):
    err = np.abs(gt_n - pr_n)[mask]
    rmse = float(np.sqrt((err ** 2).mean()))
    out = {
        "MAE": float(err.mean()),
        "RMSE": rmse,
        "PSNR": float(20 * np.log10(1.0 / rmse)) if rmse > 0 else float("inf"),
    }
    if use_ssim:
        try:
            from skimage.metrics import structural_similarity

            _, smap = structural_similarity(gt_n, pr_n, data_range=1.0, full=True)
            out["SSIM"] = float(smap[mask].mean())
        except ImportError:
            print("  [warn] scikit-image not installed; skipping SSIM (pip install scikit-image)")
    return out


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def add_hcbar(fig, top_axes, im, label):
    """Horizontal colorbar below the columns in top_axes (uses their grid-cell extents)."""
    p0 = top_axes[0].get_position(original=True)
    p1 = top_axes[-1].get_position(original=True)
    w = p1.x1 - p0.x0
    cax = fig.add_axes([p0.x0 + 0.1 * w, 0.05, 0.8 * w, 0.014])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(colors="white", labelsize=8)
    cb.outline.set_edgecolor("white")
    cax.set_title(label, color="white", fontsize=8, pad=3)


def render(group, subject_dir, mod, args):
    root = Path(args.data_root)
    f_gt = first_nifti(subject_dir / mod / "original")
    f_m1 = first_nifti(subject_dir / mod / args.model1_dir)
    if f_gt is None or f_m1 is None:
        return None  # nothing to compare for this modality

    two = args.two_models
    f_m2 = None
    if two:
        m2_root = Path(args.model2_root) if args.model2_root else root
        m2_dir = args.model2_dir or args.model1_dir
        f_m2 = first_nifti(m2_root / subject_dir.relative_to(root) / mod / m2_dir)
        if f_m2 is None:
            print(f"  [warn] {subject_dir.name} {mod}: model 2 file not found, skipped")
            return None

    gt, m1 = load_nifti(f_gt), load_nifti(f_m1)
    m2 = load_nifti(f_m2) if two else None
    for name, arr in (("model 1", m1), ("model 2", m2)):
        if arr is not None and arr.shape != gt.shape:
            raise ValueError(f"{name} shape {arr.shape} != ground truth {gt.shape}")

    params = norm_params(gt)
    gt_n, m1_n = apply_norm(gt, params), apply_norm(m1, params)
    m2_n = apply_norm(m2, params) if two else None
    mask = gt > 0

    labels = args.labels or (["Model 1", "Model 2"] if two else ["Imputed"])
    l1, l2 = labels[0], (labels[1] if len(labels) > 1 else "Model 2")

    # ---- metrics ----
    row = {"group": group, "subject": subject_dir.name, "modality": mod}
    met1 = compute_metrics(gt_n, m1_n, mask, args.ssim)
    if two:
        met2 = compute_metrics(gt_n, m2_n, mask, args.ssim)
        row.update({f"{k}_model1": v for k, v in met1.items()})
        row.update({f"{k}_model2": v for k, v in met2.items()})
        row["MAE_delta_m1_minus_m2"] = met1["MAE"] - met2["MAE"]
        row["MAD_between_models"] = float(np.abs(m1_n - m2_n)[mask].mean())
    else:
        row.update(met1)

    # ---- panels: (title, 3D volume, imshow kwargs, kind) ----
    gray = dict(cmap="gray", vmin=0, vmax=1)
    errkw = dict(cmap="inferno", vmin=0, vmax=args.err_max)
    e1 = np.abs(gt_n - m1_n)
    panels = [("Ground truth", gt_n, gray, "img"), (l1, m1_n, gray, "img")]
    if two:
        e2 = np.abs(gt_n - m2_n)
        panels += [(l2, m2_n, gray, "img"),
                   (f"|GT - {l1}|", e1, errkw, "err"),
                   (f"|GT - {l2}|", e2, errkw, "err")]
        if args.compare == "delta":
            panels.append((f"Error difference\n{l1} - {l2}", e1 - e2,
                           dict(cmap="coolwarm", vmin=-args.cmp_max, vmax=args.cmp_max), "cmp"))
            cmp_label = f"|err {l1}| - |err {l2}|" # (blue: {l1} better, red: {l2} better)
        else:
            panels.append((f"|{l1} - {l2}|", np.abs(m1_n - m2_n),
                           dict(cmap="inferno", vmin=0, vmax=args.cmp_max), "cmp"))
            cmp_label = f"|{l1} - {l2}|  (model disagreement)"
    else:
        panels.append((f"|GT - {l1}|", e1, errkw, "err"))

    center = center_point(gt, load_seg(subject_dir))
    planes = args.planes
    nrows, ncols = len(planes), len(panels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols + 0.6, 2.9 * nrows + 1.9),
                             facecolor="black", squeeze=False)
    last_im = {}
    for r, plane in enumerate(planes):
        idx = center[PLANE_AXIS[plane]]
        mask2d = take(mask, plane, idx)
        # Sagittal/coronal are reformats across the (thick) acquisition slices -> blocky with
        # nearest-neighbour; smooth them lightly and use a smooth interpolator. Display only.
        reformat = plane != args.acq_plane
        interp = args.reformat_interp if reformat else args.interp
        for c, (title, vol, kw, kind) in enumerate(panels):
            ax = axes[r, c]
            s = take(vol, plane, idx)
            if reformat:
                s = smooth2d(s, args.reformat_sigma)
            if kind != "img":
                s = np.ma.masked_where(~mask2d, s)  # keep background black
            last_im[kind] = ax.imshow(s, interpolation=interp, **kw)
            ax.axis("off")
            if r == 0:
                ax.set_title(title, color="white", fontsize=10)
            if c == 0:
                ax.text(-0.03, 0.5, f"{plane.capitalize()}\n({'xyz'[PLANE_AXIS[plane]]}={idx})",
                        transform=ax.transAxes, color="white", rotation=90,
                        va="center", ha="right", fontsize=10)

    # ---- header ----
    def fmt(m):
        s = f"MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  PSNR={m['PSNR']:.2f} dB"
        return s + (f"  SSIM={m['SSIM']:.3f}" if "SSIM" in m else "")

    head = f"{group + ' / ' if group else ''}{subject_dir.name}  -  {MODALITY_NAMES[mod]}"
    lines = [head, f"{l1}:  {fmt(met1)}"] + ([f"{l2}:  {fmt(met2)}"] if two else [])
    print("\n".join(lines))
    fig.suptitle("\n".join(lines), color="white", fontsize=11, y=0.985)
    fig.tight_layout(rect=(0, 0.1, 1, 0.9 if two else 0.92))

    # ---- colorbars ----
    err_cols = [c for c, p in enumerate(panels) if p[3] == "err"]
    add_hcbar(fig, [axes[0, c] for c in err_cols], last_im["err"], "abs. error (normalized)")
    if two:
        cmp_col = [c for c, p in enumerate(panels) if p[3] == "cmp"][0]
        add_hcbar(fig, [axes[0, cmp_col]], last_im["cmp"], cmp_label)

    out_dir = Path(args.out_dir) / group
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_compare" if two else ""
    fig.savefig(out_dir / f"{subject_dir.name}_{mod}{suffix}.png", dpi=args.dpi,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return row


def main():
    p = argparse.ArgumentParser(description="Original vs imputed BraTS2023 PNGs with error maps")
    p.add_argument("data_root")
    p.add_argument("out_dir")
    p.add_argument("--model1-dir", default="imputed", help="Folder name of model 1 inside each modality folder")
    p.add_argument("--model2-dir", help="Folder name of model 2 (enables two-model comparison)")
    p.add_argument("--model2-root", help="Separate data root for model 2 (same layout); enables comparison")
    p.add_argument("--labels", nargs="+", help="Display names, e.g. --labels UNet Diffusion")
    p.add_argument("--compare", choices=["delta", "absdiff"], default="delta",
                   help="Comparison panel: signed error difference (default) or |M1-M2|")
    p.add_argument("--planes", nargs="+", default=["sagittal", "coronal", "axial"],
                   choices=list(PLANE_AXIS))
    p.add_argument("--groups", nargs="+", help="Only these group folders (e.g. GLI Other)")
    p.add_argument("--subjects", nargs="+", help="Only these subject folder names")
    p.add_argument("--modalities", nargs="+", default=list(MODALITY_NAMES), choices=list(MODALITY_NAMES))
    p.add_argument("--limit", type=int, help="Process at most N subjects")
    p.add_argument("--err-max", type=float, default=0.5, help="Color-scale max for error maps")
    p.add_argument("--cmp-max", type=float, default=0.25, help="Color-scale max (+/-) for the comparison panel")
    p.add_argument("--ssim", action="store_true", help="Also compute SSIM (needs scikit-image)")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--interp", default="nearest",
                   help="imshow interpolation: nearest = crisp voxels (default); antialiased/bicubic = smoothed")
    p.add_argument("--acq-plane", default="axial", choices=list(PLANE_AXIS),
                   help="Native acquisition plane; drawn with --interp, the other planes as smoothed reformats")
    p.add_argument("--reformat-interp", default="lanczos",
                   help="imshow interpolation for the reformatted (non-acquisition) planes")
    p.add_argument("--reformat-sigma", type=float, default=0.8,
                   help="Gaussian sigma (voxels) applied to reformatted planes for display; 0 = off")
    args = p.parse_args()

    args.two_models = bool(args.model2_dir or args.model2_root)
    if args.two_models and not args.model2_root and args.model2_dir == args.model1_dir:
        p.error("--model2-dir must differ from --model1-dir when no --model2-root is given")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    cases = discover_subjects(args.data_root, args.model1_dir, args.groups, args.subjects, args.limit)
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
                print(f"[{i}/{len(cases)}] {group}/{sdir.name} {mod} done")

    if not rows:
        print("Nothing written - no modality had the required original/model files.")
        return

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(Path(args.out_dir) / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} PNG(s) + metrics.csv to {args.out_dir}")
    mae_keys = [k for k in keys if k.startswith("MAE") and "delta" not in k]
    for k in mae_keys:
        by_mod = defaultdict(list)
        for r in rows:
            by_mod[r["modality"]].append(r[k])
        print("  " + k + ": " + ", ".join(f"{m}={np.mean(v):.4f} (n={len(v)})" for m, v in by_mod.items()))


if __name__ == "__main__":
    main()