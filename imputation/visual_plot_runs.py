#!/usr/bin/env python3
"""
Batch-generate PNGs for ONE model imputed several times (stochastic runs) on BraTS 2023:
the runs are averaged into a mean prediction, a voxel-wise variation map is computed across
the runs, and the mean is compared with the ground truth. Non-interactive (Agg backend).

Expected layout (group folder level is optional):

    data_root/                           e.g. results/W_preprocess_imputed_multiple_runs
        51_OtherNeoplasms/
            BraTS-SSA-00018-000/
                t1n/
                    original/xxx.nii.gz             <- ground truth
                    imputed/xxx-run01.nii ... runNN <- every NIfTI in here is one run
                t1c/ ...  t2w/ ...  t2f/ ...
                seg/ (optional)                     <- only used to center the views on the tumor

Columns:  Ground truth | Mean of N runs | |GT - mean| | Variation across runs
    Variation (--var):
        std     (default)  voxel-wise standard deviation across runs
        range              max - min across runs
        cv                 std / mean (coefficient of variation)

Rows are the planes (sagittal, coronal, axial), each at the tumor centroid
(or the brain centroid if there is no segmentation).

Output: <out_dir>/<group>/<subject>_<modality>_runs.png  and  <out_dir>/metrics.csv
        (--save-nifti also writes <subject>_<modality>_mean.nii.gz / _<var>.nii.gz, normalized units)
All volumes are normalized with the ground truth's 1-99th percentile (brain voxels); errors,
variation and metrics are computed on brain voxels only (GT > 0).

metrics.csv per subject/modality:
    MAE/RMSE/PSNR[/SSIM]_mean      mean prediction vs GT
    MAE/RMSE/PSNR_run_avg/_run_sd  each run vs GT, averaged (and SD) over runs
    var_mean                       mean variation over brain voxels
    corr_var_err                   Pearson r between variation and |GT - mean| (does variation flag errors?)

Install:  pip install nibabel numpy matplotlib      (optional for --ssim: scikit-image)

Example:
    python visual_plot_runs.py ../results/W_preprocess_imputed_multiple_runs out_runs
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
VAR_TITLES = {"std": "Std across runs", "range": "Range across runs (max - min)",
              "cv": "Coeff. of variation (std / mean)"}


# ----------------------------------------------------------------------------
# Discovery / IO
# ----------------------------------------------------------------------------
def all_niftis(d: Path):
    return sorted(d.glob("*.nii*")) if d.is_dir() else []


def first_nifti(d: Path):
    hits = all_niftis(d)
    return hits[0] if hits else None


def is_subject_dir(d: Path, runs_dir: str) -> bool:
    return any((d / m / s).is_dir() for m in MODALITY_NAMES for s in ("original", runs_dir))


def discover_subjects(root, runs_dir, groups=None, subjects=None, limit=None):
    """Return list of (group_name, subject_dir). Works with or without a group level."""
    root = Path(root)
    found = []
    for a in sorted(p for p in root.iterdir() if p.is_dir()):
        if is_subject_dir(a, runs_dir):
            found.append(("", a))
        elif groups is None or a.name in groups:
            for b in sorted(p for p in a.iterdir() if p.is_dir()):
                if is_subject_dir(b, runs_dir):
                    found.append((a.name, b))
    if subjects:
        found = [(g, d) for g, d in found if d.name in set(subjects)]
    return found[:limit] if limit else found


def load_img(path):
    return nib.as_closest_canonical(nib.load(str(path)))


def load_nifti(path):
    return np.asarray(load_img(path).dataobj, dtype=np.float32)


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


def variation_map(runs, mean, kind):
    """Voxel-wise variation across the run axis (axis 0) of normalized runs."""
    if kind == "range":
        return runs.max(axis=0) - runs.min(axis=0)
    std = runs.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
    if kind == "cv":
        return np.where(mean > 0.05, std / np.maximum(mean, 1e-8), 0.0).astype(np.float32)
    return std


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
    f_gt = first_nifti(subject_dir / mod / "original")
    f_runs = all_niftis(subject_dir / mod / args.runs_dir)
    if f_gt is None or not f_runs:
        return None  # nothing to compare for this modality

    gt_img = load_img(f_gt)
    gt = np.asarray(gt_img.dataobj, dtype=np.float32)
    params = norm_params(gt)
    gt_n = apply_norm(gt, params)
    mask = gt > 0

    runs = np.empty((len(f_runs),) + gt.shape, dtype=np.float32)
    for i, f in enumerate(f_runs):
        arr = load_nifti(f)
        if arr.shape != gt.shape:
            raise ValueError(f"run {f.name} shape {arr.shape} != ground truth {gt.shape}")
        runs[i] = apply_norm(arr, params)
    n = len(runs)
    mean_n = runs.mean(axis=0)
    var_n = variation_map(runs, mean_n, args.var)
    err = np.abs(gt_n - mean_n)

    # ---- metrics ----
    row = {"group": group, "subject": subject_dir.name, "modality": mod, "n_runs": n}
    met = compute_metrics(gt_n, mean_n, mask, args.ssim)
    row.update({f"{k}_mean": v for k, v in met.items()})
    per_run = [compute_metrics(gt_n, r, mask) for r in runs]
    for k in ("MAE", "RMSE", "PSNR"):
        vals = np.array([m[k] for m in per_run])
        row[f"{k}_run_avg"] = float(vals.mean())
        row[f"{k}_run_sd"] = float(vals.std(ddof=1)) if n > 1 else 0.0
    row[f"{args.var}_mean"] = float(var_n[mask].mean())
    v, e = var_n[mask], err[mask]
    row[f"corr_{args.var}_err"] = float(np.corrcoef(v, e)[0, 1]) if v.std() > 0 and e.std() > 0 else float("nan")

    if args.save_nifti:
        out_dir = Path(args.out_dir) / group
        out_dir.mkdir(parents=True, exist_ok=True)
        for tag, vol in (("mean", mean_n), (args.var, var_n)):
            nib.save(nib.Nifti1Image(vol.astype(np.float32), gt_img.affine),
                     str(out_dir / f"{subject_dir.name}_{mod}_{tag}.nii.gz"))

    # ---- panels: (title, 3D volume, imshow kwargs, kind) ----
    # runs usually differ only slightly -> auto-scale to the 99th percentile of brain voxels
    var_max = args.var_max or max(float(np.percentile(var_n[mask], 99)), 1e-6)
    panels = [("Ground truth", gt_n, dict(cmap="gray", vmin=0, vmax=1), "img"),
              (f"Mean of {n} runs", mean_n, dict(cmap="gray", vmin=0, vmax=1), "img"),
              ("|GT - mean|", err, dict(cmap="inferno", vmin=0, vmax=args.err_max), "err"),
              (VAR_TITLES[args.var], var_n, dict(cmap="viridis", vmin=0, vmax=var_max), "var")]

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
    def fmt(m, suffix=""):
        s = f"MAE={m['MAE' + suffix]:.4f}  RMSE={m['RMSE' + suffix]:.4f}  PSNR={m['PSNR' + suffix]:.2f} dB"
        return s + (f"  SSIM={m['SSIM' + suffix]:.3f}" if "SSIM" + suffix in m else "")

    head = f"{group + ' / ' if group else ''}{subject_dir.name}  -  {MODALITY_NAMES[mod]}  -  {n} runs"
    single = (f"Single runs:  MAE={row['MAE_run_avg']:.4f}±{row['MAE_run_sd']:.4f}  "
              f"PSNR={row['PSNR_run_avg']:.2f}±{row['PSNR_run_sd']:.2f} dB   |   "
              f"mean {args.var}={row[f'{args.var}_mean']:.4f}  r({args.var}, err)={row[f'corr_{args.var}_err']:.2f}")
    fig.suptitle("\n".join([head, f"Mean of runs:  {fmt(row, '_mean')}", single]),
                 color="white", fontsize=11, y=0.985)
    fig.tight_layout(rect=(0, 0.1, 1, 0.9))

    # ---- colorbars ----
    for kind, label in (("err", "abs. error (normalized)"), ("var", f"{args.var} across runs (normalized)")):
        col = [c for c, p in enumerate(panels) if p[3] == kind][0]
        add_hcbar(fig, [axes[0, col]], last_im[kind], label)
        if kind == "var":
            last_im[kind].colorbar.ax.ticklabel_format(style="sci", scilimits=(-2, 2))

    out_dir = Path(args.out_dir) / group
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{subject_dir.name}_{mod}_runs.png", dpi=args.dpi,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return row


def main():
    p = argparse.ArgumentParser(description="Mean / variation of multiple imputation runs vs ground truth (BraTS2023 PNGs)")
    p.add_argument("data_root")
    p.add_argument("out_dir")
    p.add_argument("--runs-dir", default="imputed", help="Folder inside each modality folder holding the runs")
    p.add_argument("--var", choices=list(VAR_TITLES), default="std", help="Variation map across runs")
    p.add_argument("--planes", nargs="+", default=["sagittal", "coronal", "axial"],
                   choices=list(PLANE_AXIS))
    p.add_argument("--groups", nargs="+", help="Only these group folders (e.g. 51_OtherNeoplasms)")
    p.add_argument("--subjects", nargs="+", help="Only these subject folder names")
    p.add_argument("--modalities", nargs="+", default=list(MODALITY_NAMES), choices=list(MODALITY_NAMES))
    p.add_argument("--limit", type=int, help="Process at most N subjects")
    p.add_argument("--err-max", type=float, default=0.5, help="Color-scale max for the error map")
    p.add_argument("--var-max", type=float, help="Color-scale max for the variation map (default: 99th percentile of brain voxels)")
    p.add_argument("--ssim", action="store_true", help="Also compute SSIM of the mean (needs scikit-image)")
    p.add_argument("--save-nifti", action="store_true", help="Also save mean and variation volumes as NIfTI")
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

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    cases = discover_subjects(args.data_root, args.runs_dir, args.groups, args.subjects, args.limit)
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
                print(f"[{i}/{len(cases)}] {group}/{sdir.name} {mod} done ({res['n_runs']} runs)")

    if not rows:
        print("Nothing written - no modality had the required original/run files.")
        return

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(Path(args.out_dir) / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} PNG(s) + metrics.csv to {args.out_dir}")
    for k in ("MAE_mean", "MAE_run_avg", f"{args.var}_mean"):
        by_mod = defaultdict(list)
        for r in rows:
            by_mod[r["modality"]].append(r[k])
        print("  " + k + ": " + ", ".join(f"{m}={np.mean(v):.4f} (n={len(v)})" for m, v in by_mod.items()))


if __name__ == "__main__":
    main()
