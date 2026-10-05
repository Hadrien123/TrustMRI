#!/usr/bin/env python3
"""Preprocess BraTS-Africa for modality imputation + segmentation.

BraTS-Africa (BraTS 2023 format: <case_id>-{t1n,t1c,t2w,t2f,seg}.nii or .nii.gz) is already
co-registered to SRI24, resampled to 1 mm (240x240x155) and skull-stripped, so we do
NOT register or skull-strip again (it would only add interpolation blur).

Per case:
  1. Sanity check: all modalities + seg share shape and affine; orientation is logged.
  2. Brain mask: union of nonzero voxels across the 4 modalities (saved, for PSNR/SSIM).
  3. Normalization per modality, brain voxels only: clip to the 0.5-99.5 percentiles,
     rescale to [0, 1]. For a tanh generator use x * 2 - 1 at load time.

Output mirrors the input layout, same grid and affine as the originals:
    <out>/<class>/<case_id>/<case_id>-{t1n,t1c,t2w,t2f}.nii.gz   float32 in [0, 1]
    <out>/<class>/<case_id>/<case_id>-mask.nii.gz                uint8
    <out>/<class>/<case_id>/<case_id>-seg.nii.gz                 uint8 (if present)

Usage:
    python preprocessing/preprocess.py --limit 3     # quick check, then:
    python preprocessing/preprocess.py --workers 8
"""
import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = Path.home() / "software/BraTs Africa/BraTS-Africa Dataset"
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def find_nifti(case_dir, case_id, suffix):
    """Path to <case_id>-<suffix>.nii or .nii.gz, or None if neither exists."""
    return next((p for ext in (".nii", ".nii.gz") if (p := case_dir / f"{case_id}-{suffix}{ext}").exists()), None)


def normalize(vol, mask):
    lo, hi = np.percentile(vol[mask], [0.5, 99.5])
    return (np.clip((vol - lo) / max(hi - lo, 1e-8), 0, 1) * mask).astype(np.float32)


def process_case(job):
    row, root, out_root = job
    case_id = row["case_id"]
    case_dir = root / row["class"] / case_id
    log = {"case_id": case_id, "status": "ok", "axcodes": "", "has_seg": False}
    try:
        paths = [find_nifti(case_dir, case_id, m) for m in MODALITIES]
        if None in paths:
            raise FileNotFoundError(f"missing {[m for m, p in zip(MODALITIES, paths) if p is None]}")
        seg_path = find_nifti(case_dir, case_id, "seg")
        log["has_seg"] = seg_path is not None
        imgs = [nib.load(p) for p in paths + ([seg_path] if seg_path else [])]

        # 1. sanity check
        ref = imgs[0]
        if any(i.shape != ref.shape or not np.allclose(i.affine, ref.affine, atol=1e-3) for i in imgs[1:]):
            raise ValueError("shape/affine mismatch between files")
        log["axcodes"] = "".join(nib.aff2axcodes(ref.affine))

        # 2. brain mask, 3. normalization
        vols = [i.get_fdata(dtype=np.float32) for i in imgs[:4]]
        mask = np.any([v != 0 for v in vols], axis=0)

        out_dir = out_root / row["class"] / case_id
        out_dir.mkdir(parents=True, exist_ok=True)
        save = lambda arr, suffix: nib.save(nib.Nifti1Image(arr, ref.affine), out_dir / f"{case_id}-{suffix}.nii.gz")
        for m, v in zip(MODALITIES, vols):
            save(normalize(v, mask), m)
        save(mask.astype(np.uint8), "mask")
        if seg_path:
            save(np.asarray(imgs[4].dataobj, np.uint8), "seg")
    except Exception as e:  # log and keep going
        log["status"] = f"error: {e}"
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DATA_DIR / "BraTS-Africa")
    ap.add_argument("--split-csv", type=Path, default=REPO / "data/splits/brats_africa_split.csv")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "BraTS-Africa_preprocessed")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.split_csv) as f:
        rows = list(csv.DictReader(f))[: args.limit]

    results = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_case, [(r, args.root, args.out) for r in rows]), 1):
            results.append(res)
            print(f"[{i}/{len(rows)}] {res['case_id']}: {res['status']}")

    results.sort(key=lambda r: r["case_id"])
    with open(args.out / "preprocess_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n{len(ok)}/{len(results)} cases ok -> {args.out}")
    if len({r["axcodes"] for r in ok}) > 1:
        print("WARNING: inconsistent orientations:", sorted({r["axcodes"] for r in ok}))


if __name__ == "__main__":
    main()
