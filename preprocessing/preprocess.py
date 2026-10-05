#!/usr/bin/env python3
"""Preprocess BraTS-Africa for modality imputation + segmentation.

BraTS-Africa (BraTS 2023 format: <case_id>-{t1n,t1c,t2w,t2f,seg}.nii.gz) is already
co-registered to SRI24, resampled to 1 mm (240x240x155) and skull-stripped, so we do
NOT register or skull-strip again (it would only add interpolation blur).

Per case:
  1. Sanity check: all modalities + seg share shape and affine; orientation is logged.
  2. Brain mask: union of nonzero voxels across the 4 modalities (saved, for PSNR/SSIM).
  3. Normalization per modality, brain voxels only: clip to the 0.5-99.5 percentiles,
     rescale to [0, 1]. For a tanh generator use x * 2 - 1 at load time.
  4. Crop/pad to 192x192x160 (x, y, z): x/y cropped from 240 around the brain's bounding
     box (only background is removed), z zero-padded from 155 so no slice is lost.
     Every axis is divisible by 32, so it fits U-Nets with up to 5 downsamplings.
     Models that need another size (e.g. MedSAM2's 512x512 slices) resize at inference.
  5. Cache one compressed .npz per case:
       image (4, 192, 192, 160) float32 in MODALITIES order, mask and seg (192, 192, 160),
       affine (of the cropped volume, so outputs can be saved as NIfTI directly),
       crop_offsets (to map back to the original 240x240x155 grid).

Splits are subject level, from data/splits/brats_africa_split.csv.
At training time: modality dropout; spatial augmentation on inputs and target alike;
intensity augmentation on inputs only.

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
CROP_SHAPE = (192, 192, 160)  # (x, y, z) in nibabel order


def normalize(vol, mask):
    lo, hi = np.percentile(vol[mask], [0.5, 99.5])
    return (np.clip((vol - lo) / max(hi - lo, 1e-8), 0, 1) * mask).astype(np.float32)


def brain_offsets(mask):
    """Per axis: center the crop on the brain bounding box (clamped to the volume), or center-pad."""
    offsets = []
    for axis, (s, t) in enumerate(zip(mask.shape, CROP_SHAPE)):
        if s <= t:
            offsets.append((s - t) // 2)
            continue
        idx = np.nonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
        center = (idx[0] + idx[-1]) // 2
        offsets.append(int(np.clip(center - t // 2, 0, s - t)))
    return np.array(offsets)


def crop_or_pad(arr, offsets):
    """Fit the last 3 dims to CROP_SHAPE; offset > 0 crops, < 0 zero-pads."""
    out = np.zeros(arr.shape[:-3] + CROP_SHAPE, arr.dtype)
    sizes = [min(s, t) for s, t in zip(arr.shape[-3:], CROP_SHAPE)]
    src = tuple(slice(max(o, 0), max(o, 0) + n) for o, n in zip(offsets, sizes))
    dst = tuple(slice(max(-o, 0), max(-o, 0) + n) for o, n in zip(offsets, sizes))
    out[(...,) + dst] = arr[(...,) + src]
    return out


def process_case(job):
    row, root, out_dir = job
    case_id = row["case_id"]
    case_dir = root / row["class"] / case_id
    log = {"case_id": case_id, "status": "ok", "axcodes": "", "has_seg": False, "brain_voxels_cropped_out": ""}
    try:
        seg_path = case_dir / f"{case_id}-seg.nii.gz"
        log["has_seg"] = seg_path.exists()
        imgs = [nib.load(case_dir / f"{case_id}-{m}.nii.gz") for m in MODALITIES]
        imgs += [nib.load(seg_path)] if log["has_seg"] else []

        ref = imgs[0]
        if any(i.shape != ref.shape or not np.allclose(i.affine, ref.affine, atol=1e-3) for i in imgs[1:]):
            raise ValueError("shape/affine mismatch between files")
        log["axcodes"] = "".join(nib.aff2axcodes(ref.affine))

        vols = np.stack([i.get_fdata(dtype=np.float32) for i in imgs[:4]])
        seg = np.asarray(imgs[4].dataobj, np.uint8) if log["has_seg"] else np.zeros(ref.shape, np.uint8)
        mask = (vols != 0).any(axis=0)
        image = np.stack([normalize(v, mask) for v in vols])

        offsets = brain_offsets(mask)
        mask_c = crop_or_pad(mask, offsets)
        affine = ref.affine.copy()
        affine[:3, 3] += affine[:3, :3] @ offsets
        np.savez_compressed(
            out_dir / f"{case_id}.npz",
            image=crop_or_pad(image, offsets),
            mask=mask_c,
            seg=crop_or_pad(seg, offsets),
            affine=affine,
            crop_offsets=offsets,
        )
        log["brain_voxels_cropped_out"] = int(mask.sum() - mask_c.sum())
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
    if any(r["brain_voxels_cropped_out"] for r in ok):
        print("WARNING: crop removed brain voxels in some cases (see preprocess_log.csv)")


if __name__ == "__main__":
    main()
