#!/usr/bin/env python3
"""Run MedSAM2 (3D, box-prompted) on preprocessed BraTS-Africa cases.

Input: the NIfTI files from preprocessing/preprocess.py (240x240x155, [0, 1], brain-masked).
Each of the 155 axial 240x240 slices is resized to 512x512 for the model (MedSAM2's
sam2.1_hiera_t512 config); masks come back at 240x240, aligned with the original scans.

MedSAM2 needs a box prompt on one key slice and propagates it through the volume.
The box is taken from the ground-truth mask on the slice with the largest tumor area
(as in MedSAM2's own lesion benchmarks), so Dice is an upper bound for a box-prompted
model, not fully automatic segmentation.

Run on the GPU machine, with the medsam2 env active:
    python segmentation/run_medsam2_brats_africa.py --split test --target wt --limit 2
"""
import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor_npz

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path.home() / "software/BraTs Africa/BraTS-Africa Dataset/BraTS-Africa_preprocessed"
DEFAULT_CKPT = Path.home() / "software/MedSAM2/checkpoints/MedSAM2_latest.pt"
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]

# BraTS labels: 1 = necrotic core, 2 = edema, 3 = enhancing tumor
TARGETS = {
    "wt": ([1, 2, 3], "t2f"),  # whole tumor, best seen on FLAIR
    "tc": ([1, 3], "t1c"),     # tumor core
    "et": ([3], "t1c"),        # enhancing tumor
}
IMAGE_SIZE = 512
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)

torch.manual_seed(2024)


def to_model_input(vol):
    """(D, H, W) in [0, 1] -> (D, 3, 512, 512) ImageNet-normalized cuda tensor, as in medsam2_infer_3D_CT.py."""
    slices = [
        np.array(Image.fromarray((s * 255).astype(np.uint8)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
        for s in vol
    ]
    x = torch.from_numpy(np.stack(slices)).permute(0, 3, 1, 2).float().cuda() / 255.0
    mean = torch.tensor(IMG_MEAN).view(3, 1, 1).cuda()
    std = torch.tensor(IMG_STD).view(3, 1, 1).cuda()
    return (x - mean) / std


def box_prompt(gt):
    """Key slice = largest GT area; box = [x0, y0, x1, y1] in slice pixel coords."""
    key = int(gt.sum(axis=(1, 2)).argmax())
    ys, xs = np.nonzero(gt[key])
    return key, np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


@torch.inference_mode()
def segment(predictor, vol, key, box):
    d, h, w = vol.shape
    seg = np.zeros((d, h, w), dtype=np.uint8)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(to_model_input(vol), h, w)
        for reverse in (False, True):
            predictor.add_new_points_or_box(inference_state=state, frame_idx=key, obj_id=1, box=box)
            for idx, _, logits in predictor.propagate_in_video(state, reverse=reverse):
                seg[idx, (logits[0] > 0.0).cpu().numpy()[0]] = 1
            predictor.reset_state(state)
    return seg


def dice(a, b):
    denom = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / denom if denom else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="preprocessed NIfTI folder")
    ap.add_argument("--split-csv", type=Path, default=REPO / "data/splits/brats_africa_split.csv")
    ap.add_argument("--split", choices=["train", "test", "all"], default="test")
    ap.add_argument("--target", choices=TARGETS, default="wt")
    ap.add_argument("--modality", choices=MODALITIES, help="default depends on --target")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--cfg", default="configs/sam2.1_hiera_t512.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "outputs/medsam2")
    ap.add_argument("--limit", type=int, help="only run the first N cases (for a quick test)")
    args = ap.parse_args()

    labels, default_mod = TARGETS[args.target]
    modality = args.modality or default_mod
    out = args.out / f"{args.split}_{args.target}_{modality}"
    out.mkdir(parents=True, exist_ok=True)

    with open(args.split_csv) as f:
        cases = [r for r in csv.DictReader(f) if args.split in ("all", r["split"])][: args.limit]

    predictor = build_sam2_video_predictor_npz(args.cfg, str(args.checkpoint))

    results = []
    for i, row in enumerate(cases, 1):
        case_dir = args.data / row["class"] / row["case_id"]
        seg_path = case_dir / f"{row['case_id']}-seg.nii.gz"
        if not seg_path.exists():
            print(f"[{i}/{len(cases)}] {row['case_id']}: no seg file (needed for the box prompt), skipped")
            continue
        img = nib.load(case_dir / f"{row['case_id']}-{modality}.nii.gz")
        # (x, y, z) -> (z, x, y): one axial slice per "video frame"
        vol = np.moveaxis(img.get_fdata(dtype=np.float32), 2, 0)
        gt = np.moveaxis(np.isin(np.asarray(nib.load(seg_path).dataobj), labels), 2, 0)
        if not gt.any():
            print(f"[{i}/{len(cases)}] {row['case_id']}: no '{args.target}' voxels in GT, skipped")
            continue

        seg = segment(predictor, vol, *box_prompt(gt))
        nib.save(nib.Nifti1Image(np.moveaxis(seg, 0, 2), img.affine), out / f"{row['case_id']}.nii.gz")

        d = dice(seg.astype(bool), gt)
        results.append({"case_id": row["case_id"], "class": row["class"], "split": row["split"], "dice": round(d, 4)})
        print(f"[{i}/{len(cases)}] {row['case_id']} ({row['class']}): dice={d:.4f}")

    with open(out / "dice.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "class", "split", "dice"])
        writer.writeheader()
        writer.writerows(results)
    for cls in sorted({r["class"] for r in results}):
        scores = [r["dice"] for r in results if r["class"] == cls]
        print(f"{cls}: mean dice {np.mean(scores):.4f} over {len(scores)} cases")
    print(f"Masks and dice.csv written to {out}")


if __name__ == "__main__":
    main()
