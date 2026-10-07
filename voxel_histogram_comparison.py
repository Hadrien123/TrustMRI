#!/usr/bin/env python3

from pathlib import Path
import argparse
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon


MODALITIES = ["t1c", "t1n", "t2f", "t2w"]


def load(path):
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def brasyn_normalize(x):
    """
    Same per-volume min-max normalization used by BraSyn.
    """
    lo = float(np.min(x))
    hi = float(np.max(x))

    if hi <= lo:
        raise ValueError("Zero intensity range")

    return (x - lo) / (hi - lo)


def collect_cases(root):
    """
    root/class/subject/modality/
        original/*.nii*
        imputed/*-run01.nii*
    """

    root = Path(root)

    cases = {}

    for real_path in sorted(root.glob("*/*/*/original/*.nii*")):

        modality_dir = real_path.parent.parent
        modality = modality_dir.name
        subject = modality_dir.parent.name
        group = modality_dir.parent.parent.name

        if modality not in MODALITIES:
            continue

        preds = sorted(
            (modality_dir / "imputed").glob("*-run01.nii*")
        )

        if len(preds) != 1:
            continue

        cases[(group, subject, modality)] = {
            "real": real_path,
            "pred": preds[0],
        }

    return cases


def brain_mask(real_path, subject, real_raw):

    resolved = real_path.resolve()

    explicit = resolved.parent / f"{subject}-mask.nii.gz"

    if explicit.exists():
        mask = load(explicit) > 0
    else:
        mask = real_raw > 0

    # Critical:
    # remove zero-valued background even if an external mask contains it.
    mask = mask & (real_raw > 0)

    return mask


def probability_hist(values, edges):

    values = values[np.isfinite(values)]

    # Do not silently clip bad model values.
    inside = (
        (values >= edges[0]) &
        (values <= edges[-1])
    )

    excluded_fraction = 1.0 - np.mean(inside)

    values = values[inside]

    counts, _ = np.histogram(values, bins=edges)

    counts = counts.astype(np.float64)

    if counts.sum() > 0:
        prob = counts / counts.sum()
    else:
        prob = counts

    return prob, excluded_fraction


def distribution_metrics(gt, pred, centers):

    eps = 1e-12

    gt = gt + eps
    pred = pred + eps

    gt = gt / gt.sum()
    pred = pred / pred.sum()

    js = float(jensenshannon(gt, pred) ** 2)

    wass = float(
        wasserstein_distance(
            centers,
            centers,
            u_weights=gt,
            v_weights=pred,
        )
    )

    l1 = float(np.sum(np.abs(gt - pred)))

    intersection = float(np.sum(np.minimum(gt, pred)))

    return {
        "JSD": js,
        "Wasserstein": wass,
        "Histogram_L1": l1,
        "Histogram_intersection": intersection,
    }


def compare_dataset(
    pretrained_root,
    finetuned_root,
    dataset_name,
    out,
    bins=200,
    limit_subjects=None,
):

    pre = collect_cases(pretrained_root)
    ft = collect_cases(finetuned_root)

    common = sorted(set(pre) & set(ft))

    subjects = sorted(
        set(k[1] for k in common)
    )

    if limit_subjects:
        subjects = subjects[:limit_subjects]

    subject_set = set(subjects)

    common = [
        k for k in common
        if k[1] in subject_set
    ]

    print()
    print("=" * 70)
    print(dataset_name)
    print("=" * 70)
    print("Matched subjects:", len(subjects))
    print("Matched subject/modality pairs:", len(common))

    edges = np.linspace(0, 1, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    summary = []

    for modality in MODALITIES:

        keys = [
            k for k in common
            if k[2] == modality
        ]

        gt_subject_hists = []
        pre_subject_hists = []
        ft_subject_hists = []

        pre_outside = []
        ft_outside = []

        for key in keys:

            group, subject, modality = key

            real_raw = load(pre[key]["real"])
            pre_img = load(pre[key]["pred"])
            ft_img = load(ft[key]["pred"])

            if not (
                real_raw.shape ==
                pre_img.shape ==
                ft_img.shape
            ):
                print(
                    "Shape mismatch:",
                    subject,
                    modality,
                )
                continue

            mask = brain_mask(
                pre[key]["real"],
                subject,
                real_raw,
            )

            # Put GT into BraSyn's model intensity space.
            gt_img = brasyn_normalize(real_raw)

            gt_values = gt_img[mask]
            pre_values = pre_img[mask]
            ft_values = ft_img[mask]

            gt_hist, _ = probability_hist(
                gt_values,
                edges,
            )

            pre_hist, pre_bad = probability_hist(
                pre_values,
                edges,
            )

            ft_hist, ft_bad = probability_hist(
                ft_values,
                edges,
            )

            gt_subject_hists.append(gt_hist)
            pre_subject_hists.append(pre_hist)
            ft_subject_hists.append(ft_hist)

            pre_outside.append(pre_bad)
            ft_outside.append(ft_bad)

        if not gt_subject_hists:
            continue

        # IMPORTANT:
        # Equal subject weighting.
        #
        # We average per-subject probability distributions instead of
        # pooling every voxel from everybody.
        #
        # This prevents subjects with more brain voxels from dominating.
        gt = np.mean(gt_subject_hists, axis=0)
        pre_h = np.mean(pre_subject_hists, axis=0)
        ft_h = np.mean(ft_subject_hists, axis=0)

        # Smooth only for visualization.
        # Metrics use unsmoothed distributions.
        gt_plot = gaussian_filter1d(gt, sigma=1.5)
        pre_plot = gaussian_filter1d(pre_h, sigma=1.5)
        ft_plot = gaussian_filter1d(ft_h, sigma=1.5)

        pre_metrics = distribution_metrics(
            gt,
            pre_h,
            centers,
        )

        ft_metrics = distribution_metrics(
            gt,
            ft_h,
            centers,
        )

        row = {
            "dataset": dataset_name,
            "modality": modality,
            "subjects": len(gt_subject_hists),
            "pretrained_outside_0_1":
                np.mean(pre_outside),
            "finetuned_outside_0_1":
                np.mean(ft_outside),
        }

        for k, v in pre_metrics.items():
            row[f"pretrained_{k}"] = v

        for k, v in ft_metrics.items():
            row[f"finetuned_{k}"] = v

        # For JSD/Wasserstein/L1, lower is better.
        row["JSD_improvement"] = (
            pre_metrics["JSD"] -
            ft_metrics["JSD"]
        )

        row["Wasserstein_improvement"] = (
            pre_metrics["Wasserstein"] -
            ft_metrics["Wasserstein"]
        )

        row["Histogram_L1_improvement"] = (
            pre_metrics["Histogram_L1"] -
            ft_metrics["Histogram_L1"]
        )

        # For histogram intersection, higher is better.
        row["Intersection_improvement"] = (
            ft_metrics["Histogram_intersection"] -
            pre_metrics["Histogram_intersection"]
        )

        summary.append(row)

        plt.figure(figsize=(11, 6))

        plt.plot(
            centers,
            gt_plot,
            linewidth=2,
            label="Ground truth",
        )

        plt.plot(
            centers,
            pre_plot,
            linewidth=2,
            label="Pretrained",
        )

        plt.plot(
            centers,
            ft_plot,
            linewidth=2,
            label="Fine-tuned",
        )

        plt.xlabel("BraSyn-normalized voxel intensity")
        plt.ylabel("Mean fraction of brain voxels")

        plt.xlim(0, 1)

        plt.title(
            f"{dataset_name} - {modality}\n"
            f"{len(gt_subject_hists)} matched subjects"
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            out / f"{dataset_name}_{modality}.png",
            dpi=200,
        )

        plt.close()

        pd.DataFrame({
            "intensity": centers,
            "ground_truth": gt,
            "pretrained": pre_h,
            "finetuned": ft_h,
        }).to_csv(
            out / f"{dataset_name}_{modality}.csv",
            index=False,
        )

        print()
        print(modality)

        print(
            "  JSD:"
            f" {pre_metrics['JSD']:.6f}"
            " ->"
            f" {ft_metrics['JSD']:.6f}"
        )

        print(
            "  Wasserstein:"
            f" {pre_metrics['Wasserstein']:.6f}"
            " ->"
            f" {ft_metrics['Wasserstein']:.6f}"
        )

        print(
            "  Histogram intersection:"
            f" {pre_metrics['Histogram_intersection']:.4f}"
            " ->"
            f" {ft_metrics['Histogram_intersection']:.4f}"
        )

    return summary


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--africa-pretrained",
        required=True,
    )

    ap.add_argument(
        "--africa-finetuned",
        required=True,
    )

    ap.add_argument(
        "--brats-pretrained",
        required=True,
    )

    ap.add_argument(
        "--brats-finetuned",
        required=True,
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    ap.add_argument(
        "--limit-subjects",
        type=int,
        default=None,
    )

    args = ap.parse_args()

    out = Path(args.out).resolve()

    # User requested overwrite instead of creating new folders.
    if out.exists():
        shutil.rmtree(out)

    out.mkdir(parents=True)

    summary = []

    summary += compare_dataset(
        args.africa_pretrained,
        args.africa_finetuned,
        "Africa",
        out,
        limit_subjects=args.limit_subjects,
    )

    summary += compare_dataset(
        args.brats_pretrained,
        args.brats_finetuned,
        "BraTS2023",
        out,
        limit_subjects=args.limit_subjects,
    )

    df = pd.DataFrame(summary)

    df.to_csv(
        out / "distribution_comparison_summary.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    cols = [
        "dataset",
        "modality",
        "pretrained_JSD",
        "finetuned_JSD",
        "JSD_improvement",
        "pretrained_Wasserstein",
        "finetuned_Wasserstein",
        "Wasserstein_improvement",
        "pretrained_Histogram_intersection",
        "finetuned_Histogram_intersection",
        "Intersection_improvement",
    ]

    print(
        df[cols].to_string(
            index=False,
        )
    )

    print()
    print("Positive improvement = fine-tuning moved closer to GT.")
    print()
    print("Files written to:")
    print(out)


if __name__ == "__main__":
    main()
