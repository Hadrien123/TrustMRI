#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from skimage.metrics import structural_similarity
from scipy.stats import wilcoxon


def load_nii(path):
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img


def calc_metrics(real, pred, brain_mask):
    m = brain_mask.astype(bool)

    r = real[m]
    p = pred[m]

    if r.size == 0:
        raise ValueError("Empty brain mask")

    mae = float(np.mean(np.abs(r - p)))
    rmse = float(np.sqrt(np.mean((r - p) ** 2)))

    data_range = float(r.max() - r.min())

    if data_range <= 0:
        psnr = np.nan
        nrmse = np.nan
    else:
        mse = rmse ** 2
        psnr = float(
            20 * np.log10(data_range) - 10 * np.log10(mse)
        ) if mse > 0 else np.inf
        nrmse = float(rmse / data_range)

    if np.std(r) > 0 and np.std(p) > 0:
        corr = float(np.corrcoef(r, p)[0, 1])
    else:
        corr = np.nan

    # Crop to bounding box of the brain before SSIM.
    coords = np.argwhere(m)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1

    slices = tuple(slice(lo[d], hi[d]) for d in range(3))
    rc = real[slices]
    pc = pred[slices]

    # Use ground-truth image dynamic range.
    if data_range > 0 and min(rc.shape) >= 7:
        ssim = float(
            structural_similarity(
                rc,
                pc,
                data_range=data_range,
                win_size=7
            )
        )
    else:
        ssim = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "PSNR": psnr,
        "SSIM": ssim,
        "CORR": corr,
    }


def find_cases(root):
    cases = {}

    # Exact hierarchy:
    # root/class/subject/modality/original/file.nii*
    for real_path in root.glob("*/*/*/original/*.nii*"):
        modality_dir = real_path.parent.parent
        modality = modality_dir.name
        subject = modality_dir.parent.name
        group = modality_dir.parent.parent.name

        imputed = list(
            (modality_dir / "imputed").glob("*-run01.nii*")
        )

        if len(imputed) != 1:
            continue

        key = (group, subject, modality)
        cases[key] = {
            "real": real_path,
            "pred": imputed[0],
        }

    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pre_root = Path(args.pretrained).resolve()
    ft_root = Path(args.finetuned).resolve()
    out = Path(args.out).resolve()

    out.mkdir(parents=True, exist_ok=True)

    pre_cases = find_cases(pre_root)
    ft_cases = find_cases(ft_root)

    common = sorted(set(pre_cases) & set(ft_cases))

    print(f"Pretrained cases: {len(pre_cases)}")
    print(f"Fine-tuned cases: {len(ft_cases)}")
    print(f"Matched comparisons: {len(common)}")

    only_pre = set(pre_cases) - set(ft_cases)
    only_ft = set(ft_cases) - set(pre_cases)

    if only_pre:
        print(f"WARNING: {len(only_pre)} exist only in pretrained output")
    if only_ft:
        print(f"WARNING: {len(only_ft)} exist only in fine-tuned output")

    rows = []
    errors = []

    for i, key in enumerate(common, 1):
        group, subject, modality = key

        pre_real_path = pre_cases[key]["real"]
        pre_pred_path = pre_cases[key]["pred"]
        ft_pred_path = ft_cases[key]["pred"]

        try:
            real, real_img = load_nii(pre_real_path)
            pre, pre_img = load_nii(pre_pred_path)
            ft, ft_img = load_nii(ft_pred_path)

            if real.shape != pre.shape or real.shape != ft.shape:
                raise ValueError(
                    f"Shape mismatch: real={real.shape}, "
                    f"pre={pre.shape}, ft={ft.shape}"
                )

            # Verify geometry as well as matrix size.
            affine_pre_ok = np.allclose(
                real_img.affine, pre_img.affine, atol=1e-4
            )
            affine_ft_ok = np.allclose(
                real_img.affine, ft_img.affine, atol=1e-4
            )

            # Your original symlink resolves into
            # BraTS-Africa_preprocessed/<class>/<subject>/...
            resolved_real = pre_real_path.resolve()
            subj_dir = resolved_real.parent

            mask_path = subj_dir / f"{subject}-mask.nii.gz"

            if mask_path.exists():
                brain, _ = load_nii(mask_path)
                brain_mask = brain > 0
            else:
                # Fallback if evaluating BraTS 2023 later.
                brain_mask = real != 0

            pre_m = calc_metrics(real, pre, brain_mask)
            ft_m = calc_metrics(real, ft, brain_mask)

            row = {
                "group": group,
                "subject": subject,
                "modality": modality,
                "affine_pretrained_ok": affine_pre_ok,
                "affine_finetuned_ok": affine_ft_ok,
            }

            for name, value in pre_m.items():
                row[f"pretrained_{name}"] = value

            for name, value in ft_m.items():
                row[f"finetuned_{name}"] = value

            # Positive delta = improvement for these fields.
            row["delta_SSIM"] = ft_m["SSIM"] - pre_m["SSIM"]
            row["delta_PSNR"] = ft_m["PSNR"] - pre_m["PSNR"]
            row["delta_CORR"] = ft_m["CORR"] - pre_m["CORR"]

            # Positive = improvement here too, because lower error is better.
            row["improvement_MAE"] = pre_m["MAE"] - ft_m["MAE"]
            row["improvement_RMSE"] = pre_m["RMSE"] - ft_m["RMSE"]
            row["improvement_NRMSE"] = pre_m["NRMSE"] - ft_m["NRMSE"]

            rows.append(row)

        except Exception as e:
            errors.append((group, subject, modality, str(e)))

        if i % 25 == 0 or i == len(common):
            print(f"Processed {i}/{len(common)}")

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No matched images could be evaluated.")

    per_case = out / "per_case_metrics.csv"
    df.to_csv(per_case, index=False)

    metric_pairs = [
        ("SSIM", True),
        ("PSNR", True),
        ("CORR", True),
        ("MAE", False),
        ("RMSE", False),
        ("NRMSE", False),
    ]

    summary_rows = []

    def summarise(sub, label, modality="ALL"):
        result = {
            "group": label,
            "modality": modality,
            "n": len(sub),
        }

        for metric, higher_better in metric_pairs:
            a = sub[f"pretrained_{metric}"].to_numpy(float)
            b = sub[f"finetuned_{metric}"].to_numpy(float)

            valid = np.isfinite(a) & np.isfinite(b)
            a = a[valid]
            b = b[valid]

            result[f"pretrained_{metric}_mean"] = np.mean(a)
            result[f"finetuned_{metric}_mean"] = np.mean(b)

            if higher_better:
                improvement = b - a
            else:
                improvement = a - b

            result[f"{metric}_improvement_mean"] = np.mean(improvement)
            result[f"{metric}_improvement_median"] = np.median(improvement)
            result[f"{metric}_win_fraction"] = np.mean(improvement > 0)

            try:
                if len(improvement) > 1 and np.any(improvement != 0):
                    result[f"{metric}_wilcoxon_p"] = wilcoxon(
                        improvement
                    ).pvalue
                else:
                    result[f"{metric}_wilcoxon_p"] = np.nan
            except Exception:
                result[f"{metric}_wilcoxon_p"] = np.nan

        return result

    summary_rows.append(summarise(df, "ALL", "ALL"))

    for modality, sub in df.groupby("modality"):
        summary_rows.append(
            summarise(sub, "ALL", modality)
        )

    for group, subg in df.groupby("group"):
        summary_rows.append(
            summarise(subg, group, "ALL")
        )

        for modality, sub in subg.groupby("modality"):
            summary_rows.append(
                summarise(sub, group, modality)
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "summary.csv", index=False)

    # Identify worst regressions after fine-tuning.
    worst = df.sort_values("delta_SSIM").head(20)
    best = df.sort_values("delta_SSIM", ascending=False).head(20)

    worst.to_csv(out / "worst_20_ssim_regressions.csv", index=False)
    best.to_csv(out / "best_20_ssim_improvements.csv", index=False)

    report = out / "REPORT.txt"

    with report.open("w") as f:
        f.write("BRA SYNTHESIS MODEL COMPARISON\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Pretrained results: {pre_root}\n")
        f.write(f"Fine-tuned results: {ft_root}\n\n")

        f.write(f"Matched subject/modality pairs: {len(df)}\n")
        f.write(
            f"Unique subjects: {df['subject'].nunique()}\n"
        )
        f.write(
            f"Modalities: {', '.join(sorted(df['modality'].unique()))}\n\n"
        )

        bad_pre_affine = int((~df["affine_pretrained_ok"]).sum())
        bad_ft_affine = int((~df["affine_finetuned_ok"]).sum())

        f.write("GEOMETRY CHECK\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Pretrained affine mismatches: {bad_pre_affine}\n"
        )
        f.write(
            f"Fine-tuned affine mismatches: {bad_ft_affine}\n\n"
        )

        overall = summary[
            (summary["group"] == "ALL")
            & (summary["modality"] == "ALL")
        ].iloc[0]

        f.write("OVERALL MEAN PERFORMANCE\n")
        f.write("-" * 70 + "\n")

        for metric, higher in metric_pairs:
            old = overall[f"pretrained_{metric}_mean"]
            new = overall[f"finetuned_{metric}_mean"]
            imp = overall[f"{metric}_improvement_mean"]
            win = overall[f"{metric}_win_fraction"]
            p = overall[f"{metric}_wilcoxon_p"]

            direction = "higher is better" if higher else "lower is better"

            f.write(
                f"{metric:6s} ({direction})\n"
                f"  pretrained mean : {old:.6f}\n"
                f"  fine-tuned mean : {new:.6f}\n"
                f"  mean improvement: {imp:+.6f}\n"
                f"  cases improved  : {100*win:.1f}%\n"
                f"  paired Wilcoxon p: {p:.4g}\n\n"
            )

        f.write("\nPER MODALITY\n")
        f.write("-" * 70 + "\n")

        mods = summary[
            (summary["group"] == "ALL")
            & (summary["modality"] != "ALL")
        ]

        for _, r in mods.iterrows():
            f.write(f"\n{r['modality']}  n={int(r['n'])}\n")

            for metric in ["SSIM", "PSNR", "MAE"]:
                f.write(
                    f"  {metric}: "
                    f"{r[f'pretrained_{metric}_mean']:.6f}"
                    f" -> "
                    f"{r[f'finetuned_{metric}_mean']:.6f}"
                    f"   improved in "
                    f"{100*r[f'{metric}_win_fraction']:.1f}%"
                    f" of cases\n"
                )

        f.write("\n\nWORST 20 SSIM REGRESSIONS\n")
        f.write("-" * 70 + "\n")
        f.write(
            worst[
                [
                    "group",
                    "subject",
                    "modality",
                    "pretrained_SSIM",
                    "finetuned_SSIM",
                    "delta_SSIM",
                    "pretrained_PSNR",
                    "finetuned_PSNR",
                ]
            ].to_string(index=False)
        )
        f.write("\n")

        if errors:
            f.write("\n\nERRORS\n")
            f.write("-" * 70 + "\n")
            for e in errors:
                f.write(str(e) + "\n")

    print()
    print("=" * 70)
    print(f"Saved per-case metrics: {per_case}")
    print(f"Saved summary:          {out / 'summary.csv'}")
    print(f"Saved report:           {report}")
    print("=" * 70)


if __name__ == "__main__":
    main()
