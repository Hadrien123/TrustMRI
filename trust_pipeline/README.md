# TRUST-MRI evaluation pipeline

Evaluation and trust analysis for a two-stage pipeline:
1. a stochastic generative model **imputes** a missing MRI sequence (N samples per patient);
2. a promptable segmenter (e.g. MedSAM) **segments** the tumour on each imputed volume with the same K prompts.

This package implements everything after the two models: image and segmentation metrics, uncertainty maps, a
two-way ANOVA separating imputation from prompt variability, and a local (patch-wise) confidence model with
its evaluation. It runs end to end on synthetic data today. For real data, only the loader changes.

## Install and run

```bash
pip install -r requirements.txt

# synthetic cohort, default design: P = 8 patients, 240 x 240 x 155, N = 5 imputations, K = 5 prompts
python -m trust_mri_eval.cli --synthetic

# quick run on small volumes, without NIfTI maps
python -m trust_mri_eval.cli --synthetic --shape 96 96 64 --no-maps --run-id quick

# any Config field can be overridden (values parsed as JSON)
python -m trust_mri_eval.cli --synthetic --set n_patients=10 "split=[6,2,2]" penalty=l1 "patch_size=[3,3,1]"

# real data
python -m trust_mri_eval.cli --data-dir /path/to/patients --regions ET TC WT

pytest            # unit and end-to-end tests (~15 s)
```

All parameters live in [trust_mri_eval/config.py](trust_mri_eval/config.py): design (N, K, P), synthetic
generator, patch sizes, thresholds, tolerances, model settings, seeds and output switches. Runs are deterministic:
each synthetic patient is generated from the seed `(seed, patient index)`.

## Plugging in real data

Only [trust_mri_eval/data/io.py](trust_mri_eval/data/io.py) knows about files. It produces a
`PatientData` ([data/types.py](trust_mri_eval/data/types.py)), and nothing else in the code depends on how the
data were stored. Expected layout, one folder per patient:

```
<data_dir>/<patient_id>/
    imputed_n0.nii.gz ... imputed_n{N-1}.nii.gz     (or one 4D imputed.nii.gz, N on the last axis)
    seg_ET_n{n}_k{k}.nii.gz                          probability maps in [0, 1], same K prompts for every n
    brain_mask.nii.gz                                optional (else derived from the images)
    gt_image.nii.gz                                  optional: real held-out sequence (evaluation only)
    gt_ET.nii.gz  or  gt_mask.nii.gz                 optional: binary mask or BraTS label map
    real_flair.nii.gz                                optional: anatomical-consistency feature
```

BraTS label maps are converted per region (2023: 1 NCR, 2 SNFH, 3 ET; legacy: label 4 = ET; TC = NCR + ET,
WT = all). Each region in `--regions` is evaluated separately, with its own `seg_{region}_*` files, metrics and model.
If a different layout is needed, rewrite `load_patient` so that it returns a `PatientData`.

**If only one segmentation per imputed volume exists**, use K = 1: the ANOVA then reduces to the
imputation factor.

## What is computed

| Step | Module | Needs ground truth? |
|---|---|---|
| Crop to brain, robust rescale to [0, 1] (0.5–99.5th percentile), masks, consensus, ROI | `preprocessing.py` | ROI includes the reference when available |
| **A.** L1, L2, PSNR, SSIM (whole brain / tumour / healthy brain) per sample, mean ± std | `metrics/image_global.py` | yes |
| **B.** Patch maps of local SSIM, L1, L2 and variance; failure map; region summaries | `metrics/image_local.py` | yes, except variance |
| **C.** Mean/variance maps, pairwise SSIM (and its local map), pairwise PSNR; coverage, Spearman(std, error) | `metrics/distribution.py` | only coverage and Spearman |
| **D.** Dice, NSD, HD95, BraTS-2023 lesion-wise Dice/HD95, lesion precision/recall/F1 | `metrics/segmentation.py`, `metrics/lesions.py` | yes |
| Agreement, entropy, signed distance to the consensus boundary | `uncertainty/confidence_maps.py` | no |
| Two-way crossed ANOVA per voxel (inside the ROI) and on lesion volume / count | `uncertainty/anova.py` | no |
| Patch features | `model/features.py` | **no, by construction** |
| Patch labels (correct / incorrect) | `model/labels.py` | yes |
| Logistic confidence model, recalibration, evaluation | `model/logistic.py` | for training/evaluation |

Design choices worth knowing:

- **SSIM** uses the same formula as `skimage.metrics.structural_similarity` (uniform 7³ window, sample
  covariance), and a test checks it against skimage. It is reimplemented in float32 so that each voxel-wise map
  is computed once and averaged in any region or patch. Pairwise SSIM reuses each sample's local statistics.
- **Patch grids** ([patches.py](trust_mri_eval/patches.py)) are non-overlapping and aligned on one global
  lattice. Patch values are means over brain voxels. A patch belongs to the tumour region when the majority
  of its brain voxels are tumour. B runs at 3×3×3 (`p3`) and at the coarse 8×8×8 (`p8`).
- **Thresholds** for the failure map and for the "fraction above threshold" summaries are the 90th percentile
  of brain-patch values over **training patients only** (10th for SSIM, where low is bad).
- **ANOVA**: M(n, k) = μ + a_n + b_k + c_nk, with no replication, so the residual is the interaction. It uses
  unbiased mean-square estimators: σ²_A = (MS_A − MS_AB)/K, σ²_B = (MS_B − MS_AB)/N, σ²_AB = MS_AB.
  Negative estimates are clipped to 0. Fractions are each component over their sum. Voxel fractions are set
  to 0 where the total variance is ≤ 1e-4 (numerical noise far from any boundary). It runs only inside the
  ROI, in chunks of voxels, on probabilities by default (`anova_on="masks"` for binary masks). The report's
  "variance share" is Σ component / Σ total over the ROI.
- **Features never see the ground truth.** `compute_features` takes no reference argument, so inference runs
  the same code as training. Besides the listed features, `seg_abs_distance` (|signed distance|) is added,
  because a linear model cannot learn "close to the boundary on either side" from the signed distance alone.
  Local pairwise Dice is the pooled form Σc(c−1) / ((M−1)Σc), where c is the number of the M = N·K masks
  that are positive at a voxel. Context features are each feature smoothed with σ = 2 and 5 voxels.
- **ROI caveat.** During training and evaluation the ROI also contains the reference, so missed lesions are
  scored. On a new patient it can only be built from the predictions (`compute_roi(masks, None, ...)`).
- **Labels**: a voxel is correct if consensus = reference, or if it lies within 1 voxel of the reference
  boundary. A 3D patch is correct if ≥ 90 % of its voxels are correct; a 3×3×1 patch, if ≥ 8 of 9
  (`label_rule="auto"`).
- **Model**: StandardScaler, then logistic regression (`class_weight="balanced"`, L2 or L1). C is chosen by
  GroupKFold cross-validation grouped by patient. Patients are split into train / calibration / test
  (default 5 / 1 / 2), and Platt (or isotonic) recalibration is fitted on the calibration patients. Three
  variants are compared:
  - **full**: all features;
  - **agreement only**: the baseline;
  - **no imputation**: the ablation, which drops the imputation features and their context versions.
- **Evaluation** (test patients):
  - AUROC for detecting incorrect patches (score = 1 − confidence);
  - reliability diagram and ECE (10 bins);
  - Brier score;
  - risk–coverage curve and AURC;
  - a QU-BraTS-style curve: Dice of the kept voxels vs the fraction filtered as the confidence threshold
    rises, plus filtered-TP/TN ratios and the QU-BraTS score.

## Outputs

`outputs/<run_id>/`:

| File | Content |
|---|---|
| `metrics_per_patient.csv` | One row per patient and region; columns below. |
| `metrics_summary.json` | Configuration, patient split, local thresholds, cohort mean/std/n of every metric. Also, per model variant: C, CV scores, AUROC, ECE, Brier, AURC, QU-BraTS AUCs; full-model coefficients; QU-BraTS curves. NaN/inf are written as `null`. |
| `report.md` | Short human-readable summary of all of the above. |
| `maps/<region>/<patient>/*.nii.gz` | Maps in the original volume geometry (affine of the input). |
| `figures/<region>/<patient>.png` | Axial slice with the largest reference area: image, sample, variance, local SSIM, failure map, consensus vs reference, agreement, 3 ANOVA fractions, confidence overlay, dominant source. |
| `figures/<region>/{reliability,roc,risk_coverage,qubrats_filtering,coefficients}.png` | Model evaluation charts (test patients). |

CSV column prefixes:

| Prefix | Content |
|---|---|
| `img_*` | Module A, `_mean` / `_std` over the N samples. |
| `local_<p3\|p8>_<map>_<region>_*` | Module B summaries: `mean`, `p95` (or `p05` for SSIM), `frac_above_thr` / `frac_below_thr`. Also `local_*_failure_fraction`. |
| `dist_*` | Module C. |
| `seg_*` | Module D, plus `seg_dice_per_mask_*` (Dice of each of the N·K masks). |
| `unc_*` | Entropy and disagreement in the ROI. |
| `anova_*` | Variance shares in the ROI, plus lesion volume / count ANOVA. |
| `model_*` | Patch counts and the fraction of incorrect patches. |
| `conf_*` | Mean confidence, fraction of patches dominated by each feature group, and the per-patient AUROC (test patients). |
| `split` | train / calibration / test. |

Maps in `maps/<region>/<patient>/`:

| Map | Content |
|---|---|
| `imputed_mean`, `imputed_variance`, `pairwise_ssim_voxel` | Distribution of the N samples (module C). |
| `local_{ssim,l1,l2,variance,pairwise_ssim}_{p3,p8}`, `failure_{p3,p8}` | Patch maps (module B), painted back to voxels. |
| `consensus`, `agreement`, `entropy`, `signed_distance` | Confidence maps. |
| `anova_var_{imputation,prompt,interaction,total}`, `anova_frac_*` | Voxel-wise ANOVA. |
| `confidence` | P(segmentation correct) per patch; NaN outside the scored patches. |
| `contribution_{imputation,segmentation,anatomy,context}` | Coefficient × standardised value, summed per group. |
| `dominant_source` | Group with the largest absolute contribution (1 imputation, 2 segmentation, 3 anatomy, 4 context; 0 = not scored). |

## Synthetic data

[data/synthetic.py](trust_mri_eval/data/synthetic.py) generates the synthetic cohort. Each patient has:
- an ellipsoidal brain with grey/white matter, a cortical CSF rim, two ventricles and smooth texture;
- 1–3 ellipsoidal tumours (radius 5–25 voxels) with a bright enhancing rim.

Imputations add three kinds of error to the real image:
- smooth noise, three times stronger near tumour boundaries;
- small elastic warps around the tumours;
- with probability 0.3 per sample, a hallucinated bright blob.

Segmentations threshold each smoothed imputation, then apply a prompt-specific boundary shift (−2…+2 voxels)
and a smooth jitter field fixed per prompt, plus a small (n, k) interaction jitter. A prompt may also ignore a
non-largest lesion. This makes the crossed design non-trivial. Numbers obtained on synthetic data only show
that the pipeline works; they say nothing about real models.

## Layout

```
trust_mri_eval/
  config.py  cli.py  pipeline.py  preprocessing.py  patches.py  report.py
  data/        types.py (PatientData), synthetic.py, io.py
  metrics/     ssim.py, image_global.py, image_local.py, distribution.py, segmentation.py, lesions.py
  uncertainty/ confidence_maps.py, anova.py
  model/       features.py, labels.py, logistic.py
  viz/         plots.py
tests/
```
