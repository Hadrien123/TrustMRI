# TRUST-MRI evaluation pipeline

Evaluation and trust analysis for a two-stage pipeline:
1. a stochastic generative model **imputes** a missing MRI sequence (N samples per patient, 240 x 240 x 155);
2. a promptable segmenter **segments** the tumour on each imputed volume with the same K prompts.

This package implements everything after the two models: image and segmentation metrics, local maps, the
spread of the N samples, a two-way ANOVA (imputation vs prompt) and a patch-wise logistic confidence model
whose output is a confidence heatmap. It reads the files written by the preprocessing, imputation (BraSyn)
and segmentation (MedSAM2) steps; it does not generate any image. It reuses the code of the imputation codebase
([BraSyn_tutorial](https://github.com/WinstonHuTiger/BraSyn_tutorial)) and of the official
[BraTS_evaluation](https://github.com/BraTS/BraTS_evaluation).

## Install and run

Python 3.10–3.13 (the official [BraTS_evaluation](https://github.com/BraTS/BraTS_evaluation), built on
[panoptica](https://github.com/BrainLesion/panoptica), needs `numpy<2.3` and `pandas<3`).

```bash
py -V:3.12 -m venv .venv          # Windows; elsewhere: python3.12 -m venv .venv
.venv/Scripts/activate            # Windows; elsewhere: source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/WinstonHuTiger/BraSyn_tutorial ../external/BraSyn_tutorial   # imputation codebase

python -m trust_mri_eval.cli --data-dir /path/to/cases                      # region WT, imputed modality inferred
python -m trust_mri_eval.cli --data-dir /path/to/cases --set modality=t1c region=TC save_maps=false run_id=t1c
BRASYN_REPO=/other/BraSyn_tutorial python -m trust_mri_eval.cli --data-dir /path/to/cases   # other clone

pytest            # toy cases written to a temporary folder (~10 s)
```

[imputation_eval.ipynb](imputation_eval.ipynb) evaluates one subject step by step with the modules of BraSyn and
BraTS_evaluation: imputed runs vs real, spread of the runs, Dice per (run, prompt) and its ANOVA. Set `IMPUTED_DIR`
(output of `run_imputation.py`) and `GT_DIR`.
All parameters are in [config.py](trust_mri_eval/config.py); any field can be overridden with `--set name=value`.

## Pipeline → code

| Step | What | Code | Ground truth? |
|---|---|---|---|
| Loading | every volume in BraSyn space with BraSyn's code: IPL orientation, 144 x 192 x 192 crop, real image min-max rescaled (`toGrayScale`) | `data/io.py` | no |
| Preprocessing | threshold the segmentations, ROI | `preprocessing.py` | ROI includes the reference when available |
| Imputation, global | L1, L2 in the brain; SSIM (BraSyn `util.ssim`) and PSNR (as in BraSyn `test.py`) over the cropped volume | `pipeline.quality` | yes |
| Imputation, local | patch maps of SSIM, L1, L2 (vs real) and variance (across the N images) | `metrics/image_local.py`, `pipeline.trust` | yes, except variance |
| Distribution of the N images | mean variance (brain), mean pairwise SSIM and PSNR (BraSyn) | `pipeline.trust` | no |
| Segmentation | official BraTS evaluation: DSC, NSD (global), lesion-wise F1 (`evaluate_single_exam`) | `pipeline.quality` | yes |
| Segmentation agreement | fraction of the N x K masks calling each voxel tumour | `pipeline.trust` | no |
| ANOVA | per voxel, variance of the N x K probabilities split into imputation / prompt / interaction | `uncertainty/anova.py` | no |
| Confidence model | features per patch → logistic regression → P(segmentation correct) heatmap | `model/` | labels only |

Quality metrics compare **one** output with the ground truth: imputed image `eval_imputation` and its
segmentation with prompt `eval_prompt`. Trust metrics use all N x K outputs and never the ground truth.

Design choices worth knowing:

- **SSIM**: the reported numbers come from BraSyn's `util.ssim` (Gaussian 11³ window, σ = 1.5, mean over the
  whole cropped volume), on the GPU when there is one. On CPU it takes ~30 s per pair of volumes (dense 3D
  convolutions), so N runs cost N + N(N−1)/2 calls. `util.ssim` only returns the mean, so the local SSIM maps
  come from skimage's `structural_similarity(full=True)` with the same Gaussian window.
- **Patches** are non-overlapping (`patch_size`, default 3 x 3 x 3) on one lattice shared by local maps,
  features and labels. Patch values are means over brain voxels.
- **Segmentation metrics** come from BraTS_evaluation's `evaluate_single_exam` with its `gli` config. Lesion-wise F1 is panoptica's recognition quality, TP / (TP + ½FP + ½FN), with
  lesions = connected components matched one-to-one on Dice. In panoptica 2.1.7 the global NSD ignores the
  voxel spacing (tolerance 0.5 voxel); the official numbers are kept as they are.
- **ANOVA**: M(n, k) = μ + a_n + b_k + c_nk without replication (the residual is the interaction), unbiased
  estimators σ²_A = (MS_A − MS_AB)/K, σ²_B = (MS_B − MS_AB)/N, σ²_AB = MS_AB, negatives clipped to 0. Computed
  inside the ROI; the reported share is Σ component / Σ total over the ROI. With K = 1 only imputation remains.
- **Features** (no ground truth, so inference runs the same code): patch variance and pairwise SSIM of the N
  images, and the pooled pairwise Dice of the N x K masks, Σc(c−1) / ((M−1)Σc) with c the number of the
  M = N·K masks positive at a voxel. Scored patches are those at least half inside the ROI. At inference,
  build the ROI from the predictions only: `compute_roi(masks, None, ...)`.
- **Labels**: a voxel is correct if the evaluated segmentation equals the reference or lies within 1 voxel of
  the reference boundary; a patch is correct if ≥ 90 % of its voxels are.
- **Model**: StandardScaler + logistic regression (`class_weight="balanced"`, L2 or L1), C chosen by
  cross-validation grouped by patient. A fraction of the patients (`test_fraction`, default 0.25, at least one
  when there are two patients or more) is held out; the report gives their AUROC for detecting incorrect patches.

## Data

Only [data/io.py](trust_mri_eval/data/io.py) reads files, through BraSyn's dataloader transform; it returns a `PatientData` ([data/types.py](trust_mri_eval/data/types.py)) in BraSyn space. Files are found
**by name** anywhere under `--data-dir`, so the outputs of the three steps can sit in any sub-folders:

```
<case>-{t1c,t1n,t2f,t2w}.nii.gz          real preprocessed sequences (preprocessing/preprocess.py)
<case>-mask.nii.gz                       brain mask
<case>-seg.nii.gz                        BraTS label map (2023: 1 NCR, 2 SNFH, 3 ET)
<case>-<mod>-runNN.nii[.gz]              N imputations of <mod> (imputation/run_imputation.py)
<case>-<mod>-runNN-promptKK.nii[.gz]     segmentation of run NN with prompt KK, binary or probability
```

Only cases with imputations are evaluated, and every run needs the same K prompts. The real `<case>-<mod>`
image and `-seg` are the ground truth (evaluation and labels only); without them only the ground-truth-free
outputs are computed. If `run_imputation.py`'s `<mod>/original/` link to the real image is also found, it is
used, since it is the image BraSyn is compared with. The region (`region`, default WT, as segmented by MedSAM2)
is taken from `-seg`.

## Outputs

`outputs/<run_id>/`:

| File | Content |
|---|---|
| `metrics_per_patient.csv` | one row per patient: N, K, `img_*` (imputation vs real), `seg_*` (BraTS), `dist_*` (N images), `anova_share_*`, `split` |
| `metrics_summary.json` | configuration, split, cohort mean / std / n of every metric, model C and test AUROC |
| `maps/<patient>/*.nii.gz` | in BraSyn space (IPL, 144 x 192 x 192, with its affine): `local_{ssim,l1,l2,variance}`, `seg_agreement`, `anova_{var,frac}_*`, `patch_label`, `confidence` (NaN where undefined) |
| `figures/<patient>.png` | axial slice: local quality maps, agreement, ANOVA shares, confidence heatmap next to the true patch labels |

## Layout

```
trust_mri_eval/
  __init__.py  imports BraSyn from BRASYN_REPO (default ../external/BraSyn_tutorial)
  config.py  cli.py  pipeline.py  preprocessing.py  patches.py
  data/        types.py (PatientData), io.py
  metrics/     image_local.py (local SSIM, L1, L2 maps)
  uncertainty/ anova.py
  model/       features.py, labels.py, logistic.py
  viz/         plots.py
tests/
```
