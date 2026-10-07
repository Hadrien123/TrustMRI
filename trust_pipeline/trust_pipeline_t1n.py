# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "torchvision",
#     "nibabel",
#     "numpy<2.3",
#     "pandas<3",
#     "scipy",
#     "scikit-image",
#     "scikit-learn",
#     "statsmodels",
#     "matplotlib",
#     "pyyaml",
#     "tqdm",
#     "dominate",
#     "blitz-bayesian-pytorch",
#     "beautifulsoup4",
#     "requests",
#     "pillow",
#     "packaging",
#     "hydra-core",
#     "iopath",
#     "SimpleITK",
#     "BraTS-evaluation",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import shutil
    import subprocess
    import sys
    import urllib.request
    from itertools import combinations

    import marimo as mo
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from scipy import ndimage as ndi
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    return (
        F,
        LogisticRegression,
        StandardScaler,
        anova_lm,
        combinations,
        make_pipeline,
        mo,
        ndi,
        nib,
        np,
        ols,
        pd,
        plt,
        psnr,
        shutil,
        subprocess,
        sys,
        torch,
        train_test_split,
        urllib,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Trust pipeline on BraTS-Africa (t1n)

    Structure of `imputation_eval.ipynb`, on every test subject:

    0. pipeline: N imputations of t1n per subject (`run_imputation.py`, `config_trust_t1n.yaml`), then K MedSAM2
       segmentations per imputation (`segment_imputed.py`, team 2's `segmentation_module(1).py`);
    1. imputation: imputed runs vs real image, spread of the N runs;
    2. segmentation: BraTS metrics per (run, prompt), confidence map of the N x K masks;
    3. ANOVA: Dice variability due to imputation vs prompt;
    4. all subjects: local logistic regression trained on 60 subjects, evaluated on the other 13.

    Upload next to this notebook: `config_trust_t1n.yaml`, `segment_imputed.py`, `segmentation_module(1).py`,
    `latest_net_G.pth` (BraSyn fine-tuned on BraTS-Africa) and `test.zip.part00`, `part01`... (`test.zip` split; `test/<class>/<subject>/<subject>-{t1c,t1n,t2f,t2w,seg}.nii.gz`).
    """)
    return


@app.cell
def _(mo, shutil, subprocess, sys, torch, urllib):
    HERE = mo.notebook_dir()
    IMPUTATION_CONFIG = HERE / "config_trust_t1n.yaml"
    GT_DIR = HERE / "test"  # data_dir of IMPUTATION_CONFIG
    IMPUTED_DIR = HERE / "results/trust_t1n"  # its output_dir
    WORK = HERE / "work"
    MOD = "t1n"
    N_TEST = 13  # subjects held out to evaluate the logistic regression
    PATCH = 3  # local metrics: means over PATCH^3 voxels
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if not GT_DIR.exists():  # test.zip, uploaded in parts of < 100 MB (molab's upload limit)
        WORK.mkdir(exist_ok=True)
        with open(WORK / "test.zip", "wb") as _zip:
            for _part in sorted(HERE.glob("test.zip.part*")):
                _zip.write(_part.read_bytes())
        shutil.unpack_archive(WORK / "test.zip", HERE)
    for _url in ["https://github.com/WinstonHuTiger/BraSyn_tutorial.git", "https://github.com/Hadrien123/TrustMRI.git",
                 "https://github.com/bowang-lab/MedSAM2.git"]:
        _dst = WORK / _url.split("/")[-1].removesuffix(".git")
        if not _dst.exists():
            subprocess.run(["git", "clone", "--depth", "1", _url, str(_dst)], check=True)
    _ckpt = WORK / "MedSAM2/checkpoints/MedSAM2_latest.pt"
    if not _ckpt.exists():
        urllib.request.urlretrieve("https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt", _ckpt)
    # Upstream BraSyn bug: passes a 4x4 affine to ornt_transform, which expects an orientation
    _ds = WORK / "BraSyn_tutorial/project/data/brain_3D_random_mod_dataset.py"
    _ds.write_text(_ds.read_text().replace("ornt_transform(affine, targ_ornt)", "ornt_transform(nib.orientations.io_orientation(affine), targ_ornt)"))

    sys.path.insert(0, str(WORK / "BraSyn_tutorial/project"))
    from util.ssim import create_window  # BraSyn's Gaussian SSIM window
    from data.data_augmentation_3D import getBetterOrientation, toGrayScale  # BraSyn's preprocessing
    from brats_evaluation import config_path, evaluate_single_exam  # BraTS_evaluation
    from panoptica import Panoptica_Evaluator

    EVALUATOR = Panoptica_Evaluator.load_from_config(str(config_path("gli")))  # BraTS glioma config
    return (
        DEVICE,
        GT_DIR,
        HERE,
        IMPUTATION_CONFIG,
        IMPUTED_DIR,
        MOD,
        N_TEST,
        PATCH,
        WORK,
        create_window,
        getBetterOrientation,
        toGrayScale,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Pipeline

    Imputation skips the subjects already done; segmentation runs once (it ends by writing `segmentation_t1n.csv`).
    """)
    return


@app.cell
def _(HERE, IMPUTATION_CONFIG, IMPUTED_DIR, MOD, WORK, pd, subprocess, sys):
    def run(*cmd):
        """Run a script, streaming its output into the cell."""
        with subprocess.Popen([sys.executable, *map(str, cmd)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
            for line in p.stdout:
                print(line, end="")
        assert p.returncode == 0, f"{cmd} failed"


    run(WORK / "TrustMRI/imputation/run_imputation.py", "--config", IMPUTATION_CONFIG)
    if not (IMPUTED_DIR / f"segmentation_{MOD}.csv").exists():
        run(HERE / "segment_imputed.py")
    mod_dirs = sorted(IMPUTED_DIR.glob(f"*/*/{MOD}"))  # <class>/<subject>/<mod>
    seg_per_run = pd.read_csv(IMPUTED_DIR / f"segmentation_{MOD}.csv")
    seg_per_run.groupby("case_id")[["dice", "nsd", "hd95_mm", "mean_uncertainty_tumor"]].mean()
    return (mod_dirs,)


@app.cell
def _(N_TEST, mod_dirs, train_test_split):
    # Subjects split once, stratified by class (51_OtherNeoplasms / 95_Glioma)
    train_dirs, test_dirs = train_test_split(mod_dirs, test_size=N_TEST, random_state=0,
                                             stratify=[d.parent.parent.name for d in mod_dirs])
    f"{len(train_dirs)} train / {len(test_dirs)} test subjects"
    return (test_dirs,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Load one subject

    `load` is BraSyn's dataloader transform (`data/brain_3D_random_mod_dataset.py`): reorient to IPL, then crop.
    The real image is also rescaled to [0, 1] with `toGrayScale`; the imputed runs already are (they are BraSyn's output).
    """)
    return


@app.cell
def _(getBetterOrientation, nib, np, toGrayScale):
    def load(path):
        return getBetterOrientation(nib.load(path), "IPL").get_fdata()[8:152, 24:216, 24:216]


    def load_subject(mod_dir):
        real = toGrayScale(load(next((mod_dir / "original").glob("*.nii*")))).astype(np.float32)
        imputed = np.stack([load(p) for p in sorted((mod_dir / "imputed").glob("*.nii*"))]).astype(np.float32)
        return real, imputed

    return load, load_subject


@app.cell
def _(load_subject, mod_dirs):
    mod_dir = mod_dirs[0]  # change the index to pick another subject
    real, imputed = load_subject(mod_dir)
    z = real.shape[0] // 2  # middle axial slice (axis 0 is inferior-superior in IPL)
    f"{mod_dir.parent.name} | real {real.shape} | imputed {imputed.shape} (N runs, I, P, L)"
    return imputed, mod_dir, real, z


@app.cell
def _(imputed, plt, real, z):
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 4))
    for _ax, (_title, _img) in zip(_axes, {"real": real, "imputed (run 0)": imputed[0], "std across runs": imputed.std(0)}.items()):
        _ax.imshow(_img[z], cmap="gray")
        _ax.set_title(_title)
        _ax.axis("off")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Metrics: global and local versions

    Every voxel-wise metric map has two versions, inside the brain (`real > 0`):
    - **global**: mean over the brain;
    - **local**: mean over each `PATCH`³ patch (patch grid, NaN outside the brain), to localise the errors.

    `ssim_map` is BraSyn's `util.ssim` (same Gaussian window and constants), which only returns the mean, kept per voxel.
    """)
    return


@app.cell
def _(DEVICE, F, PATCH, create_window, np, plt, torch):
    def ssim_map(a, b):
        x, y = (torch.from_numpy(np.ascontiguousarray(v, np.float32))[None, None].to(DEVICE) for v in (a, b))
        window = create_window(11, x.shape).to(x)

        def blur(v):
            return F.conv3d(v, window, padding=5)

        mx, my = blur(x), blur(y)
        vx, vy, cxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
        s = (2 * mx * my + 0.01 ** 2) * (2 * cxy + 0.03 ** 2) / ((mx ** 2 + my ** 2 + 0.01 ** 2) * (vx + vy + 0.03 ** 2))
        return s[0, 0].cpu().numpy()


    def patches(volume, brain):
        """Local version: mean of each PATCH^3 patch over its brain voxels (NaN where it has none)."""
        n = [d // PATCH for d in volume.shape]

        def block_sum(v):
            return v.reshape(n[0], PATCH, n[1], PATCH, n[2], PATCH).sum((1, 3, 5))

        with np.errstate(invalid="ignore", divide="ignore"):
            return block_sum(volume * brain) / block_sum(brain.astype(np.float32))


    def show(local_maps, z, cmap="viridis"):
        """Axial slice z (in voxels) of each local (patch) map."""
        fig, axes = plt.subplots(1, len(local_maps), figsize=(4 * len(local_maps), 4))
        for ax, (title, m) in zip(np.atleast_1d(axes), local_maps.items()):
            im = ax.imshow(m[z // PATCH], cmap=cmap)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
        return fig

    return patches, show, ssim_map


@app.cell
def _(mo):
    mo.md(r"""
    ## 1a. Imputed vs real

    Each run against the real image: SSIM, L1 and L2 (global and local, the local maps averaged over the runs), and PSNR
    as in BraSyn's `test.py` (`data_range` = range of the imputed image).
    """)
    return


@app.cell
def _(np, patches, pd, psnr, ssim_map):
    def compare_to_real(real, imputed):
        brain, rows, local = real > 0, [], {}
        for run in imputed:
            maps = {"SSIM": ssim_map(real, run), "L1": np.abs(run - real), "L2": (run - real) ** 2}
            rows.append({**{k: float(m[brain].mean()) for k, m in maps.items()},
                         "PSNR": psnr(real, run, data_range=run.max() - run.min())})
            for k, m in maps.items():
                local[k] = local.get(k, 0) + patches(m, brain) / len(imputed)
        return pd.DataFrame(rows), local

    return (compare_to_real,)


@app.cell
def _(compare_to_real, imputed, mo, real, show, z):
    _per_run, _local_quality = compare_to_real(real, imputed)
    mo.vstack([show({f"local {k}": m for k, m in _local_quality.items()}, z), _per_run.agg(["mean", "std"]).T])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1b. Spread across the N runs (no ground truth needed)

    Same input, N outputs: voxel-wise variance across the runs, SSIM and PSNR between every pair of runs (global and local;
    the local PSNR of a pair uses the mean squared difference of each patch).
    """)
    return


@app.cell
def _(combinations, np, patches, psnr, ssim_map):
    def spread(real, imputed):
        brain, pairs = real > 0, list(combinations(imputed, 2))
        variance, pair_ssim, pair_psnr, ssim_values = imputed.var(0), 0, 0, []
        for a, b in pairs:
            m = ssim_map(a, b)
            ssim_values.append(float(m[brain].mean()))
            pair_ssim = pair_ssim + m / len(pairs)
            mse = np.maximum(patches((a - b) ** 2, brain), 1e-10)  # identical patch: 100 dB, not inf
            pair_psnr = pair_psnr + 10 * np.log10((b.max() - b.min()) ** 2 / mse) / len(pairs)
        summary = {"mean variance": float(variance[brain].mean()), "pairwise SSIM": np.mean(ssim_values),
                   "pairwise PSNR": np.mean([psnr(a, b, data_range=b.max() - b.min()) for a, b in pairs])}
        return summary, {"variance": patches(variance, brain), "pairwise SSIM": patches(pair_ssim, brain),
                         "pairwise PSNR": pair_psnr}

    return (spread,)


@app.cell
def _(imputed, mo, pd, real, show, spread, z):
    _summary, _local_spread = spread(real, imputed)
    mo.vstack([show({f"local {k}": m for k, m in _local_spread.items()}, z), pd.Series(_summary)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Segmentation

    One mask per (imputed run n, prompt k), written by `segment_imputed.py`:
    `<class>/<subject>/<mod>/segmentation/runNN/<subject>_imputed_prompt_masks/promptKK.nii.gz` (whole tumour).
    Each mask is scored against the reference `<subject>-seg.nii*` with `brats_evaluation.evaluate_single_exam`
    (config `gli`, whole-tumour group). Dice and NSD measure the whole mask; the lesion-wise F1 measures lesion
    detection (lesions found, missed or invented).
    """)
    return


@app.cell
def _(DEVICE, F, GT_DIR, load, ndi, np, pd, torch):
    CROSS = torch.tensor(ndi.generate_binary_structure(3, 1), dtype=torch.float32, device=DEVICE)[None, None]


    def surface(m):
        """Boundary voxels of binary masks (B, 1, D, H, W) on the GPU: m minus its erosion (as ndi.binary_erosion)."""
        return m * (F.conv3d(m, CROSS, padding=1) < CROSS.sum())


    def within_one_voxel(s):
        """Voxels at distance <= 1 voxel of s (as ndi.distance_transform_edt(~s) <= 1)."""
        return (F.conv3d(s, CROSS, padding=1) > 0).float()


    def load_segmentations(mod_dir, n_runs):
        gt = load(next(GT_DIR.rglob(f"{mod_dir.parent.name}-seg.nii*"))).astype(np.uint8)  # BraTS label map
        files = sorted((mod_dir / "segmentation").glob("run*/*_prompt_masks/prompt_*.nii*"))  # run-major
        return gt, np.stack([load(f) > 0.5 for f in files]).reshape(n_runs, -1, *gt.shape)


    def segment_scores(gt, masks):
        """Dice and NSD of each (run, prompt) mask against the whole tumour, on the GPU, as the BraTS evaluation
        (panoptica) computes them: NSD = mean of the fractions of each surface lying on the other one (tolerance 0.5)."""
        g = torch.as_tensor(gt > 0, dtype=torch.float32, device=DEVICE)[None, None]
        g_surface = surface(g)
        rows = []
        for n in range(masks.shape[0]):
            m = torch.as_tensor(masks[n], dtype=torch.float32, device=DEVICE)[:, None]  # (K, 1, D, H, W)
            m_surface = surface(m)
            for k in range(len(m)):
                dice = 2 * (m[k] * g).sum() / (m[k].sum() + g.sum())
                shared = (m_surface[k] * g_surface).sum()
                nsd = (shared / m_surface[k].sum() + shared / g_surface.sum()) / 2
                rows.append({"n": n, "k": k, "dice": float(dice), "nsd": float(nsd)})
        return pd.DataFrame(rows)

    return load_segmentations, segment_scores, surface, within_one_voxel


@app.cell
def _(imputed, load_segmentations, mod_dir, segment_scores):
    gt, masks = load_segmentations(mod_dir, len(imputed))
    per_mask = segment_scores(gt, masks)
    per_mask.drop(columns=["n", "k"]).agg(["mean", "std"]).T
    return gt, masks, per_mask


@app.cell
def _(mo):
    mo.md(r"""
    ### Confidence map of the N x K segmentations

    Fraction of the N x K masks that call each voxel tumour: 0 or 1 where they all agree, in between where the result
    depends on the imputed run or the prompt (mostly at the edges). Red: reference contour.
    """)
    return


@app.cell
def _(gt, masks, np, plt, real):
    def agreement(masks):
        return masks.mean((0, 1))


    _p = agreement(masks)
    _zt = int(np.argmax((gt > 0).sum((1, 2))))  # axial slice with the largest tumour
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.imshow(real[_zt], cmap="gray")
    _im = _ax.imshow(np.ma.masked_equal(_p[_zt], 0), cmap="viridis", vmin=0, vmax=1, alpha=0.8)
    _ax.contour(gt[_zt] > 0, levels=[0.5], colors="red", linewidths=1)
    _ax.set_title("agreement of the N x K masks")
    _ax.axis("off")
    _fig.colorbar(_im, ax=_ax, fraction=0.046)
    _fig
    return (agreement,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Imputation × segmentation

    Two-way ANOVA of the Dice per (n, k), to split the variability between imputation and prompt
    (one mask per cell, so the residual is the interaction).
    """)
    return


@app.cell
def _(anova_lm, ols, per_mask):
    _anova = anova_lm(ols("dice ~ C(n) + C(k)", per_mask).fit())
    _anova.assign(share=_anova["sum_sq"] / _anova["sum_sq"].sum())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. All subjects: metrics and local confidence model

    Per patch of the brain, local features computed **without** the ground truth: SSIM, PSNR and variance between the
    runs (imputation); pairwise Dice and NSD between the MedSAM2 passes (segmentation). Target (needs the ground truth):
    local accuracy, the fraction of the patch's brain voxels segmented right, averaged over the N x K passes. A logistic
    regression on this fractional target, trained on the 60 train subjects, predicts it in [0, 1] for the 13 test subjects.
    """)
    return


@app.cell
def _(
    DEVICE,
    F,
    PATCH,
    agreement,
    combinations,
    np,
    patches,
    pd,
    surface,
    torch,
    within_one_voxel,
):
    def local_pass_agreement(masks, brain):
        """Pairwise Dice and NSD (tolerance 1 voxel) between the K passes of each run, per patch, averaged over the
        pairs and the runs (1 where the patch has no mask voxel / no boundary). On the GPU."""
        b = torch.as_tensor(brain, dtype=torch.float32, device=DEVICE)[None, None]

        def pool(v):  # per patch, mean over its brain voxels (the ratio of two pools is the ratio of the sums)
            return F.avg_pool3d(v * b, PATCH)

        dice, nsd, n_pairs = 0, 0, 0
        for masks_n in masks:
            m = torch.as_tensor(masks_n, dtype=torch.float32, device=DEVICE)[:, None]  # (K, 1, D, H, W)
            surf = surface(m)
            near = within_one_voxel(surf)
            for i, j in combinations(range(len(m)), 2):
                d = 2 * pool(m[i] * m[j]) / (pool(m[i]) + pool(m[j]))
                s = (pool(surf[i] * near[j]) + pool(surf[j] * near[i])) / (pool(surf[i]) + pool(surf[j]))
                dice, nsd, n_pairs = dice + torch.nan_to_num(d, nan=1.0), nsd + torch.nan_to_num(s, nan=1.0), n_pairs + 1
        return (dice / n_pairs)[0, 0].cpu().numpy(), (nsd / n_pairs)[0, 0].cpu().numpy()


    def patch_table(real, gt, masks, local_spread):
        brain, p = real > 0, agreement(masks)
        roi = patches(brain.astype(np.float32), brain) > 0  # every patch with brain voxels
        pair_dice, pair_nsd = local_pass_agreement(masks, brain)
        features = {"SSIM": local_spread["pairwise SSIM"], "PSNR": local_spread["pairwise PSNR"],
                    "variance": local_spread["variance"], "pairwise Dice": pair_dice, "pairwise NSD": pair_nsd}
        # Target: fraction of the patch's brain voxels segmented right, averaged over the N x K passes (= 1 - |p - GT|)
        accuracy = patches(1 - np.abs(p - (gt > 0)), brain)
        tumour = patches(((gt > 0) | (p > 0)).astype(np.float32), brain) > 0  # patch touches the GT or a mask
        return pd.DataFrame({**{k: f[roi] for k, f in features.items()},
                             "accuracy": accuracy[roi], "tumour": tumour[roi]}), roi

    return (patch_table,)


@app.cell
def _(
    IMPUTED_DIR,
    MOD,
    anova_lm,
    compare_to_real,
    load_segmentations,
    load_subject,
    mo,
    mod_dirs,
    ols,
    patch_table,
    pd,
    segment_scores,
    spread,
    test_dirs,
):
    _rows, tables = [], {}
    for _d in mo.status.progress_bar(mod_dirs, title="Evaluating subjects"):
        _real, _imputed = load_subject(_d)
        _gt, _masks = load_segmentations(_d, len(_imputed))
        _per_run, _ = compare_to_real(_real, _imputed)
        _summary, _local_spread = spread(_real, _imputed)
        _per_mask = segment_scores(_gt, _masks)
        _anova = anova_lm(ols("dice ~ C(n) + C(k)", _per_mask).fit())
        tables[_d.parent.name] = patch_table(_real, _gt, _masks, _local_spread)
        _rows.append({"subject": _d.parent.name, "class": _d.parent.parent.name,
                      "split": "test" if _d in test_dirs else "train", "n_runs": len(_imputed),
                      **_per_run.mean().add_suffix(" (mean over runs)"), **_summary,
                      **_per_mask.drop(columns=["n", "k"]).mean().add_suffix(" (mean over masks)"),
                      **(_anova["sum_sq"] / _anova["sum_sq"].sum()).rename(lambda s: f"Dice variance share {s}")})
    results = pd.DataFrame(_rows).set_index("subject")
    pd.to_pickle(tables, IMPUTED_DIR / f"tables_{MOD}.pkl")  # patch features: reload with pd.read_pickle
    results
    return results, tables


@app.cell
def _(mo):
    mo.md(r"""
    ### Local logistic regression → predicted local accuracy

    Fitted once on the patches of the 60 train subjects, then predicts the local accuracy of the 13 test subjects.
    MAE and R² against the true local accuracy, on all patches and on the patches touching the tumour.
    """)
    return


@app.cell
def _(
    IMPUTED_DIR,
    LogisticRegression,
    MOD,
    N_TEST,
    StandardScaler,
    make_pipeline,
    mo,
    mod_dirs,
    np,
    pd,
    plt,
    results,
    tables,
    train_test_split,
):
    from sklearn.metrics import mean_absolute_error, r2_score

    # Subjects BraSyn was fine-tuned on (in ~/software/brasyn_finetune): left out, the split is redrawn on the others
    LEAKED = {f"BraTS-SSA-{_n}-000" for _n in ["00138", "00169", "00175", "00188", "00210", "00215", "00220", "00221", "00225"]}
    _clean = [d for d in mod_dirs if d.parent.name not in LEAKED]
    _train_dirs, _test_dirs = train_test_split(_clean, test_size=N_TEST, random_state=0,
                                               stratify=[d.parent.parent.name for d in _clean])

    FEATURES = ["SSIM", "PSNR", "variance", "pairwise Dice", "pairwise NSD"]
    _train = pd.concat([tables[d.parent.name][0] for d in _train_dirs])
    # Logistic regression on a fractional target: each patch counts as right with weight accuracy, wrong with 1 - accuracy
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(pd.concat([_train[FEATURES]] * 2), np.r_[np.ones(len(_train)), np.zeros(len(_train))],
              logisticregression__sample_weight=np.r_[_train["accuracy"], 1 - _train["accuracy"]])


    def errors(table):
        """MAE and R² of the predicted local accuracy, on all patches and on the patches touching the tumour."""
        pred = model.predict_proba(table[FEATURES])[:, 1]
        y, t = table["accuracy"].to_numpy(), table["tumour"].to_numpy()
        return {"MAE": mean_absolute_error(y, pred), "R2": r2_score(y, pred),
                "MAE tumour": mean_absolute_error(y[t], pred[t]), "R2 tumour": r2_score(y[t], pred[t])}


    _test_ids = [d.parent.name for d in _test_dirs]
    _errors = pd.DataFrame({s: errors(tables[s][0]) for s in _test_ids}).T
    _errors.loc["pooled"] = errors(pd.concat([tables[s][0] for s in _test_ids]))
    _clean_ids = [d.parent.name for d in _clean]
    results.loc[_clean_ids].assign(split=["test" if s in _test_ids else "train" for s in _clean_ids]).to_csv(
        IMPUTED_DIR / f"metrics_{MOD}.csv")
    _errors.to_csv(IMPUTED_DIR / f"local_accuracy_{MOD}.csv")

    _s = _test_ids[0]  # one test subject, at the patch slice with the most tumour patches
    _table, _roi = tables[_s]
    _maps = {"true local accuracy": _table["accuracy"], "predicted": model.predict_proba(_table[FEATURES])[:, 1]}
    _tumour = np.zeros(_roi.shape)
    _tumour[_roi] = _table["tumour"]
    _z = int(np.argmax(_tumour.sum((1, 2))))
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 5))
    for _ax, (_title, _values) in zip(_axes, _maps.items()):
        _volume = np.full(_roi.shape, np.nan)
        _volume[_roi] = _values
        _im = _ax.imshow(_volume[_z], cmap="RdYlBu", vmin=0, vmax=1)
        _ax.set_title(f"{_s}: {_title}")
        _ax.axis("off")
    _fig.colorbar(_im, ax=_axes, fraction=0.046)
    mo.vstack([_fig, _errors])
    return FEATURES, LEAKED, model


@app.cell
def _(LEAKED, mo, plt, results):
    # Two-way ANOVA of the Dice per (run n, prompt k), per subject (section 3): share of the Dice variance due to the
    # imputation C(n), the prompt C(k) and their interaction (Residual), over the subjects BraSyn was not fine-tuned on
    _shares = results.loc[~results.index.isin(LEAKED),
                          ["Dice variance share C(n)", "Dice variance share C(k)", "Dice variance share Residual"]]
    _shares.columns = ["imputation", "prompt", "interaction"]
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _shares.plot.box(ax=_ax)
    _ax.set_ylabel("share of the Dice variance")
    _ax.set_title(f"Two-way ANOVA of the Dice ({len(_shares)} subjects)")
    mo.vstack([mo.md(f"Imputation explains more Dice variance than the prompt in "
                     f"**{(_shares.imputation > _shares.prompt).mean():.0%}** of the subjects."),
               _fig, _shares.describe().T[["mean", "50%", "std", "min", "max"]]])
    return


@app.cell
def _(
    FEATURES,
    IMPUTED_DIR,
    MOD,
    PATCH,
    load,
    mo,
    model,
    np,
    pd,
    plt,
    tables,
    toGrayScale,
):
    # Regression weights: coefficients on the standardized features (change of the log-odds of a right voxel
    # for +1 std of the feature)
    _coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
    _weights = pd.concat([_coefs, pd.Series({"intercept": model[-1].intercept_[0]})]).rename("weight").to_frame()

    # |predicted - true| local accuracy per patch, averaged per subject: unweighted (each patch counts once) and
    # weighted by the patch's brain voxels (each brain voxel counts once); train vs test, to check for overfitting
    _split = pd.read_csv(IMPUTED_DIR / f"metrics_{MOD}.csv", index_col=0)["split"]
    _rows = []
    for _s, _set in _split.items():
        _table, _roi = tables[_s]
        _brain = toGrayScale(load(next(next(IMPUTED_DIR.glob(f"*/{_s}/{MOD}/original")).glob("*.nii*")))) > 0
        _n = [d // PATCH for d in _brain.shape]
        _voxels = _brain.reshape(_n[0], PATCH, _n[1], PATCH, _n[2], PATCH).sum((1, 3, 5))[_roi]
        _error = np.abs(model.predict_proba(_table[FEATURES])[:, 1] - _table["accuracy"].to_numpy())
        _rows.append({"split": _set, "mean |error|": _error.mean(),
                      "weighted mean |error|": np.average(_error, weights=_voxels)})
    _abs_errors = pd.DataFrame(_rows, index=_split.index)

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _abs_errors.boxplot(column=["mean |error|", "weighted mean |error|"], by="split", ax=list(_axes))
    for _ax in _axes:
        _ax.set_ylabel("|predicted - true local accuracy|")
    _fig.suptitle("Absolute error of the predicted local accuracy per subject: train vs test")
    _fig.tight_layout()
    mo.vstack([mo.md("### Regression weights"), _weights, _fig])
    return


if __name__ == "__main__":
    app.run()
