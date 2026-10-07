# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "torchvision",
#     "nibabel",
#     "numpy",
#     "scipy",
#     "scikit-image",
#     "matplotlib",
#     "pyyaml",
#     "tqdm",
#     "polars",
#     "pandas",
#     "dominate",
#     "blitz-bayesian-pytorch",
#     "beautifulsoup4",
#     "requests",
#     "pillow",
#     "packaging",
#     "hydra-core",
#     "iopath",
#     "SimpleITK",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import importlib.util
    import os
    import shutil
    import subprocess
    import sys
    import time
    import urllib.request

    import marimo as mo
    import polars as pl
    import torch
    import yaml

    return importlib, mo, os, pl, shutil, subprocess, sys, time, torch, urllib, yaml


@app.cell
def _(mo):
    HERE = mo.notebook_dir()
    SUBJECT_DIR = HERE  # <sid>-{t1c,t1n,t2f,t2w,seg}.nii.gz, uploaded next to the notebook
    SID = "BraTS-SSA-00007-000"
    MOD = "t2f"  # modality to impute from the other three
    N_RUNS = 10  # run k uses seed k
    NOISE = 0.1  # input noise, fraction of brain std (as in imputation/config.yaml)
    N_PROMPTS = 5  # perturbed MedSAM2 box prompts per image
    SEG_MODULE = HERE / "segmentation_module(1).py"  # team 2's MedSAM2 script, uploaded next to the notebook
    OUT_DIR = HERE / "outputs" / f"{SID}-{MOD}"
    SEG_DIR = OUT_DIR / "segmentation"
    WORK = HERE / "imputation_timing_work"

    _missing = [_m for _m in ["t1c", "t1n", "t2f", "t2w", "seg"] if not list(SUBJECT_DIR.glob(f"{SID}-{_m}.nii*"))]
    assert not _missing, f"{_missing} absent(s) de {SUBJECT_DIR}. .nii.gz trouvés : {[str(_p) for _p in HERE.rglob('*.nii*')]}"
    return MOD, NOISE, N_PROMPTS, N_RUNS, OUT_DIR, SEG_DIR, SEG_MODULE, SID, SUBJECT_DIR, WORK


@app.cell
def _(WORK, subprocess, sys, torch, yaml):
    BRASYN, TRUSTMRI = WORK / "BraSyn_tutorial", WORK / "TrustMRI"
    for _url, _dst in [
        ("https://github.com/WinstonHuTiger/BraSyn_tutorial.git", BRASYN),  # includes the weights
        ("https://github.com/Hadrien123/TrustMRI.git", TRUSTMRI),
    ]:
        if not _dst.exists():
            subprocess.run(["git", "clone", "--depth", "1", _url, str(_dst)], check=True)

    # Upstream BraSyn bug: passes a 4x4 affine to ornt_transform, which expects an orientation
    _ds = BRASYN / "project/data/brain_3D_random_mod_dataset.py"
    _ds.write_text(_ds.read_text().replace("ornt_transform(affine, targ_ornt)", "ornt_transform(nib.orientations.io_orientation(affine), targ_ornt)"))

    sys.path.insert(0, str(BRASYN / "project"))
    sys.path.insert(0, str(TRUSTMRI / "imputation"))
    import run_imputation as ri

    params = yaml.safe_load((BRASYN / "mlcube/workspace/parameters.yaml").read_text())
    params["gpu_ids"] = "0" if torch.cuda.is_available() else "-1"  # GPU 0, else CPU
    return BRASYN, params, ri


@app.cell
def _(
    BRASYN,
    MOD,
    NOISE,
    N_RUNS,
    OUT_DIR,
    SID,
    SUBJECT_DIR,
    WORK,
    os,
    params,
    pl,
    ri,
    shutil,
    time,
):
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.mkdir(parents=True)
    shutil.copy(ri.find_image(str(SUBJECT_DIR), SID, MOD), OUT_DIR)

    # BraSyn input: <in_dir>/<sid>/ with the 3 other modalities as .nii.gz
    _in_dir = WORK / f"in_{MOD}"
    shutil.rmtree(_in_dir, ignore_errors=True)
    ri.stage_gz([("", SID, str(SUBJECT_DIR))], str(_in_dir))
    os.remove(_in_dir / SID / f"{SID}-{MOD}.nii.gz")

    os.chdir(BRASYN / "project")  # BraSyn uses cwd-relative paths
    _t0 = time.time()
    ri.run_pass(str(_in_dir), str(OUT_DIR), params, str(BRASYN / "mlcube/workspace/additional_files/weights/your_weight_name"),
                MOD, ".nii.gz", N_RUNS, NOISE, 0.0)

    # Each run ends by saving its image, so file mtimes give per-run durations (run 1 includes model loading)
    run_paths = [OUT_DIR / f"{SID}-{MOD}-run{_r:02d}.nii.gz" for _r in range(1, N_RUNS + 1)]
    _ends = [_p.stat().st_mtime for _p in run_paths]
    timings = pl.DataFrame({"seed": range(1, N_RUNS + 1), "seconds": [_e - _s for _s, _e in zip([_t0] + _ends, _ends)]})
    timings.write_csv(OUT_DIR / "timings.csv")
    timings
    return run_paths, timings


@app.cell
def _(OUT_DIR, mo, timings, torch):
    _device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    mo.md(f"**Total sur {_device} : {timings['seconds'].sum():.1f} s**, soit {timings['seconds'][1:].mean():.1f} s/run hors chargement du modèle. Dossier : `{OUT_DIR}`")
    return


@app.cell
def _(MOD, N_PROMPTS, SEG_MODULE, WORK, importlib, subprocess, sys, urllib):
    MEDSAM2 = WORK / "MedSAM2"
    if not MEDSAM2.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/bowang-lab/MedSAM2.git", str(MEDSAM2)], check=True)
    _ckpt = MEDSAM2 / "checkpoints/MedSAM2_latest.pt"
    if not _ckpt.exists():
        urllib.request.urlretrieve("https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt", _ckpt)
    sys.path.append(str(MEDSAM2))  # appended: MedSAM2's data/ folder must not shadow BraSyn's data package

    # Team 2's segmentation module, loaded from its file (its name is not importable) and pointed at MedSAM2
    _spec = importlib.util.spec_from_file_location("segmentation_module", SEG_MODULE)
    seg = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(seg)
    seg.CONFIG_PATH = MEDSAM2 / "sam2/configs/sam2.1_hiera_t512.yaml"
    seg.CHECKPOINT = _ckpt
    seg.N_PROMPTS = N_PROMPTS
    seg.MODALITY = MOD  # segment the imputed modality
    predictor = seg.load_model()
    return predictor, seg


@app.cell
def _(MOD, OUT_DIR, SEG_DIR, SID, SUBJECT_DIR, pl, predictor, ri, run_paths, seg):
    # As in the module's main(): oracle prompt (slice and box) from the GT, then each image through process_modality.
    # The module seeds the box perturbations from case_id + modality, so every imputed run gets the same N_PROMPTS boxes.
    _gt_nii, _gt = seg.load_nifti(ri.find_image(str(SUBJECT_DIR), SID, "seg"))
    _gt_binary = seg.make_gt_binary(_gt)
    _prompt_z, _box = seg.get_prompt_slice_and_box(_gt_binary)
    _case = {"case_id": SID, "category": ""}

    _rows = []
    _images = [("real", ri.find_image(str(OUT_DIR), SID, MOD))] + [(f"run{_r:02d}", _p) for _r, _p in enumerate(run_paths, 1)]
    for _name, _path in _images:
        seg.OUTPUT_DIR = SEG_DIR / _name
        _res = seg.process_modality(predictor, _case, "real" if _name == "real" else "imputed", str(_path),
                                    _gt_nii, _gt_binary, _prompt_z, _box)
        _rows.append({"image": _name, **_res})

    seg_results = pl.DataFrame(_rows)
    seg_results.write_csv(SEG_DIR / "segmentation_results.csv")
    seg_results.select("image", "dice", "iou", "nsd", "hd95_mm", "mean_uncertainty_tumor", "mean_uncertainty_predicted_tumor")
    return (seg_results,)


@app.cell
def _(OUT_DIR, mo, seg_results, shutil):
    mo.stop(seg_results.is_empty())  # zip once the segmentation has written its files
    _zip = shutil.make_archive(str(OUT_DIR), "zip", OUT_DIR.parent, OUT_DIR.name)
    with open(_zip, "rb") as _f:
        _data = _f.read()
    mo.download(data=_data, filename=f"{OUT_DIR.name}.zip", label="Télécharger les résultats (zip)")
    return


if __name__ == "__main__":
    app.run()
