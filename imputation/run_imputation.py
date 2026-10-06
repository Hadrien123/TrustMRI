"""Leave-one-modality-out imputation of a BraTS split with the pretrained BraSyn model.

For each modality in the config, that modality is withheld for every subject, the other three
are fed to BraSyn, and the synthesised image is written (in the original image grid) n_runs times:
    <output_dir>/<class folder>/<subject>/<modality>/original/<subject>-<modality>.nii[.gz]   (symlink)
    <output_dir>/<class folder>/<subject>/<modality>/imputed/<subject>-<modality>-runNN<output_ext>
BraSyn is deterministic, so runs differ only through the perturbations below (run k uses seed k):
  noise   -- Gaussian noise of noise x (each input's brain-voxel std) added to the inputs' brain voxels
  dropout -- element-wise dropout with probability p after SIT's internal ReLUs, kept on at inference
             (MC dropout; the model was trained without dropout)
Set both to 0 for a plain, unperturbed imputation.

Usage:  python run_imputation.py [--config config.yaml] [--n-runs 10] [--noise 0.05] [--dropout 0.02]
"""
import argparse
import os
import shutil
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from tqdm import tqdm

ALL_MODALITIES = ["t1c", "t1n", "t2f", "t2w"]
# Crop BraSyn's dataloader applies to the IPL-reoriented volume (brain_3D_random_mod_dataset.py)
CROP = (slice(8, 152), slice(24, 216), slice(24, 216))


def find_subjects(data_dir):
    """[(class_folder, subject_id, subject_path)] for <data_dir>/<class>/<subject>/."""
    subjects = []
    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            subjects += [(cls, sid, os.path.join(cls_path, sid)) for sid in sorted(os.listdir(cls_path))
                         if os.path.isdir(os.path.join(cls_path, sid))]
    return subjects


def find_image(path, sid, mod):
    """<path>/<sid>-<mod>.nii or .nii.gz, or None."""
    return next((p for p in (os.path.join(path, f"{sid}-{mod}{e}") for e in (".nii", ".nii.gz"))
                 if os.path.exists(p)), None)


def imputed_path(out, cls, sid, mod, run, ext):
    return os.path.join(out, cls, sid, mod, "imputed", f"{sid}-{mod}-run{run:02d}{ext}")


def stage_gz(subjects, gz_dir):
    """BraSyn's loader only reads *.nii.gz; convert each modality once into gz_dir/<subject>/."""
    for _, sid, path in subjects:
        os.makedirs(os.path.join(gz_dir, sid), exist_ok=True)
        for mod in ALL_MODALITIES:
            dst = os.path.join(gz_dir, sid, f"{sid}-{mod}.nii.gz")
            if os.path.exists(dst):
                continue
            src = find_image(path, sid, mod)
            if src is None:
                raise FileNotFoundError(f"{sid}: no {mod} image in {path}")
            nib.save(nib.load(src), dst + ".part.nii.gz")
            os.replace(dst + ".part.nii.gz", dst)


def link_originals(subjects, out, mod):
    """Symlink each subject's real <mod> image into <out>/<class>/<subject>/<mod>/original/."""
    for cls, sid, path in subjects:
        src = os.path.realpath(find_image(path, sid, mod))
        dst = os.path.join(out, cls, sid, mod, "original", os.path.basename(src))
        if not os.path.lexists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)


def to_original_space(pred, src_path):
    """Pad BraSyn's cropped (144,192,192) IPL prediction back to full size and reorient it to
    the source image's own orientation/affine/shape."""
    src = nib.load(src_path)
    ipl = src.as_reoriented(ornt_transform(io_orientation(src.affine), axcodes2ornt("IPL")))
    full = np.zeros(ipl.shape, dtype=np.float32)
    full[CROP] = pred
    img = nib.Nifti1Image(full, ipl.affine)
    img = img.as_reoriented(ornt_transform(io_orientation(img.affine), io_orientation(src.affine)))
    assert img.shape == src.shape and np.allclose(img.affine, src.affine), "output grid != input grid"
    out = nib.Nifti1Image(np.asarray(img.dataobj, dtype=np.float32), src.affine, src.header)
    out.set_data_dtype(np.float32)
    return out


def add_mc_dropout(netG, p):
    """SIT has no dropout layers: apply element-wise dropout after every ReLU except in the output head,
    active even in eval mode, so each forward pass samples a different sub-network."""
    for name, m in netG.named_modules():
        if isinstance(m, torch.nn.ReLU) and "out" not in name.split("."):
            m.register_forward_hook(lambda _m, _in, y: F.dropout(y, p, training=True))


def perturb(A, rel_std):
    """Add Gaussian noise of rel_std x each channel's brain std to brain voxels of A (1,C,D,H,W),
    which BraSyn has scaled to [0,1] with background at 0; background stays 0."""
    brain = A > 0
    std = torch.stack([c[b].std() for c, b in zip(A[0], brain[0])]).view(1, -1, 1, 1, 1)
    return torch.where(brain, (A + torch.randn_like(A) * std * rel_std).clamp(0, 1), A)


def run_pass(data_path, flat_out, params, weights_dir, mod, save_ext, n_runs, noise, dropout):
    """Impute `mod` n_runs times for every subject folder in data_path -> flat_out/<subject>-<mod>-runNN<ext>.
    Mirrors BraSyn's generate_missing_modality.infer() but saves in the original image space."""
    from data import create_dataset
    from generate_missing_modality import OPTIONS
    from models import create_model

    opt = OPTIONS()
    for k, v in params.items():
        setattr(opt, k, v)
    weights_dir = os.path.abspath(weights_dir)
    opt.checkpoints_dir, opt.name = os.path.dirname(weights_dir), os.path.basename(weights_dir)
    opt.dataroot, opt.output_dir = data_path, flat_out
    opt.gpu_ids = [int(i) for i in str(opt.gpu_ids).split(",") if int(i) >= 0]
    if opt.gpu_ids:
        torch.cuda.set_device(opt.gpu_ids[0])
    opt.num_threads, opt.batch_size, opt.serial_batches = 0, 1, True
    opt.paired, opt.no_flip, opt.display_id, opt.phase = True, True, -1, "test"
    os.makedirs(flat_out, exist_ok=True)

    dataset = create_dataset(opt)
    model = create_model(opt)
    for i, data in enumerate(tqdm(dataset, desc=f"impute {mod}")):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)
            model.parallelize()
            model.eval()
            if dropout:
                add_mc_dropout(model.netG, dropout)
        assert data["test_target_modality"][0] == mod
        src_path = data["A_paths"][0]
        sid = os.path.basename(os.path.dirname(src_path))
        for run in range(1, n_runs + 1):
            torch.manual_seed(run)  # seeds the input noise (CPU) and the dropout masks (GPU)
            model.set_input({**data, "A": perturb(data["A"], noise) if noise else data["A"]})
            model.test()
            pred = model.fake_B.detach().cpu().squeeze().numpy()
            nib.save(to_original_space(pred, src_path), os.path.join(flat_out, f"{sid}-{mod}-run{run:02d}{save_ext}"))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=os.path.join(here, "config.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="validate config/inputs and print the plan; write nothing")
    ap.add_argument("--n-runs", type=int, help="imputations per subject and modality (overrides config n_runs)")
    ap.add_argument("--noise", type=float, help="input noise, as a fraction of brain std (overrides config noise)")
    ap.add_argument("--dropout", type=float, help="MC dropout probability (overrides config dropout)")
    args = ap.parse_args()
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    def P(p):  # resolve relative to the config file
        return os.path.normpath(os.path.join(cfg_dir, os.path.expanduser(p)))

    data_dir, out, work = P(cfg["data_dir"]), P(cfg["output_dir"]), P(cfg["work_dir"])
    repo, weights = P(cfg["model"]["brasyn_repo"]), P(cfg["model"]["weights_dir"])
    ext = cfg.get("output_ext", ".nii")
    mods = cfg.get("modalities", ALL_MODALITIES)
    n_runs = args.n_runs if args.n_runs is not None else cfg.get("n_runs", 1)
    noise = args.noise if args.noise is not None else cfg.get("noise", 0.0)
    dropout = args.dropout if args.dropout is not None else cfg.get("dropout", 0.0)
    assert ext in (".nii", ".nii.gz") and set(mods) <= set(ALL_MODALITIES), "bad output_ext/modalities in config"
    assert n_runs >= 1 and noise >= 0 and 0 <= dropout < 1, "need n_runs >= 1, noise >= 0, 0 <= dropout < 1"
    assert os.path.exists(os.path.join(weights, "latest_net_G.pth")), f"no latest_net_G.pth in {weights}"

    subjects = find_subjects(data_dir)
    if cfg.get("limit"):
        subjects = subjects[: cfg["limit"]]
    print(f"{len(subjects)} subjects, passes: {mods}, {n_runs} run(s) each (noise={noise}, dropout={dropout})\nout: {out}")

    def todo_for(mod):  # subjects missing any of the n_runs imputations of mod
        return [s for s in subjects
                if not all(os.path.exists(imputed_path(out, s[0], s[1], mod, r, ext)) for r in range(1, n_runs + 1))]

    if args.dry_run:
        missing = [(sid, m) for _, sid, path in subjects for m in ALL_MODALITIES if find_image(path, sid, m) is None]
        classes = {}
        for cls, _, _ in subjects:
            classes[cls] = classes.get(cls, 0) + 1
        print(f"data_dir : {data_dir}\nweights  : {weights}\nbrasyn   : {repo}\nwork dir : {work}")
        print(f"classes  : {classes}\nmissing input images: {missing or 'none'}")
        print(f"GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'}")
        if n_runs > 1 and not noise and not dropout:
            print("WARNING: noise and dropout are both 0, so all runs will be identical")
        for mod in mods:
            todo = len(todo_for(mod))
            print(f"pass {mod}: use {[m for m in ALL_MODALITIES if m != mod]} -> {todo} to impute, {len(subjects) - todo} already done")
        c, s, _ = subjects[0]
        print(f"example output: {imputed_path(out, c, s, mods[0], 1, ext)}")
        return

    gz_dir = os.path.join(work, "gz")
    print("Converting inputs to .nii.gz ...")
    stage_gz(subjects, gz_dir)

    # BraSyn modules import as top-level packages and use cwd-relative paths
    sys.path.insert(0, os.path.join(repo, "project"))
    os.chdir(os.path.join(repo, "project"))
    with open(P(cfg["model"]["parameters_file"])) as f:
        params = yaml.safe_load(f)

    for mod in mods:
        link_originals(subjects, out, mod)
        todo = todo_for(mod)
        print(f"\n=== Dropping {mod}: {len(todo)} to do, {len(subjects) - len(todo)} already done ===")
        if not todo:
            continue
        in_dir, flat_out = os.path.join(work, f"in_{mod}"), os.path.join(work, f"raw_{mod}")
        shutil.rmtree(in_dir, ignore_errors=True)
        shutil.rmtree(flat_out, ignore_errors=True)
        for _, sid, _ in todo:  # 3 remaining modalities per subject, as symlinks
            os.makedirs(os.path.join(in_dir, sid))
            for m in ALL_MODALITIES:
                if m != mod:
                    fn = f"{sid}-{m}.nii.gz"
                    os.symlink(os.path.join(gz_dir, sid, fn), os.path.join(in_dir, sid, fn))

        run_pass(in_dir, flat_out, params, weights, mod, ext, n_runs, noise, dropout)

        for cls, sid, _ in todo:
            for run in range(1, n_runs + 1):
                dst = imputed_path(out, cls, sid, mod, run, ext)
                src = os.path.join(flat_out, os.path.basename(dst))
                if not os.path.exists(src):
                    print(f"WARNING: no output for {sid} ({mod}, run {run})")
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)

    shutil.rmtree(work, ignore_errors=True)
    print(f"\nDone. Results in {out}")


if __name__ == "__main__":
    main()
