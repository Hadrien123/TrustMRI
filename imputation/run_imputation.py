"""Leave-one-modality-out imputation of a BraTS split with the pretrained BraSyn model.

For each modality in the config, that modality is withheld for every subject, the other three
are fed to BraSyn, and the synthesised image is written (in the original image grid) to
    <output_dir>/<class folder>/<subject>/<subject>-<modality><output_ext>
so after all passes every subject folder holds its imputed modalities, mirroring the dataset.

Usage:  python run_imputation.py [--config config.yaml]
"""
import argparse
import os
import shutil
import sys

import nibabel as nib
import numpy as np
import torch
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


def stage_gz(subjects, gz_dir):
    """BraSyn's loader only reads *.nii.gz; convert each modality once into gz_dir/<subject>/."""
    for _, sid, path in subjects:
        os.makedirs(os.path.join(gz_dir, sid), exist_ok=True)
        for mod in ALL_MODALITIES:
            dst = os.path.join(gz_dir, sid, f"{sid}-{mod}.nii.gz")
            if os.path.exists(dst):
                continue
            src = next((p for p in (os.path.join(path, f"{sid}-{mod}{e}") for e in (".nii", ".nii.gz"))
                        if os.path.exists(p)), None)
            if src is None:
                raise FileNotFoundError(f"{sid}: no {mod} image in {path}")
            nib.save(nib.load(src), dst + ".part.nii.gz")
            os.replace(dst + ".part.nii.gz", dst)


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


def run_pass(data_path, flat_out, params, weights_dir, mod, save_ext):
    """Impute `mod` for every subject folder in data_path -> flat_out/<subject>-<mod><ext>.
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
        model.set_input(data)
        model.test()
        assert data["test_target_modality"][0] == mod
        pred = model.fake_B.detach().cpu().squeeze().numpy()
        src_path = data["A_paths"][0]
        sid = os.path.basename(os.path.dirname(src_path))
        nib.save(to_original_space(pred, src_path), os.path.join(flat_out, f"{sid}-{mod}{save_ext}"))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=os.path.join(here, "config.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="validate config/inputs and print the plan; write nothing")
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
    assert ext in (".nii", ".nii.gz") and set(mods) <= set(ALL_MODALITIES), "bad output_ext/modalities in config"
    assert os.path.exists(os.path.join(weights, "latest_net_G.pth")), f"no latest_net_G.pth in {weights}"

    subjects = find_subjects(data_dir)
    if cfg.get("limit"):
        subjects = subjects[: cfg["limit"]]
    print(f"{len(subjects)} subjects, passes: {mods}\nout: {out}")

    if args.dry_run:
        missing = [(sid, m) for _, sid, path in subjects for m in ALL_MODALITIES
                   if not any(os.path.exists(os.path.join(path, f"{sid}-{m}{e}")) for e in (".nii", ".nii.gz"))]
        classes = {}
        for cls, _, _ in subjects:
            classes[cls] = classes.get(cls, 0) + 1
        print(f"data_dir : {data_dir}\nweights  : {weights}\nbrasyn   : {repo}\nwork dir : {work}")
        print(f"classes  : {classes}\nmissing input images: {missing or 'none'}")
        print(f"GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'}")
        for mod in mods:
            done = sum(os.path.exists(os.path.join(out, c, s, f"{s}-{mod}{ext}")) for c, s, _ in subjects)
            print(f"pass {mod}: use {[m for m in ALL_MODALITIES if m != mod]} -> {len(subjects) - done} to impute, {done} already done")
        c, s, _ = subjects[0]
        print(f"example output: {os.path.join(out, c, s, f'{s}-{mods[0]}{ext}')}")
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
        todo = [s for s in subjects if not os.path.exists(os.path.join(out, s[0], s[1], f"{s[1]}-{mod}{ext}"))]
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

        run_pass(in_dir, flat_out, params, weights, mod, ext)

        for cls, sid, _ in todo:
            fn = f"{sid}-{mod}{ext}"
            if not os.path.exists(os.path.join(flat_out, fn)):
                print(f"WARNING: no output for {sid} ({mod})")
                continue
            os.makedirs(os.path.join(out, cls, sid), exist_ok=True)
            shutil.move(os.path.join(flat_out, fn), os.path.join(out, cls, sid, fn))

    shutil.rmtree(work, ignore_errors=True)
    print(f"\nDone. Results in {out}")


if __name__ == "__main__":
    main()
