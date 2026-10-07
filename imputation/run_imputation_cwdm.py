"""Leave-one-modality-out imputation of a BraTS split with the pretrained cWDM models.

cWDM (Friedrich et al. 2024, https://github.com/pfriedri/cwdm) is a 3D conditional wavelet diffusion
model with one network per target modality, each conditioned on the other three. For each modality in
the config, that modality is withheld for every subject and synthesised n_runs times, written in the
original image grid with the same layout as run_imputation.py:
    <output_dir>/<class folder>/<subject>/<modality>/original/<subject>-<modality>.nii[.gz]   (symlink)
    <output_dir>/<class folder>/<subject>/<modality>/imputed/<subject>-<modality>-runNN<output_ext>
Sampling is stochastic: run k uses seed k, so runs are independent draws from the model.

Pre/post-processing follows the official repo (bratsloader.py, scripts/sample.py):
  in : each input clipped to its 0.1-99.9 % quantiles over the whole volume, scaled to [0, 1],
       zero-padded to 240x240x160 and cropped to 224x224x160
  out: 1000 DDPM steps, clamped to [0, 1], zeroed outside the brain (any input nonzero),
       padded back to 240x240x155

Usage:  python run_imputation_cwdm.py [--config config_africa_cwdm_test.yaml] [--n-runs 5] [--dry-run]
"""
import argparse
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch
import yaml

from run_imputation import ALL_MODALITIES, find_image, find_subjects, imputed_path, link_originals

IN_SHAPE = (240, 240, 155)
CROP = (slice(8, 232), slice(8, 232), slice(0, 160))  # 240x240x160 -> 224x224x160, as in bratsloader.py

# Model/diffusion settings of the released weights (run.sh, MODEL='unet')
MODEL_ARGS = dict(
    image_size=224, num_channels=64, num_res_blocks=2, channel_mult="1,2,2,4,4", num_heads=1,
    attention_resolutions="", use_scale_shift_norm=False, bottleneck_attention=False, resample_2d=False,
    additive_skips=False, use_freq=False, dims=3, num_groups=32, in_channels=32, out_channels=8,
    class_cond=False, learn_sigma=False, use_fp16=False, dataset="brats",
    diffusion_steps=1000, noise_schedule="linear", predict_xstart=True,
    rescale_timesteps=False, rescale_learned_sigmas=False,
)


def clip_and_normalize(img):
    """bratsloader.clip_and_normalize: clip to whole-volume 0.1/99.9 % quantiles, min-max to [0, 1]."""
    img = np.clip(img, np.quantile(img, 0.001), np.quantile(img, 0.999))
    return (img - img.min()) / max(img.max() - img.min(), 1e-8)


def load_input(path):
    """Normalised input as a (1,1,224,224,160) tensor, plus its nibabel image."""
    img = nib.load(path)
    assert img.shape == IN_SHAPE, f"{path}: shape {img.shape}, cWDM expects {IN_SHAPE}"
    vol = np.zeros((240, 240, 160), dtype=np.float32)
    vol[..., :155] = clip_and_normalize(img.get_fdata())
    return torch.from_numpy(vol[CROP]).view(1, 1, 224, 224, 160), img


def load_model(weights, device):
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

    params = model_and_diffusion_defaults()
    params.update(MODEL_ARGS)
    model, diffusion = create_model_and_diffusion(**params)
    diffusion.mode = "i2i"
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.to(device)  # cWDM's to() returns None
    model.eval()
    return model, diffusion


def impute(model, diffusion, dwt, idwt, conds, seed):
    """One cWDM sample (224,224,160) in [0, 1] from the three conditioning volumes."""
    cond = []
    for c in conds:  # 8 wavelet sub-bands per input, low-pass scaled by 1/3 as in training
        LLL, *high = dwt(c)
        cond += [LLL / 3.0, *high]
    cond = torch.cat(cond, dim=1)

    torch.manual_seed(seed)
    noise = torch.randn(1, 8, 112, 112, 80, device=cond.device)
    with torch.no_grad():
        sample = diffusion.p_sample_loop(model=model, shape=noise.shape, noise=noise, cond=cond,
                                         clip_denoised=True, model_kwargs={}, progress=False)
        bands = [sample[:, [i]] for i in range(8)]
        bands[0] = bands[0] * 3.0
        sample = idwt(*bands)
    return sample.clamp(0, 1).squeeze().cpu().numpy()


def to_original_space(pred, brain, ref):
    """Pad the (224,224,160) prediction back to 240x240x155 on the reference input's grid."""
    full = np.zeros((240, 240, 160), dtype=np.float32)
    full[CROP] = pred
    full = full[..., :155] * brain
    out = nib.Nifti1Image(full, ref.affine, ref.header)
    out.set_data_dtype(np.float32)
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=os.path.join(here, "config_africa_cwdm_test.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="validate config/inputs and print the plan; write nothing")
    ap.add_argument("--n-runs", type=int, help="imputations per subject and modality (overrides config n_runs)")
    args = ap.parse_args()
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    def P(p):  # resolve relative to the config file
        return os.path.normpath(os.path.join(cfg_dir, os.path.expanduser(p)))

    data_dir, out, repo = P(cfg["data_dir"]), P(cfg["output_dir"]), P(cfg["model"]["cwdm_repo"])
    weights = {m: P(w) for m, w in cfg["model"]["weights"].items()}
    ext = cfg.get("output_ext", ".nii")
    mods = cfg.get("modalities", ALL_MODALITIES)
    n_runs = args.n_runs if args.n_runs is not None else cfg.get("n_runs", 1)
    assert ext in (".nii", ".nii.gz") and set(mods) <= set(ALL_MODALITIES), "bad output_ext/modalities in config"
    assert n_runs >= 1, "need n_runs >= 1"
    for m in mods:
        assert os.path.exists(weights.get(m, "")), f"no weights for {m}: {weights.get(m)}"

    subjects = find_subjects(data_dir)
    if cfg.get("limit"):
        subjects = subjects[: cfg["limit"]]
    print(f"{len(subjects)} subjects, passes: {mods}, {n_runs} run(s) each\nout: {out}")

    def todo_for(mod):  # (subject, run) imputations of mod not written yet
        return [(s, r) for s in subjects for r in range(1, n_runs + 1)
                if not os.path.exists(imputed_path(out, s[0], s[1], mod, r, ext))]

    if args.dry_run:
        missing = [(sid, m) for _, sid, path in subjects for m in ALL_MODALITIES if find_image(path, sid, m) is None]
        classes = {}
        for cls, _, _ in subjects:
            classes[cls] = classes.get(cls, 0) + 1
        print(f"data_dir : {data_dir}\ncwdm     : {repo}")
        for m in mods:
            print(f"weights {m}: {weights[m]}")
        print(f"classes  : {classes}\nmissing input images: {missing or 'none'}")
        print(f"GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'}")
        for mod in mods:
            todo = len(todo_for(mod))
            print(f"pass {mod}: use {[m for m in ALL_MODALITIES if m != mod]} -> {todo} to impute, "
                  f"{len(subjects) * n_runs - todo} already done")
        c, s, _ = subjects[0]
        print(f"example output: {imputed_path(out, c, s, mods[0], 1, ext)}")
        return

    sys.path.insert(0, repo)
    from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

    device = torch.device("cuda")
    dwt, idwt = DWT_3D("haar"), IDWT_3D("haar")

    for mod in mods:
        link_originals(subjects, out, mod)
        todo = todo_for(mod)
        print(f"\n=== Dropping {mod}: {len(todo)} to do ===")
        if not todo:
            continue
        model, diffusion = load_model(weights[mod], device)
        cond_mods = [m for m in ALL_MODALITIES if m != mod]  # order used in training (scripts/sample.py)
        loaded = None
        for i, ((cls, sid, path), run) in enumerate(todo, 1):
            if loaded != sid:
                inputs = [load_input(find_image(path, sid, m)) for m in cond_mods]
                conds = [t.to(device) for t, _ in inputs]
                ref = inputs[0][1]
                brain = np.any([img.get_fdata() != 0 for _, img in inputs], axis=0).astype(np.float32)
                loaded = sid
            t0 = time.time()
            pred = impute(model, diffusion, dwt, idwt, conds, seed=run)
            dst = imputed_path(out, cls, sid, mod, run, ext)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst[: -len(ext)] + ".part" + ext
            nib.save(to_original_space(pred, brain, ref), tmp)
            os.replace(tmp, dst)
            print(f"[{mod} {i}/{len(todo)}] {sid} run {run:02d}  {time.time() - t0:.0f}s", flush=True)
        del model, diffusion
        torch.cuda.empty_cache()

    print(f"\nDone. Results in {out}")


if __name__ == "__main__":
    main()
