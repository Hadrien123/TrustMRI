"""NIfTI loading/saving. The rest of the code only sees :class:`PatientData`, in BraSyn space: every volume goes
through BraSyn's dataloader transform (IPL orientation, 144 x 192 x 192 crop), the real image through ``toGrayScale``.

Files are found by name anywhere under the data folder (sub-folders do not matter), with the BraTS
naming used by the preprocessing, imputation and segmentation steps::

    <case>-<mod>.nii[.gz]                  real sequence, mod in t1c/t1n/t2f/t2w; the imputed one is the
                                           ground truth image (evaluation only). If both the preprocessed
                                           one and run_imputation.py's <mod>/original/ link are found,
                                           the original wins: it is the image BraSyn is compared with.
    <case>-mask.nii.gz                     brain mask; if absent, real image > 0 or mean imputed > 0
    <case>-seg.nii.gz                      BraTS label map (evaluation and labels only)
    <case>-<mod>-runNN.nii[.gz]            N imputations of <mod> (imputation/run_imputation.py)
    <case>-<mod>-runNN-promptKK.nii[.gz]   segmentation of run NN with prompt KK, binary or in [0, 1]
                                           (the same K prompts for every run)

Only cases with imputations are loaded. Missing ground truth is allowed: the pipeline then only
computes the ground-truth-free outputs.
"""
from __future__ import annotations

import re
from pathlib import Path

import nibabel as nib
import numpy as np
from data.data_augmentation_3D import getBetterOrientation, toGrayScale   # BraSyn's preprocessing

from .types import PatientData

CROP = (slice(8, 152), slice(24, 216), slice(24, 216))   # BraSyn's dataloader crop of the IPL volume

_NAME = re.compile(r"(?P<case>.+?)-(?P<kind>t1c|t1n|t2f|t2w|mask|seg)"
                   r"(?:-run(?P<n>\d+)(?:-prompt(?P<k>\d+))?)?\.nii(?:\.gz)?")

Key = tuple[str, int | None, int | None]   # (kind, run, prompt)


def index_cases(data_dir: str | Path) -> dict[str, dict[Key, Path]]:
    """Files of every case that has at least one imputation, by case id then (kind, run, prompt)."""
    cases: dict[str, dict[Key, Path]] = {}
    for path in sorted(Path(data_dir).rglob("*.nii*")):
        if m := _NAME.fullmatch(path.name):
            key = (m["kind"], _int(m["n"]), _int(m["k"]))
            files = cases.setdefault(m["case"], {})
            if key in files and (files[key].parent.name == "original") == (path.parent.name == "original"):
                raise ValueError(f"two files for {m['case']} {key}: {files[key]} and {path}")
            if key not in files or path.parent.name == "original":
                files[key] = path
    return {case: files for case, files in cases.items() if any(n is not None for _, n, _ in files)}


def load_case(case: str, files: dict[Key, Path], modality: str | None = None, region: str = "WT") -> PatientData:
    """Load one case of :func:`index_cases`, in BraSyn space, for one imputed modality (inferred if unique)
    and tumour region."""
    imputed_mods = sorted({kind for kind, n, _ in files if n is not None})
    modality = modality or (imputed_mods[0] if len(imputed_mods) == 1 else None)
    if modality not in imputed_mods:
        raise ValueError(f"{case}: imputed modalities {imputed_mods}, set the modality to one of them")
    runs = sorted(n for kind, n, k in files if kind == modality and n is not None and k is None)
    prompts = sorted({k for kind, _, k in files if kind == modality and k is not None})
    if not prompts:
        raise FileNotFoundError(f"{case}: no segmentations {case}-{modality}-runNN-promptKK.nii.gz")
    missing = [(n, k) for n in runs for k in prompts if (modality, n, k) not in files]
    if missing:
        raise ValueError(f"{case}: incomplete runs x prompts design, missing (run, prompt) = {missing[:5]}")

    labels = _read_opt(files, ("seg", None, None))
    gt_image = _read_opt(files, (modality, None, None))
    first, affine = _load(files[(modality, runs[0], None)])
    imputed = np.stack([first, *(_read(files[(modality, n, None)]) for n in runs[1:])])
    seg_probs = np.empty((len(runs), len(prompts), *imputed.shape[1:]), dtype=np.float32)
    for i, n in enumerate(runs):
        for j, k in enumerate(prompts):
            seg_probs[i, j] = _read(files[(modality, n, k)])
    brain = _read_opt(files, ("mask", None, None))
    if brain is None:
        brain = (gt_image if gt_image is not None else imputed.mean(0)) > 0
    return PatientData(
        patient_id=case,
        imputed=imputed,
        seg_probs=np.clip(seg_probs, 0, 1, out=seg_probs),
        brain_mask=brain,
        gt_image=None if gt_image is None else toGrayScale(gt_image).astype(np.float32),
        gt_mask=None if labels is None else brats_region_mask(labels, region),
        region=region,
        affine=affine,
    ).validate()


def brats_region_mask(labels: np.ndarray, region: str) -> np.ndarray:
    """Binary mask of a BraTS region from a label map (a binary mask passes through).

    Labels 2023 (1 NCR, 2 SNFH, 3 ET; BraTS-Africa), or legacy when label 4 is present (4 = ET).
    ET = enhancing; TC = NCR + ET; WT = all labels.
    """
    labels = np.rint(labels).astype(np.int16)
    if labels.max() <= 1:
        return labels > 0
    et = 4 if (labels == 4).any() else 3
    masks = {"ET": labels == et, "TC": (labels == 1) | (labels == et), "WT": labels > 0}
    if region.upper() not in masks:
        raise ValueError(f"unknown region {region!r}")
    return masks[region.upper()]


def save_nifti(volume: np.ndarray, path: str | Path, affine: np.ndarray | None = None) -> None:
    """Write a 3D array as NIfTI (bool -> uint8)."""
    if volume.dtype == bool:
        volume = volume.astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume, np.eye(4) if affine is None else affine), str(path))


def _int(s: str | None) -> int | None:
    return None if s is None else int(s)


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """BraSyn's dataloader transform (reorient to IPL, then crop); returns the volume and its affine."""
    image = getBetterOrientation(nib.load(str(path)), "IPL")
    shift = np.eye(4)
    shift[:3, 3] = [s.start for s in CROP]
    return image.get_fdata(dtype=np.float32)[CROP], image.affine @ shift


def _read(path: Path) -> np.ndarray:
    return _load(path)[0]


def _read_opt(files: dict[Key, Path], key: Key) -> np.ndarray | None:
    return _read(files[key]) if key in files else None
