"""Shared fixtures: small toy cases (two spherical lesions, noisy imputations, threshold segmentations)
written in the layout of data/io.py, so the tests run in seconds. Needs BraSyn_tutorial (see trust_mri_eval)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from trust_mri_eval.config import Config
from trust_mri_eval.data.io import index_cases, load_case, save_nifti
from trust_mri_eval.data.types import PatientData


def make_case(seed: int, shape=(64, 64, 48), n: int = 3, k: int = 3) -> PatientData:
    """Prompt k shifts the segmentation threshold; prompt 0 also misses the small lesion. With an identity
    affine, BraSyn's crop keeps x, y < 40 and z < 40, where the brain is."""
    rng = np.random.default_rng(seed)
    grid = np.indices(shape)
    mid = np.array([20, 20, 20])

    def ball(center, r):
        return sum((g - c) ** 2 for g, c in zip(grid, center)) <= r * r

    brain = ball(mid, 18)
    small = ball(mid + [9, -9, 4], 3.5)
    gt_mask = ball(mid + rng.uniform(-2, 2, 3), 6) | small
    gt_image = ((0.4 + 0.6 * gt_mask) * brain).astype(np.float32)
    imputed = np.stack([ndi.gaussian_filter(gt_image + 0.1 * rng.standard_normal(shape), 1) * brain
                        for _ in range(n)]).astype(np.float32)
    seg = [[(ndi.gaussian_filter(image, 1) > 0.7 + 0.05 * (j - 1)) & brain & ~(small & (j == 0))
            for j in range(k)] for image in imputed]
    return PatientData(f"case{seed}", imputed, np.array(seg, np.float32), brain, gt_image, gt_mask,
                       region="WT").validate()


def write_case(p: PatientData, folder, modality: str = "t1c") -> None:
    c = p.patient_id
    save_nifti(p.gt_image, folder / f"{c}-{modality}.nii.gz")
    save_nifti(p.brain_mask, folder / f"{c}-mask.nii.gz")
    save_nifti(p.gt_mask, folder / f"{c}-seg.nii.gz")
    for n in range(p.n_imputations):
        save_nifti(p.imputed[n], folder / "imputed" / f"{c}-{modality}-run{n + 1:02d}.nii.gz")
        for k in range(p.n_prompts):
            save_nifti(p.seg_probs[n, k], folder / "seg" / f"{c}-{modality}-run{n + 1:02d}-prompt{k + 1:02d}.nii.gz")


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    for i in range(4):
        write_case(make_case(i), root / f"case{i}")
    return root


@pytest.fixture(scope="session")
def small_patient(data_dir) -> PatientData:
    """case0 as the pipeline sees it (BraSyn space)."""
    return load_case("case0", index_cases(data_dir)["case0"])


@pytest.fixture(scope="session")
def small_cfg(data_dir) -> Config:
    return Config(data_dir=data_dir, cv_folds=2, save_maps=False, save_figures=False)
