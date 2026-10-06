import numpy as np
import pytest

from trust_mri_eval.data.io import brats_region_mask, load_patient, save_nifti
from trust_mri_eval.data.synthetic import generate_patient
from trust_mri_eval.data.types import PatientData


def test_synthetic_shapes_and_dtypes(small_patient, small_cfg):
    p = small_patient
    n, k, s = small_cfg.n_imputations, small_cfg.n_prompts, small_cfg.volume_shape
    assert p.imputed.shape == (n, *s) and p.imputed.dtype == np.float32
    assert p.seg_probs.shape == (n, k, *s) and p.seg_probs.dtype == np.float32
    assert p.brain_mask.dtype == np.uint8 and p.gt_mask.dtype == np.uint8
    assert 0 <= p.seg_probs.min() and p.seg_probs.max() <= 1
    assert p.gt_mask.sum() > 0 and not (p.gt_mask & ~p.brain_mask.astype(bool)).any()
    assert (p.imputed[:, p.brain_mask == 0] == 0).all()


def test_synthetic_is_meaningful(small_patient):
    p = small_patient
    gt, brain = p.gt_mask.astype(bool), p.brain_mask.astype(bool)
    # tumour brighter than healthy brain, samples close to but not equal to the truth
    assert p.gt_image[gt].mean() > p.gt_image[brain & ~gt].mean() + 0.1
    err = np.abs(p.imputed - p.gt_image)[:, brain].mean()
    assert 0 < err < 0.1
    # segmentations overlap the reference
    consensus = p.seg_probs.mean((0, 1)) >= 0.5
    dice = 2 * (consensus & gt).sum() / (consensus.sum() + gt.sum())
    assert dice > 0.5
    # prompts produce different masks
    assert not np.array_equal(p.seg_probs[0, 0] > 0.5, p.seg_probs[0, -1] > 0.5)


def test_synthetic_deterministic(small_cfg):
    a, b = generate_patient(1, small_cfg), generate_patient(1, small_cfg)
    np.testing.assert_array_equal(a.imputed, b.imputed)
    np.testing.assert_array_equal(a.seg_probs, b.seg_probs)


def test_validate_rejects_bad_shapes(small_patient):
    with pytest.raises(ValueError):
        PatientData("x", small_patient.imputed, small_patient.seg_probs[:, :, :10],
                    small_patient.brain_mask).validate()


def test_brats_regions():
    lab = np.array([0, 1, 2, 3])
    assert brats_region_mask(lab, "ET").tolist() == [False, False, False, True]
    assert brats_region_mask(lab, "TC").tolist() == [False, True, False, True]
    assert brats_region_mask(lab, "WT").tolist() == [False, True, True, True]
    legacy = np.array([0, 1, 2, 4])
    assert brats_region_mask(legacy, "ET").tolist() == [False, False, False, True]


def test_nifti_roundtrip(small_patient, tmp_path):
    p = small_patient
    d = tmp_path / "pat01"
    for n in range(p.n_imputations):
        save_nifti(p.imputed[n], d / f"imputed_n{n:02d}.nii.gz")
        for k in range(p.n_prompts):
            save_nifti(p.seg_probs[n, k], d / f"seg_ET_n{n}_k{k}.nii.gz")
    save_nifti(p.brain_mask, d / "brain_mask.nii.gz")
    save_nifti(p.gt_image, d / "gt_image.nii.gz")
    save_nifti((p.gt_mask * 3).astype(np.uint8), d / "gt_mask.nii.gz")  # BraTS 2023 ET label
    q = load_patient(d, region="ET")
    assert q.patient_id == "pat01"
    np.testing.assert_allclose(q.imputed, p.imputed)
    np.testing.assert_allclose(q.seg_probs, p.seg_probs)
    np.testing.assert_array_equal(q.gt_mask, p.gt_mask)
    assert q.real_flair is None and q.spacing == (1.0, 1.0, 1.0)
