import numpy as np

from trust_mri_eval.data.io import load_patient, save_nifti
from trust_mri_eval.data.synthetic import generate_patient


def test_synthetic_patient(small_patient, small_cfg):
    p = small_patient
    assert p.imputed.shape == (small_cfg.n_imputations, *small_cfg.volume_shape)
    assert p.seg_probs.shape == (small_cfg.n_imputations, small_cfg.n_prompts, *small_cfg.volume_shape)
    gt, brain = p.gt_mask.astype(bool), p.brain_mask.astype(bool)
    assert p.gt_image[gt].mean() > p.gt_image[brain & ~gt].mean()          # tumour is brighter
    consensus = p.seg_probs.mean((0, 1)) >= 0.5
    assert 2 * (consensus & gt).sum() / (consensus.sum() + gt.sum()) > 0.5  # segmentations overlap it


def test_synthetic_deterministic(small_cfg):
    np.testing.assert_array_equal(generate_patient(1, small_cfg).seg_probs, generate_patient(1, small_cfg).seg_probs)


def test_nifti_roundtrip(small_patient, tmp_path):
    p = small_patient
    for n in range(p.n_imputations):
        save_nifti(p.imputed[n], tmp_path / f"imputed_n{n}.nii.gz")
        for k in range(p.n_prompts):
            save_nifti(p.seg_probs[n, k], tmp_path / f"seg_ET_n{n}_k{k}.nii.gz")
    save_nifti(p.brain_mask, tmp_path / "brain_mask.nii.gz")
    save_nifti(p.gt_mask, tmp_path / "gt_mask.nii.gz")
    q = load_patient(tmp_path)
    np.testing.assert_allclose(q.seg_probs, p.seg_probs)
    np.testing.assert_array_equal(q.gt_mask, p.gt_mask)
