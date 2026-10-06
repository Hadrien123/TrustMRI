import numpy as np

from trust_mri_eval.preprocessing import prepare, robust_rescale


def test_robust_rescale():
    vol = np.random.default_rng(0).normal(-3, 2, (20, 20, 20)).astype(np.float32)
    mask = np.zeros(vol.shape, bool)
    mask[2:-2, 2:-2, 2:-2] = True
    out = robust_rescale(vol, mask)
    assert out.min() == 0 and out.max() == 1 and (out[~mask] == 0).all()


def test_prepare(small_patient):
    prep = prepare(small_patient)
    assert prep.brain.sum() == small_patient.brain_mask.sum()
    assert prep.roi[prep.gt_mask].all() and prep.roi[prep.masks.any((0, 1))].all()
    np.testing.assert_array_equal(prep.uncrop(prep.gt_mask), small_patient.gt_mask.astype(bool))
