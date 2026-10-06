import numpy as np
import pytest

from trust_mri_eval.preprocessing import compute_roi, consensus_mask, prepare, robust_rescale


def test_robust_rescale_range_and_background():
    rng = np.random.default_rng(0)
    vol = rng.normal(-3, 2, (20, 20, 20)).astype(np.float32)   # z-scored-like, negative values
    mask = np.zeros(vol.shape, bool)
    mask[2:-2, 2:-2, 2:-2] = True
    out = robust_rescale(vol, mask)
    assert out.min() == 0 and out.max() == 1 and out.dtype == np.float32
    assert (out[~mask] == 0).all()


def test_consensus_is_mean_then_threshold():
    probs = np.zeros((2, 2, 1, 1, 3), np.float32)
    probs[:, :, 0, 0, 0] = [[1, 1], [0, 0]]      # mean 0.5 -> in
    probs[:, :, 0, 0, 1] = [[1, 0.9], [0, 0]]    # mean 0.475 -> out
    assert consensus_mask(probs).ravel().tolist() == [True, False, False]


def test_roi_contains_masks_and_dilates():
    m = np.zeros((1, 1, 21, 21, 21), bool)
    m[0, 0, 10, 10, 10] = True
    gt = np.zeros((21, 21, 21), bool)
    gt[2, 2, 2] = True
    roi = compute_roi(m, gt, dilation=5)
    assert roi[10, 10, 15] and not roi[10, 10, 16] and roi[2, 2, 7]
    assert not compute_roi(m, None, 5)[2, 2, 2]               # inference: no reference


def test_prepare_crops_to_brain(small_patient):
    prep = prepare(small_patient)
    assert prep.brain.shape == prep.consensus.shape == prep.imputed.shape[1:]
    assert prep.brain.sum() == small_patient.brain_mask.sum()
    assert 0 <= prep.imputed.min() and prep.imputed.max() <= 1
    assert prep.roi[prep.gt_mask].all() and prep.roi[prep.masks.any((0, 1))].all()
    full = prep.uncrop(prep.gt_mask)
    np.testing.assert_array_equal(full, small_patient.gt_mask.astype(bool))
