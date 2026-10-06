import numpy as np
import pytest
from skimage.metrics import structural_similarity

from trust_mri_eval.metrics.distribution import (interval_coverage, mean_variance_maps, pairwise_psnr,
                                                 pairwise_ssim, spread_error_correlation)
from trust_mri_eval.metrics.image_global import image_global_metrics, mean_std, psnr
from trust_mri_eval.metrics.image_local import (failure_map, flag_threshold, patch_maps, patch_regions,
                                                sample_variance, summarize_patch_map, voxel_error_maps)
from trust_mri_eval.metrics.ssim import ssim_map
from trust_mri_eval.patches import PatchGrid


@pytest.fixture
def images():
    rng = np.random.default_rng(0)
    gt = rng.random((24, 24, 24)).astype(np.float32)
    brain = np.zeros(gt.shape, bool)
    brain[3:-3, 3:-3, 3:-3] = True
    tumour = np.zeros(gt.shape, bool)
    tumour[9:15, 9:15, 9:15] = True          # aligned to the 3-voxel patch grid
    return gt, brain, tumour


def test_ssim_map_matches_skimage(images):
    gt, _, _ = images
    other = np.clip(gt + 0.1 * np.random.default_rng(1).standard_normal(gt.shape), 0, 1).astype(np.float32)
    _, ref = structural_similarity(gt.astype(np.float64), other.astype(np.float64), data_range=1.0,
                                   win_size=7, full=True)
    np.testing.assert_allclose(ssim_map(gt, other), ref, atol=1e-4)


def test_identical_images(images):
    gt, brain, tumour = images
    samples = np.stack([gt, gt, gt])
    m = mean_std(image_global_metrics(gt, samples, brain, tumour))
    assert m["l1_mean"] == 0 and m["l2_mean"] == 0 and m["psnr_mean"] == np.inf
    for r in ("brain", "tumour", "healthy"):
        assert m[f"ssim_{r}_mean"] == pytest.approx(1.0, abs=1e-5)
    assert sample_variance(samples).max() == 0
    values, local = pairwise_ssim(samples, brain)
    assert len(values) == 3 and np.allclose(values, 1, atol=1e-5) and np.allclose(local[brain], 1, atol=1e-5)
    assert all(np.isinf(pairwise_psnr(samples, brain)))
    assert interval_coverage(samples, gt, brain) == 1.0


def test_errors_are_restricted_to_brain(images):
    gt, brain, tumour = images
    noisy = gt.copy()
    noisy[~brain] += 5                       # background errors must not count
    m = image_global_metrics(gt, noisy[None], brain, tumour)
    assert m["l1"][0] == 0 and np.isinf(m["psnr"][0])


def test_psnr_value():
    a = np.zeros((4, 4, 4), np.float32)
    b = a + 0.1
    assert psnr(a, b, np.ones_like(a, bool)) == pytest.approx(20.0, abs=1e-4)


def test_local_maps_and_failure(images):
    gt, brain, tumour = images
    bad = gt.copy()
    bad[tumour] = 1 - bad[tumour]            # large error in the tumour only
    samples = np.stack([bad, bad])
    grid = PatchGrid.over(gt.shape, (3, 3, 3))
    pm = patch_maps(voxel_error_maps(gt, samples), grid, brain)
    regions = patch_regions(grid, brain, tumour)
    assert np.nanmean(pm["l1"][regions["healthy"]]) == 0
    thr = flag_threshold([pm["l1"][regions["brain"]]], 90)
    fail = failure_map(pm["l1"], thr)
    assert fail.dtype == np.uint8 and fail[regions["tumour"]].mean() > 0.5 and fail[regions["healthy"]].sum() == 0
    s = summarize_patch_map(pm["l1"], regions, thr)
    assert s["healthy_frac_above_thr"] == 0 and s["tumour_mean"] > s["healthy_mean"]
    assert grid.expand(fail.astype(np.float32)).shape == gt.shape


def test_mean_variance_and_spread_calibration():
    rng = np.random.default_rng(0)
    truth = rng.random((10, 10, 10)).astype(np.float32)
    scale = np.linspace(0.01, 0.3, 10, dtype=np.float32)[:, None, None] * np.ones_like(truth)
    samples = truth + scale * rng.standard_normal((50, *truth.shape)).astype(np.float32)
    mean, var = mean_variance_maps(samples)
    assert var.shape == truth.shape and np.allclose(mean, truth, atol=0.15)
    mask = np.ones_like(truth, bool)
    observed = truth + scale * rng.standard_normal(truth.shape).astype(np.float32)  # one more draw
    cov = interval_coverage(samples, observed, mask)
    assert 0.8 < cov < 1.0                          # calibrated spread: ~90 % nominal coverage
    err = np.abs(truth + scale * rng.standard_normal(truth.shape) - truth)
    assert spread_error_correlation(np.sqrt(var), err) > 0.3
