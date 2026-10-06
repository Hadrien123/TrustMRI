import numpy as np
import pytest
from skimage.metrics import structural_similarity

from trust_mri_eval.metrics.distribution import interval_coverage, pairwise_ssim
from trust_mri_eval.metrics.image_global import image_global_metrics
from trust_mri_eval.metrics.image_local import sample_variance
from trust_mri_eval.metrics.ssim import ssim_map

GT = np.random.default_rng(0).random((24, 24, 24)).astype(np.float32)
BRAIN = np.zeros(GT.shape, bool)
BRAIN[3:-3, 3:-3, 3:-3] = True
TUMOUR = np.zeros(GT.shape, bool)
TUMOUR[9:15, 9:15, 9:15] = True


def test_ssim_map_matches_skimage():
    other = np.clip(GT + 0.1 * np.random.default_rng(1).standard_normal(GT.shape), 0, 1).astype(np.float32)
    _, ref = structural_similarity(GT.astype(float), other.astype(float), data_range=1.0, win_size=7, full=True)
    np.testing.assert_allclose(ssim_map(GT, other), ref, atol=1e-4)


def test_identical_images():
    samples = np.stack([GT, GT, GT])
    m = image_global_metrics(GT, samples, BRAIN, TUMOUR)
    assert max(m["l1"]) == 0 and max(m["l2"]) == 0
    assert min(m["ssim_brain"]) == pytest.approx(1.0, abs=1e-5)
    assert sample_variance(samples).max() == 0
    assert np.allclose(pairwise_ssim(samples, BRAIN)[0], 1, atol=1e-5)
    assert interval_coverage(samples, GT, BRAIN) == 1.0
