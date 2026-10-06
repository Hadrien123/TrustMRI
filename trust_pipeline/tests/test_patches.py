import numpy as np

from trust_mri_eval.patches import PatchGrid, bounding_box


def test_grid_sum_mean_expand():
    vol = np.arange(7 * 6 * 5, dtype=np.float32).reshape(7, 6, 5)
    g = PatchGrid.over(vol.shape, (3, 3, 3))
    assert g.n == (3, 2, 2)
    m = g.mean(vol)
    assert m[0, 0, 0] == vol[:3, :3, :3].mean()
    assert m[2, 1, 1] == vol[6:, 3:, 3:].mean()          # partial patch: padding ignored
    assert g.coverage(np.ones_like(vol))[2, 1, 1] == np.float32(1 * 3 * 2 / 27)
    up = g.expand(m)
    assert up.shape == vol.shape and up[1, 1, 1] == m[0, 0, 0] and up[6, 5, 4] == m[2, 1, 1]


def test_grid_over_box_is_aligned():
    mask = np.zeros((20, 20, 20), bool)
    mask[7:12, 4:6, 10:11] = True
    g = PatchGrid.over(mask.shape, (3, 3, 3), bounding_box(mask))
    assert g.start == (6, 3, 9) and g.n == (2, 1, 1)
    # parent-shaped and box-shaped inputs give the same result
    np.testing.assert_array_equal(g.sum(mask), g.sum(g.crop(mask)))
    assert g.sum(mask).sum() == mask.sum()


def test_mean_with_weights_nan_when_empty():
    vol = np.ones((6, 6, 6), np.float32)
    w = np.zeros_like(vol)
    w[:3, :3, :3] = 1
    m = PatchGrid.over(vol.shape, (3, 3, 3)).mean(vol, w)
    assert m[0, 0, 0] == 1 and np.isnan(m[1, 1, 1])
