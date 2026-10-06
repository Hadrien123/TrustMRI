import numpy as np

from trust_mri_eval.patches import PatchGrid


def test_grid_mean_and_expand():
    vol = np.arange(7 * 6 * 5, dtype=np.float32).reshape(7, 6, 5)
    grid = PatchGrid.over(vol.shape, (3, 3, 3))
    m = grid.mean(vol)
    assert grid.n == (3, 2, 2) and m[0, 0, 0] == vol[:3, :3, :3].mean()
    assert m[2, 1, 1] == vol[6:, 3:, 3:].mean()                    # partial patch: padding ignored
    assert grid.expand(m).shape == vol.shape
