import numpy as np
import pytest

from trust_mri_eval.uncertainty.anova import anova_maps, anova_scalar, anova_two_way
from trust_mri_eval.uncertainty.confidence_maps import agreement_map, binary_entropy, signed_distance


def _crossed(n, k, v, s_a, s_b, s_ab, seed=0):
    rng = np.random.default_rng(seed)
    return (0.3 + s_a * rng.standard_normal((n, 1, v)) + s_b * rng.standard_normal((1, k, v))
            + s_ab * rng.standard_normal((n, k, v)))


def test_anova_recovers_injected_variances():
    s_a, s_b, s_ab = 0.2, 0.1, 0.05
    res = anova_two_way(_crossed(5, 5, 40_000, s_a, s_b, s_ab), clip_negative=False)
    # unbiased estimators: averages over voxels match the injected variances
    assert res["var_imputation"].mean() == pytest.approx(s_a ** 2, rel=0.03)
    assert res["var_prompt"].mean() == pytest.approx(s_b ** 2, rel=0.03)
    assert res["var_interaction"].mean() == pytest.approx(s_ab ** 2, rel=0.03)


def test_anova_sum_identities():
    res = anova_two_way(_crossed(4, 3, 1000, 0.2, 0.1, 0.05, seed=1))
    np.testing.assert_allclose(res["ss_imputation"] + res["ss_prompt"] + res["ss_interaction"], res["ss_total"])
    np.testing.assert_allclose(res["var_imputation"] + res["var_prompt"] + res["var_interaction"], res["var_total"])
    fsum = res["frac_imputation"] + res["frac_prompt"] + res["frac_interaction"]
    np.testing.assert_allclose(fsum[res["var_total"] > 0], 1.0)
    assert all((res[f"var_{c}"] >= 0).all() for c in ("imputation", "prompt", "interaction"))


def test_anova_pure_effects_and_constant():
    n, k = 4, 3
    only_a = np.repeat(np.arange(n, dtype=float)[:, None], k, 1)            # varies with n only
    r = anova_scalar(only_a)
    assert r["frac_imputation"] == pytest.approx(1) and r["var_imputation"] == pytest.approx(np.var(np.arange(n), ddof=1))
    only_b = np.repeat(np.arange(k, dtype=float)[None, :], n, 0)
    assert anova_scalar(only_b)["frac_prompt"] == pytest.approx(1)
    const = anova_scalar(np.ones((n, k)))
    assert const["var_total"] == 0 and const["frac_imputation"] == 0


def test_anova_k1_reduces_to_imputation_factor():
    vals = np.arange(5, dtype=float)[:, None]
    r = anova_scalar(vals)
    assert r["var_imputation"] == pytest.approx(np.var(np.arange(5), ddof=1))
    assert r["var_prompt"] == 0 and r["var_interaction"] == 0 and r["frac_imputation"] == 1


def test_anova_maps_chunked_matches_direct():
    vals = _crossed(3, 4, 6 * 7 * 8, 0.2, 0.1, 0.05).reshape(3, 4, 6, 7, 8).astype(np.float32)
    mask = np.zeros((6, 7, 8), bool)
    mask[1:5, 2:6, :] = True
    maps = anova_maps(vals, mask, chunk=17)
    direct = anova_two_way(vals)
    np.testing.assert_allclose(maps["frac_prompt"][mask], direct["frac_prompt"][mask], rtol=1e-5)
    assert (maps["var_total"][~mask] == 0).all()


def test_confidence_maps():
    masks = np.zeros((2, 2, 5, 5, 5), bool)
    masks[:, :, 2, 2, 2] = True
    masks[0, 0, 1, 1, 1] = True
    agr = agreement_map(masks)
    assert agr[2, 2, 2] == 1 and agr[1, 1, 1] == 0.25 and agr[0, 0, 0] == 0
    h = binary_entropy(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(h, [0, 1, 0], atol=1e-6)
    m = np.zeros((9, 9, 9), bool)
    m[3:6, 3:6, 3:6] = True
    sd = signed_distance(m)
    assert sd[4, 4, 4] < 0 and sd[0, 0, 0] > 0 and sd[3, 4, 4] == -1 and sd[2, 4, 4] == 1
