import numpy as np
import pytest

from trust_mri_eval.uncertainty.anova import anova_two_way


def crossed(n, k, v, s_a, s_b, s_ab):
    rng = np.random.default_rng(0)
    return (s_a * rng.standard_normal((n, 1, v)) + s_b * rng.standard_normal((1, k, v))
            + s_ab * rng.standard_normal((n, k, v)))


def test_anova_recovers_injected_variances():
    res = anova_two_way(crossed(5, 5, 40_000, 0.2, 0.1, 0.05), clip_negative=False)
    assert res["var_imputation"].mean() == pytest.approx(0.2 ** 2, rel=0.03)
    assert res["var_prompt"].mean() == pytest.approx(0.1 ** 2, rel=0.03)
    assert res["var_interaction"].mean() == pytest.approx(0.05 ** 2, rel=0.03)


def test_anova_sum_identity():
    res = anova_two_way(crossed(4, 3, 1000, 0.2, 0.1, 0.05))
    np.testing.assert_allclose(res["ss_imputation"] + res["ss_prompt"] + res["ss_interaction"], res["ss_total"])
    fractions = res["frac_imputation"] + res["frac_prompt"] + res["frac_interaction"]
    np.testing.assert_allclose(fractions[res["var_total"] > 0], 1.0)
