"""Two-way crossed ANOVA without replication: how much of the segmentation variability comes from
the imputation (factor A, index n), the prompt (factor B, index k) and their interaction.

Model, per voxel (or per scalar such as a lesion volume)::

    M(n, k) = mu + a_n + b_k + c_nk,   n = 1..N, k = 1..K

with random effects a ~ (0, s2_A), b ~ (0, s2_B), c ~ (0, s2_AB). The segmenter is deterministic
given (image, prompt), so there is no replication and the residual *is* the interaction. Expected
mean squares give the unbiased (method-of-moments) estimators::

    E[MS_A]  = s2_AB + K s2_A   ->  s2_A  = (MS_A - MS_AB) / K
    E[MS_B]  = s2_AB + N s2_B   ->  s2_B  = (MS_B - MS_AB) / N
    E[MS_AB] = s2_AB            ->  s2_AB = MS_AB

Negative estimates are clipped to 0 by default (standard practice); fractions are each component
over their sum, so they sum to 1 wherever there is any variability (0 elsewhere).
With K = 1 only the imputation factor is identifiable (s2_A = MS_A); with N = 1 only the prompt.
"""
from __future__ import annotations

import numpy as np

COMPONENTS = ("imputation", "prompt", "interaction")


def anova_two_way(values: np.ndarray, clip_negative: bool = True, min_total: float = 0.0) -> dict[str, np.ndarray]:
    """Variance decomposition of ``values`` with shape (N, K, ...) along the first two axes.

    Returns arrays of shape ``values.shape[2:]``:
    ``var_imputation, var_prompt, var_interaction, var_total`` (sum of the three components),
    ``frac_*`` (Sobol-like fractions), and the sums of squares ``ss_imputation, ss_prompt,
    ss_interaction, ss_total`` (``ss_total`` = sum of the other three, exactly).
    Fractions are 0 where ``var_total <= min_total`` (no meaningful variability to split).
    """
    y = np.asarray(values, dtype=np.float64)
    n, k = y.shape[:2]
    grand = y.mean(axis=(0, 1))
    row = y.mean(axis=1) - grand                 # a_n, shape (N, ...)
    col = y.mean(axis=0) - grand                 # b_k, shape (K, ...)
    resid = y - grand - row[:, None] - col[None, :]

    ss_a = k * (row ** 2).sum(0)
    ss_b = n * (col ** 2).sum(0)
    ss_ab = (resid ** 2).sum((0, 1))
    ss_tot = ((y - grand) ** 2).sum((0, 1))

    zeros = np.zeros_like(grand)
    if n > 1 and k > 1:
        ms_a, ms_b, ms_ab = ss_a / (n - 1), ss_b / (k - 1), ss_ab / ((n - 1) * (k - 1))
        var_a, var_b, var_ab = (ms_a - ms_ab) / k, (ms_b - ms_ab) / n, ms_ab
    elif n > 1:                                  # K = 1: imputation only
        var_a, var_b, var_ab = ss_a / (n - 1), zeros, zeros
    elif k > 1:                                  # N = 1: prompt only
        var_a, var_b, var_ab = zeros, ss_b / (k - 1), zeros
    else:
        var_a = var_b = var_ab = zeros
    if clip_negative:
        var_a, var_b, var_ab = (np.maximum(v, 0.0) for v in (var_a, var_b, var_ab))

    total = var_a + var_b + var_ab
    has_var = total > min_total
    safe = np.where(has_var, total, 1.0)
    out = {"var_imputation": var_a, "var_prompt": var_b, "var_interaction": var_ab, "var_total": total,
           "ss_imputation": ss_a, "ss_prompt": ss_b, "ss_interaction": ss_ab, "ss_total": ss_tot}
    for name, v in zip(COMPONENTS, (var_a, var_b, var_ab)):
        out[f"frac_{name}"] = np.where(has_var, v / safe, 0.0)
    return out


def anova_maps(values: np.ndarray, mask: np.ndarray, chunk: int = 200_000, min_total: float = 1e-4,
               keys: tuple[str, ...] = ("var_imputation", "var_prompt", "var_interaction", "var_total",
                                        "frac_imputation", "frac_prompt", "frac_interaction")) -> dict[str, np.ndarray]:
    """Voxel-wise ANOVA of ``values`` (N, K, *S) computed only inside ``mask``, in chunks of voxels.

    Returns float32 maps of shape S (0 outside ``mask``) for the requested ``keys``. Fractions are
    0 where the total variance is <= ``min_total`` (default 1e-4, for probabilities whose variance is
    at most 0.25: avoids splitting numerical noise far from any boundary).
    """
    spatial = values.shape[2:]
    flat = values.reshape(*values.shape[:2], -1)
    idx = np.flatnonzero(mask.ravel())
    out = {key: np.zeros(int(np.prod(spatial)), dtype=np.float32) for key in keys}
    for start in range(0, idx.size, chunk):
        sel = idx[start:start + chunk]
        res = anova_two_way(flat[:, :, sel], min_total=min_total)
        for key in keys:
            out[key][sel] = res[key]
    return {key: v.reshape(spatial) for key, v in out.items()}


def anova_scalar(values: np.ndarray) -> dict[str, float]:
    """ANOVA of one scalar per (n, k), e.g. the predicted lesion volume. ``values`` is (N, K)."""
    res = anova_two_way(np.asarray(values, dtype=np.float64)[:, :, None])
    return {key: float(v[0]) for key, v in res.items()}
