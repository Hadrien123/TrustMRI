"""Shared fixtures: a small synthetic configuration so the tests run in seconds."""
from __future__ import annotations

import pytest

from trust_mri_eval.config import Config


@pytest.fixture(scope="session")
def small_cfg() -> Config:
    return Config(volume_shape=(64, 64, 48), n_patients=4, n_imputations=3, n_prompts=3,
                  syn_tumour_radius=(4.0, 8.0), split=(2, 1, 1), cv_folds=2,
                  save_maps=False, save_figures=False)


@pytest.fixture(scope="session")
def small_patient(small_cfg):
    from trust_mri_eval.data.synthetic import generate_patient
    return generate_patient(0, small_cfg)
