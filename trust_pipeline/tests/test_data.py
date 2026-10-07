import numpy as np

from conftest import make_case
from trust_mri_eval.data.io import index_cases, load_case


def test_load_case_in_brasyn_space(data_dir):
    cases = index_cases(data_dir)
    assert sorted(cases) == [f"case{i}" for i in range(4)]
    p, raw = load_case("case0", cases["case0"]), make_case(0)
    assert p.imputed.shape == (3, 40, 40, 40)                       # IPL, BraSyn crop
    assert p.gt_mask.sum() == raw.gt_mask.sum()                      # the whole tumour is inside the crop
    np.testing.assert_array_equal(p.seg_probs.sum((2, 3, 4)), raw.seg_probs.sum((2, 3, 4)))
    assert p.gt_image.min() == 0 and p.gt_image.max() == 1           # toGrayScale
