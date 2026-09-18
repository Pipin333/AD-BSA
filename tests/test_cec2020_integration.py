"""
================================================================================
  AD-BSA Test Suite: CEC 2020 50D Integration Tests
================================================================================
"""

import sys
import numpy as np
from opfunu.cec_based import cec2020

sys.path.insert(0, "src")
from ad_bsa import AD_BSA, EvaluatorWrapper


def test_cec2020_f1_50d_run():
    f_class = getattr(cec2020, "F12020")
    f_inst = f_class(ndim=50)

    evaluator = EvaluatorWrapper(f_inst.evaluate, f_bias=getattr(f_inst, "f_bias", 0.0))
    bounds = np.column_stack([f_inst.lb, f_inst.ub])

    opt = AD_BSA(evaluator, bounds=bounds, max_evaluations=1000, seed=42)
    res = opt.optimize()

    assert np.isfinite(res.best_fitness)
    assert res.best_position.shape == (50,)
    assert res.total_evaluations <= 1100
