"""
================================================================================
  AD-BSA Test Suite: Boundary Handling & Reflection Tests
================================================================================
"""

import sys
import numpy as np

sys.path.insert(0, "src")
from ad_bsa import reflect_boundaries


def test_reflect_boundaries_within():
    lb = np.array([-10.0, -10.0])
    ub = np.array([10.0, 10.0])
    candidates = np.array([[0.0, 5.0], [-5.0, 2.0]])
    base = np.zeros_like(candidates)

    reflected = reflect_boundaries(candidates, lb, ub, base)
    assert np.allclose(reflected, candidates)


def test_reflect_boundaries_violation():
    lb = np.array([-10.0, -10.0])
    ub = np.array([10.0, 10.0])
    candidates = np.array([[-15.0, 25.0], [50.0, -100.0]])
    base = np.zeros_like(candidates)

    reflected = reflect_boundaries(candidates, lb, ub, base)
    assert np.all(reflected >= lb)
    assert np.all(reflected <= ub)
