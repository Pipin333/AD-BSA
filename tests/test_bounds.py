"""
================================================================================
  AD-BSA Test Suite: Boundary Handling & Reflection Tests
================================================================================
"""

import sys
import numpy as np

sys.path.insert(0, "src")
from ad_bsa import (
    reflect_boundaries,
    bound_constraint_shade,
    bound_constraint_clamp,
)


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


def test_bound_constraint_shade_canonical_rule():
    lb = np.array([-10.0, -10.0])
    ub = np.array([10.0, 10.0])
    # candidato con violaciones: [-15.0, 25.0]
    # base: [0.0, 2.0]
    # esperado:
    # componente 0 (< -10.0): (lb[0] + base[0]) / 2 = (-10 + 0) / 2 = -5.0
    # componente 1 (> 10.0): (ub[1] + base[1]) / 2 = (10 + 2) / 2 = 6.0
    candidates = np.array([[-15.0, 25.0]])
    base = np.array([[0.0, 2.0]])

    fixed = bound_constraint_shade(candidates, lb, ub, base)
    assert np.allclose(fixed, [[-5.0, 6.0]])


def test_bound_constraint_clamp():
    lb = np.array([-10.0, -10.0])
    ub = np.array([10.0, 10.0])
    candidates = np.array([[-15.0, 25.0], [5.0, -2.0]])

    clamped = bound_constraint_clamp(candidates, lb, ub)
    assert np.allclose(clamped, [[-10.0, 10.0], [5.0, -2.0]])
