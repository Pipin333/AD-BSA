"""
================================================================================
  AD-BSA Test Suite: Algorithm Execution & Convergence Tests
================================================================================
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, "src")
from ad_bsa import (
    AD_BSA,
    jSO,
    CMA_ES,
    L_SHADE,
    CanonicalCuckooSearch,
    StandardDE,
    StandardPSO,
    EvaluatorWrapper,
)


def sphere(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.sum(x**2, axis=1)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2)**2 + (1.0 - x[:, :-1])**2, axis=1)


@pytest.mark.parametrize("algo_class", [
    AD_BSA,
    jSO,
    CMA_ES,
    L_SHADE,
    CanonicalCuckooSearch,
    StandardDE,
    StandardPSO,
])

def test_all_algorithms_sphere_convergence(algo_class):
    dim = 5
    bounds = np.array([[-10.0, 10.0]] * dim)
    evaluator = EvaluatorWrapper(sphere)

    optimizer = algo_class(evaluator, bounds=bounds, max_evaluations=2500, seed=42)
    res = optimizer.optimize()

    assert np.isfinite(res.best_fitness)
    assert res.best_position.shape == (dim,)
    assert res.best_fitness < 50.0


def test_ad_bsa_csc_repulsion_stability():
    """Verifica que el operador de repulsión cosecante acotada |csc| no produzca NaNs ni explosiones numéricas."""
    dim = 10
    bounds = np.array([[-100.0, 100.0]] * dim)
    evaluator = EvaluatorWrapper(rosenbrock)

    opt = AD_BSA(evaluator, bounds=bounds, max_evaluations=3000, max_repulsion_force=4.0, seed=123)
    res = opt.optimize()

    assert not np.isnan(res.best_fitness)
    assert not np.any(np.isnan(res.best_position))
    assert np.all(res.best_position >= -100.0)
    assert np.all(res.best_position <= 100.0)
    assert len(res.history_best_fitness) > 1


def test_canonical_population_and_archive_ratios():
    """Verifica que L-SHADE y jSO utilicen las fórmulas canónicas de N_init y ratios de archivo."""
    dim = 10
    bounds = np.array([[-100.0, 100.0]] * dim)
    evaluator = EvaluatorWrapper(sphere)

    # L-SHADE canónico: N_init = 18 * D = 180
    lshade = L_SHADE(evaluator, bounds=bounds, max_evaluations=50000, seed=42)
    assert lshade.N_init == 18 * dim == 180
    assert lshade.arc_rate == 1.4

    # jSO canónico: N_init = round(25 * sqrt(D) * ln(D)), arc_rate = 2.6
    import math
    expected_jso_N = int(round(25.0 * math.sqrt(dim) * math.log(dim)))
    jso_opt = jSO(evaluator, bounds=bounds, max_evaluations=50000, seed=42)
    assert jso_opt.N_init == expected_jso_N
    assert jso_opt.arc_rate == 2.6


def test_ad_bsa_pure_annealing_memory():
    """Verifica que la memoria histórica de F_escape en AD-BSA almacene el valor raw sin double damping."""
    dim = 5
    bounds = np.array([[-10.0, 10.0]] * dim)
    evaluator = EvaluatorWrapper(sphere)

    opt = AD_BSA(evaluator, bounds=bounds, max_evaluations=2000, seed=42)
    opt.optimize()

    # Tras optimizar, la memoria de Lehmer para escape no debe haber colapsado a cero
    assert np.all(opt.Memory_F_escape > 0.05)

