"""
================================================================================
  AD-BSA: Adaptive Differential Boogeyman Search Algorithm
  Module: src/ad_bsa/utils.py
  Utilities: Boundary constraints, evaluation counters, statistical analysis
================================================================================
"""

from typing import Callable, Tuple
import numpy as np
from scipy import stats


def reflect_boundaries(
    candidates: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    base: np.ndarray
) -> np.ndarray:
    """
    Refleja los candidatos fuera de los límites del espacio de búsqueda.
    Si la reflexión directa excede el rango opuesto, interpola a la mitad del camino hacia la base.
    """
    fixed = candidates.copy()

    # Violación de límite inferior
    lower_mask = fixed < lb
    fixed[lower_mask] = 2.0 * lb[np.newaxis, :].repeat(len(fixed), axis=0)[lower_mask] - candidates[lower_mask]
    re_violate_low = (fixed < lb) | (fixed > ub)
    if np.any(re_violate_low):
        fixed[re_violate_low] = 0.5 * (base[re_violate_low] + lb[np.newaxis, :].repeat(len(fixed), axis=0)[re_violate_low])

    # Violación de límite superior
    upper_mask = fixed > ub
    fixed[upper_mask] = 2.0 * ub[np.newaxis, :].repeat(len(fixed), axis=0)[upper_mask] - candidates[upper_mask]
    re_violate_high = (fixed < lb) | (fixed > ub)
    if np.any(re_violate_high):
        fixed[re_violate_high] = 0.5 * (base[re_violate_high] + ub[np.newaxis, :].repeat(len(fixed), axis=0)[re_violate_high])

    return np.clip(fixed, lb, ub)


def bound_constraint_shade(
    candidates: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    base: np.ndarray
) -> np.ndarray:
    """
    Regla canónica de manejo de restricciones de frontera en SHADE, L-SHADE y jSO:
      - Si v_{i,j} < lb_j: v_{i,j} = (lb_j + base_{i,j}) / 2.0
      - Si v_{i,j} > ub_j: v_{i,j} = (ub_j + base_{i,j}) / 2.0
    Garantiza que las soluciones mutadas permanezcan dentro de la región factible
    interpolarizando a mitad de camino hacia la posición parental.
    """
    fixed = candidates.copy()
    low_mask = fixed < lb
    high_mask = fixed > ub

    lb_mat = np.broadcast_to(lb, fixed.shape)
    ub_mat = np.broadcast_to(ub, fixed.shape)

    fixed[low_mask] = 0.5 * (lb_mat[low_mask] + base[low_mask])
    fixed[high_mask] = 0.5 * (ub_mat[high_mask] + base[high_mask])

    return np.clip(fixed, lb, ub)


def bound_constraint_clamp(
    candidates: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray
) -> np.ndarray:
    """
    Regla canónica de truncamiento directo a límites (Simple Bounds Clamping)
    usada tradicionalmente en DE estándar y Cuckoo Search.
    """
    return np.clip(candidates, lb, ub)


class EvaluatorWrapper:
    """
    Wrapper para funciones objetivo continuas que contabiliza evaluaciones
    y maneja tanto entradas 1D (vector individual) como 2D (población completa).
    """

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], f_bias: float = 0.0):
        self.func = func
        self.f_bias = f_bias
        self.evaluations = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            self.evaluations += 1
            return float(self.func(x))
        elif x.ndim == 2:
            batch_size = x.shape[0]
            # Intentar vectorizado si la función objetivo admite arrays 2D y retorna array de tamaño batch_size
            try:
                res = np.asarray(self.func(x), dtype=np.float64)
                if res.ndim == 1 and len(res) == batch_size:
                    self.evaluations += batch_size
                    return res
            except Exception:
                pass

            # Evaluación iterativa fila por fila
            self.evaluations += batch_size
            return np.array([float(self.func(ind)) for ind in x], dtype=np.float64)
        else:
            raise ValueError(f"Dimensión de entrada inválida: {x.ndim}. Debe ser 1D o 2D.")



def compute_wilcoxon(scores_a: list, scores_b: list, alpha: float = 0.05) -> Tuple[float, str]:
    """
    Calcula el test de rangos signados de Wilcoxon / Mann-Whitney U entre dos algoritmos.
    Retorna (p_value, sign), donde sign es '+', '-', o '='.
    """
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)

    if np.allclose(arr_a, arr_b, atol=1e-12):
        return 1.0, "="

    try:
        stat, p_val = stats.wilcoxon(arr_a, arr_b, alternative="two-sided")
    except Exception:
        stat, p_val = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")

    med_a = float(np.median(arr_a))
    med_b = float(np.median(arr_b))

    if p_val < alpha:
        if med_a < med_b:
            return float(p_val), "+"
        else:
            return float(p_val), "-"
    return float(p_val), "="
