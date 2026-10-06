"""AD-BSA Utility Functions and Helper Classes.

Provides boundary constraint handlers (reflective, SHADE midpoint, clamp),
objective function evaluation wrapper, and non-parametric statistical tests.
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
    """Reflect candidate vectors that violate search space boundaries.

    Implements adaptive reflective boundary handling. If the direct reflection
    exceeds the opposite bound, the coordinate is projected to the midpoint
    between the base parental position and the violated boundary.

    Args:
        candidates: Candidate population array of shape (N, D).
        lb: 1D array of lower bounds of length D.
        ub: 1D array of upper bounds of length D.
        base: 1D or 2D array of parental base coordinates of shape (N, D).

    Returns:
        np.ndarray: Feasible population array of shape (N, D) clipped to [lb, ub].
    """
    fixed = candidates.copy()

    lb_expanded = np.broadcast_to(lb, fixed.shape)
    ub_expanded = np.broadcast_to(ub, fixed.shape)

    # Lower bound violation
    lower_mask = fixed < lb
    fixed[lower_mask] = 2.0 * lb_expanded[lower_mask] - candidates[lower_mask]
    re_violate_low = (fixed < lb) | (fixed > ub)
    if np.any(re_violate_low):
        fixed[re_violate_low] = 0.5 * (
            base[re_violate_low] + lb_expanded[re_violate_low]
        )

    # Upper bound violation
    upper_mask = fixed > ub
    fixed[upper_mask] = 2.0 * ub_expanded[upper_mask] - candidates[upper_mask]
    re_violate_high = (fixed < lb) | (fixed > ub)
    if np.any(re_violate_high):
        fixed[re_violate_high] = 0.5 * (
            base[re_violate_high] + ub_expanded[re_violate_high]
        )

    return np.clip(fixed, lb, ub)


def bound_constraint_shade(
    candidates: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    base: np.ndarray
) -> np.ndarray:
    """Apply the canonical midpoint boundary constraint rule of SHADE/jSO.

    If a trial coordinate violates a bound:
        - v_{i,j} = (lb_j + base_{i,j}) / 2.0  if v_{i,j} < lb_j
        - v_{i,j} = (ub_j + base_{i,j}) / 2.0  if v_{i,j} > ub_j

    Args:
        candidates: Candidate population array of shape (N, D).
        lb: 1D array of lower bounds of length D.
        ub: 1D array of upper bounds of length D.
        base: Parental coordinate array of shape (N, D).

    Returns:
        np.ndarray: Feasible coordinate array bounded to [lb, ub].
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
    """Apply direct bounds clamping to candidate solutions.

    Truncates coordinates violating search limits directly to lb or ub.

    Args:
        candidates: Candidate population array of shape (N, D) or (D,).
        lb: Lower boundary vector.
        ub: Upper boundary vector.

    Returns:
        np.ndarray: Clamped coordinate array.
    """
    return np.clip(candidates, lb, ub)


class EvaluatorWrapper:
    """Objective function wrapper tracking cumulative evaluations.

    Supports both 1D individual vectors and 2D population batches,
    maintaining an internal evaluation counter.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        f_bias: float = 0.0
    ) -> None:
        """Initialize the evaluator wrapper.

        Args:
            func: Target callable objective function.
            f_bias: Theoretical optimal fitness bias offset. Defaults to 0.0.
        """
        self.func = func
        self.f_bias = float(f_bias)
        self.evaluations = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate candidate input vector or batch.

        Args:
            x: Input array of shape (D,) or (N, D).

        Returns:
            np.ndarray or float: Function evaluation value(s).

        Raises:
            ValueError: If input dimensionality is not 1D or 2D.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            self.evaluations += 1
            return float(self.func(x_arr))
        elif x_arr.ndim == 2:
            batch_size = x_arr.shape[0]
            # Attempt vectorized batch evaluation
            try:
                res = np.asarray(self.func(x_arr), dtype=np.float64)
                if res.ndim == 1 and len(res) == batch_size:
                    self.evaluations += batch_size
                    return res
            except Exception:
                pass

            # Fallback to row-by-row iteration
            self.evaluations += batch_size
            return np.array(
                [float(self.func(ind)) for ind in x_arr],
                dtype=np.float64
            )
        else:
            raise ValueError(
                f"Invalid input dimensions: {x_arr.ndim}. Must be 1D or 2D."
            )


def compute_wilcoxon(
    scores_a: list,
    scores_b: list,
    alpha: float = 0.05
) -> Tuple[float, str]:
    """Compute the two-sided Wilcoxon signed-rank test between two algorithms.

    Falls back to Mann-Whitney U test if paired sample sizes or ranks diverge.

    Args:
        scores_a: Collection of fitness error scores from algorithm A.
        scores_b: Collection of fitness error scores from algorithm B.
        alpha: Significance threshold level. Defaults to 0.05.

    Returns:
        Tuple[float, str]: Pair of (p_value, sign) where:
            '+' indicates algorithm A significantly outperforms B
                (p < alpha, med_a < med_b),
            '-' indicates algorithm B significantly outperforms A
                (p < alpha, med_b < med_a),
            '=' indicates no statistically significant difference (p >= alpha).
    """
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)

    if np.allclose(arr_a, arr_b, atol=1e-12):
        return 1.0, "="

    try:
        _, p_val = stats.wilcoxon(arr_a, arr_b, alternative="two-sided")
    except Exception:
        _, p_val = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")

    med_a = float(np.median(arr_a))
    med_b = float(np.median(arr_b))

    if p_val < alpha:
        if med_a < med_b:
            return float(p_val), "+"
        return float(p_val), "-"
    return float(p_val), "="
