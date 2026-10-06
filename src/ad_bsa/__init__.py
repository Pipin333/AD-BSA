"""AD-BSA: Adaptive Differential Boogeyman Search Algorithm.

A continuous global metaheuristic package combining bounded cosecant
repulsion barriers, multi-boogeyman anti-attractors, historical Lehmer
parameter memories, and linear population size reduction.
"""

from .algorithm import AD_BSA, OptimizationResult
from .competitors import (
    jSO,
    CMA_ES,
    L_SHADE,
    CanonicalCuckooSearch,
    StandardDE,
    StandardPSO,
)
from .utils import (
    reflect_boundaries,
    bound_constraint_shade,
    bound_constraint_clamp,
    EvaluatorWrapper,
    compute_wilcoxon,
)

__version__ = "1.0.0"
__author__ = "Felipe Riquelme Salvo"
__all__ = [
    "AD_BSA",
    "OptimizationResult",
    "jSO",
    "CMA_ES",
    "L_SHADE",
    "CanonicalCuckooSearch",
    "StandardDE",
    "StandardPSO",
    "reflect_boundaries",
    "bound_constraint_shade",
    "bound_constraint_clamp",
    "EvaluatorWrapper",
    "compute_wilcoxon",
]
