"""
================================================================================
  AD-BSA: CEC 2020 50D Benchmark Quickstart
================================================================================
"""

import os
import sys
import numpy as np
from opfunu.cec_based import cec2020

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from ad_bsa import AD_BSA, EvaluatorWrapper


# 1. Cargar función F4 de la CEC 2020 en 50 Dimensiones
# (Expanded Rosenbrock's plus Griewank's Function, f_bias = 1900.0)
f4_class = getattr(cec2020, "F42020")
f4 = f4_class(ndim=50)

evaluator = EvaluatorWrapper(f4.evaluate, f_bias=f4.f_bias)
bounds = np.column_stack([f4.lb, f4.ub])

print("Optimizando CEC 2020 F4 (50D, 50,000 evaluaciones)...")
opt = AD_BSA(
    objective_func=evaluator,
    bounds=bounds,
    max_evaluations=50000,
    seed=2026
)

res = opt.optimize()
error = res.best_fitness - f4.f_bias

print(f"\n--- RESULTADO F4 (50D) ---")
print(f"Mejor Fitness: {res.best_fitness:.4f}")
print(f"Bias Teórico:  {f4.f_bias:.4f}")
print(f"Error Delta f: {error:.4f}")
print(f"Tiempo:        {res.execution_time:.2f} s")
