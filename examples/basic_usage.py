"""
================================================================================
  AD-BSA Minimal Quickstart Example
================================================================================
"""

import os
import sys
import numpy as np

# Permite ejecutar el script directamente sin requerir pip install previo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from ad_bsa import AD_BSA


# 1. Definir la función objetivo continua (vectorizada N x D o 1D)
def sphere(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.sum(x**2, axis=1)

# 2. Definir límites del espacio de búsqueda [-100, 100]^30
dim = 30
bounds = np.array([[-100.0, 100.0]] * dim)

# 3. Instanciar y optimizar con AD-BSA
optimizer = AD_BSA(
    objective_func=sphere,
    bounds=bounds,
    max_evaluations=30000,
    seed=42
)

result = optimizer.optimize()

# 4. Imprimir resultados
print(f"=== RESULTADOS DE OPTIMIZACIÓN AD-BSA ===")
print(f"Mejor Fitness Encontrado: {result.best_fitness:.6e}")
print(f"Evaluaciones Totales:    {result.total_evaluations}")
print(f"Generaciones:            {result.generations}")
print(f"Tiempo de Cómputo:       {result.execution_time:.3f} s")
print(f"Norma L2 del Óptimo:     {np.linalg.norm(result.best_position):.6e}")
