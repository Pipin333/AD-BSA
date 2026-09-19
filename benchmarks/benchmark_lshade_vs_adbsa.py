"""
================================================================================
  BENCHMARK: L-SHADE vs AD-BSA v2 (Runtime & Solution Quality Comparison)
  
  Verifiable Experimental Conditions:
    - Protocol: 30 Dimensions, 75,000 evaluations (MaxNFE), 10 independent
      runs with deterministic seeds (seed = 1000 + r * 17).
    - Environment: x86_64 CPU, Windows, Python 3.13, NumPy 2.x.
    - Consolidated Empirical Results:
        * L-SHADE average runtime per run: 6.38 s
        * AD-BSA v2 average runtime per run: 0.80 s
        * Empirical execution speedup: ~8.0x (NumPy matrix vectorization)
        * Solution quality (fitness): Canonical L-SHADE achieves higher
          exploitative precision across the 6 standard functions under this protocol.
================================================================================
"""

import math
import os
import sys
import time
from typing import Callable, Dict, List, Tuple
import numpy as np
from scipy import stats

# Permitir importación tanto desde la raíz del repo como desde benchmarks/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ad_bsa_v2 import AD_BSA_v2


# ==============================================================================
#  1. IMPLEMENTACIÓN CANÓNICA DE L-SHADE (Tanabe & Fukunaga, 2014)
# ==============================================================================

class L_SHADE:
    """
    Linear Population Size Reduction Success-History Based Adaptive Differential Evolution.
    Referencia: Tanabe, R., & Fukunaga, A. S. (2014).
    Improving the search performance of SHADE using linear population size reduction.
    Proc. IEEE CEC 2014, pp. 1658-1665.
    """

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        max_evaluations: int = 75000,
        memory_size: int = 6,
        p_best_rate: float = 0.11,
        arc_rate: float = 1.4,
        seed: int = None
    ):
        self.f = objective_func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.max_nfe = max_evaluations
        self.H = memory_size
        self.p_rate = p_best_rate
        self.arc_rate = arc_rate
        self.rng = np.random.default_rng(seed)

        # Configuración canónica de población de L-SHADE: N_init = 18 * D
        self.N_init = min(18 * self.dim, max(50, int(self.max_nfe / 200)))
        self.N_min = 4
        self.N = self.N_init

    def optimize(self) -> Tuple[np.ndarray, float, float, int]:
        t0 = time.perf_counter()
        low, high = self.bounds[:, 0], self.bounds[:, 1]

        # Inicialización de población
        pop = self.rng.uniform(low, high, size=(self.N, self.dim))
        fits = self.f(pop)
        nfe = self.N

        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]

        # Memoria histórica de parámetros
        M_F = np.full(self.H, 0.5, dtype=np.float64)
        M_CR = np.full(self.H, 0.5, dtype=np.float64)
        k_mem = 0

        # Archivo externo de soluciones reemplazadas
        archive: List[np.ndarray] = []
        max_archive_size = int(self.arc_rate * self.N_init)

        generations = 0

        while nfe < self.max_nfe:
            generations += 1
            S_F: List[float] = []
            S_CR: List[float] = []
            delta_f: List[float] = []

            # Ordenar población por fitness
            sort_idx = np.argsort(fits)
            pop = pop[sort_idx]
            fits = fits[sort_idx]

            # Tamaño del conjunto p-best
            p_num = max(2, int(round(self.p_rate * self.N)))

            trial_pop = np.empty_like(pop)
            trial_F = np.empty(self.N, dtype=np.float64)
            trial_CR = np.empty(self.N, dtype=np.float64)

            # Generación de vectores de prueba
            for i in range(self.N):
                r_k = self.rng.integers(0, self.H)

                # Muestreo de CR
                cr = self.rng.normal(M_CR[r_k], 0.1)
                cr = np.clip(cr, 0.0, 1.0)
                trial_CR[i] = cr

                # Muestreo de F (Cauchy)
                f_val = stats.cauchy.rvs(loc=M_F[r_k], scale=0.1, random_state=self.rng)
                while f_val <= 0.0:
                    f_val = stats.cauchy.rvs(loc=M_F[r_k], scale=0.1, random_state=self.rng)
                f_val = min(1.0, f_val)
                trial_F[i] = f_val

                # Selección de p-best
                p_idx = self.rng.integers(0, p_num)
                x_pbest = pop[p_idx]

                # Selección de r1 y r2
                r1 = self.rng.integers(0, self.N)
                while r1 == i:
                    r1 = self.rng.integers(0, self.N)
                x_r1 = pop[r1]

                # r2 de P U A
                pool_size = self.N + len(archive)
                r2 = self.rng.integers(0, pool_size)
                while r2 == i or r2 == r1:
                    r2 = self.rng.integers(0, pool_size)

                if r2 < self.N:
                    x_r2 = pop[r2]
                else:
                    x_r2 = archive[r2 - self.N]

                # Mutación canónica DE/current-to-pbest/1
                v = pop[i] + f_val * (x_pbest - pop[i]) + f_val * (x_r1 - x_r2)

                # Cruce binomial
                j_rand = self.rng.integers(0, self.dim)
                mask = (self.rng.uniform(0.0, 1.0, size=self.dim) < cr)
                mask[j_rand] = True
                u = np.where(mask, v, pop[i])

                # Reparación de bordes (midpoint reflection)
                under = u < low
                over = u > high
                u[under] = (pop[i][under] + low[under]) / 2.0
                u[over] = (pop[i][over] + high[over]) / 2.0

                trial_pop[i] = u

            # Evaluación de candidatos
            eval_budget = min(self.N, self.max_nfe - nfe)
            if eval_budget <= 0:
                break

            trial_fits = self.f(trial_pop[:eval_budget])
            nfe += eval_budget

            # Selección codiciosa y archivo
            new_pop = pop.copy()
            new_fits = fits.copy()

            for i in range(eval_budget):
                if trial_fits[i] <= fits[i]:
                    archive.append(pop[i].copy())
                    if len(archive) > max_archive_size:
                        del_idx = self.rng.integers(0, len(archive))
                        archive.pop(del_idx)

                    new_pop[i] = trial_pop[i]
                    new_fits[i] = trial_fits[i]

                    if trial_fits[i] < fits[i]:
                        S_F.append(trial_F[i])
                        S_CR.append(trial_CR[i])
                        delta_f.append(abs(fits[i] - trial_fits[i]))

                    if trial_fits[i] < best_f:
                        best_f = trial_fits[i]
                        best_x = trial_pop[i].copy()

            pop = new_pop
            fits = new_fits

            # Actualización de memoria histórica (Lehmer mean)
            if len(S_F) > 0 and sum(delta_f) > 0:
                weights = np.array(delta_f, dtype=np.float64) / sum(delta_f)
                s_f_arr = np.array(S_F, dtype=np.float64)
                s_cr_arr = np.array(S_CR, dtype=np.float64)

                # Lehmer mean para F
                m_f_new = np.sum(weights * (s_f_arr ** 2)) / np.sum(weights * s_f_arr)
                # Weighted arithmetic mean para CR
                m_cr_new = np.sum(weights * s_cr_arr)

                M_F[k_mem] = np.clip(m_f_new, 0.1, 1.0)
                M_CR[k_mem] = np.clip(m_cr_new, 0.0, 1.0)
                k_mem = (k_mem + 1) % self.H

            # Linear Population Size Reduction (LPSR)
            next_N = int(round(self.N_init - (nfe / self.max_nfe) * (self.N_init - self.N_min)))
            next_N = max(self.N_min, next_N)

            if next_N < self.N:
                # Truncar los peores individuos
                sort_idx = np.argsort(fits)
                pop = pop[sort_idx[:next_N]]
                fits = fits[sort_idx[:next_N]]
                self.N = next_N
                max_archive_size = int(round(self.arc_rate * self.N))
                while len(archive) > max_archive_size:
                    archive.pop(self.rng.integers(0, len(archive)))

        runtime = time.perf_counter() - t0
        return best_x, best_f, runtime, nfe


# ==============================================================================
#  2. SUITE DE FUNCIONES BENCHMARK CONTINUAS 30D
# ==============================================================================

def f1_sphere(x):
    return np.sum(np.atleast_2d(x) ** 2, axis=1)

def f2_rosenbrock(x):
    x_2d = np.atleast_2d(x)
    return np.sum(100.0 * (x_2d[:, 1:] - x_2d[:, :-1]**2)**2 + (x_2d[:, :-1] - 1.0)**2, axis=1)

def f3_schwefel222(x):
    x_2d = np.abs(np.atleast_2d(x))
    return np.sum(x_2d, axis=1) + np.prod(x_2d, axis=1)

def f4_rastrigin(x):
    x_2d = np.atleast_2d(x)
    return np.sum(x_2d**2 - 10.0 * np.cos(2.0 * np.pi * x_2d) + 10.0, axis=1)

def f5_ackley(x):
    x_2d = np.atleast_2d(x)
    d = x_2d.shape[1]
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x_2d**2, axis=1) / d))
    term2 = -np.exp(np.sum(np.cos(2.0 * np.pi * x_2d), axis=1) / d)
    return term1 + term2 + 20.0 + np.e

def f6_griewank(x):
    x_2d = np.atleast_2d(x)
    d = x_2d.shape[1]
    sum_sq = np.sum(x_2d**2, axis=1) / 4000.0
    i_vec = np.sqrt(np.arange(1, d + 1))
    prod_cos = np.prod(np.cos(x_2d / i_vec), axis=1)
    return sum_sq - prod_cos + 1.0


# ==============================================================================
#  3. EJECUCIÓN EXPERIMENTAL RIGUROSA
# ==============================================================================

def run_head_to_head_comparison():
    D = 30
    MaxNFE = 75000
    N_RUNS = 10  # 10 corridas independientes por función

    benchmarks = [
        ("Sphere (f1)", f1_sphere, np.array([[-100.0, 100.0]] * D)),
        ("Rosenbrock (f2)", f2_rosenbrock, np.array([[-30.0, 30.0]] * D)),
        ("Schwefel 2.22 (f3)", f3_schwefel222, np.array([[-10.0, 10.0]] * D)),
        ("Rastrigin (f4)", f4_rastrigin, np.array([[-5.12, 5.12]] * D)),
        ("Ackley (f5)", f5_ackley, np.array([[-32.0, 32.0]] * D)),
        ("Griewank (f6)", f6_griewank, np.array([[-600.0, 600.0]] * D)),
    ]

    print("=" * 105)
    print("  COMPARACIÓN EXPERIMENTAL HEAD-TO-HEAD: L-SHADE (2014) vs AD-BSA v2 (2026)")
    print(f"  Dimension: {D}D | MaxNFE: {MaxNFE:,} | Corridas Independientes: {N_RUNS}")
    print("=" * 105)

    results = {}

    for name, func, bounds in benchmarks:
        print(f"\n>>> Evaluando {name}...")
        lshade_fits = []
        lshade_times = []
        adbsa_fits = []
        adbsa_times = []

        for r in range(N_RUNS):
            seed = 1000 + r * 17

            # 1. L-SHADE
            lshade = L_SHADE(objective_func=func, bounds=bounds, max_evaluations=MaxNFE, seed=seed)
            _, l_fit, l_time, _ = lshade.optimize()
            lshade_fits.append(l_fit)
            lshade_times.append(l_time)

            # 2. AD-BSA v2
            adbsa = AD_BSA_v2(objective_func=func, bounds=bounds, max_evaluations=MaxNFE, annealing_power=1.8, seed=seed)
            res_a = adbsa.optimize()
            adbsa_fits.append(res_a.safe_house_fitness)
            adbsa_times.append(res_a.execution_time_seconds)

        # Wilcoxon signed-rank test
        try:
            stat, p_val = stats.wilcoxon(np.array(adbsa_fits) - np.array(lshade_fits))
        except Exception:
            p_val = 1.0

        results[name] = {
            "lshade_mean": np.mean(lshade_fits),
            "lshade_std": np.std(lshade_fits),
            "lshade_med": np.median(lshade_fits),
            "lshade_time": np.mean(lshade_times),
            "adbsa_mean": np.mean(adbsa_fits),
            "adbsa_std": np.std(adbsa_fits),
            "adbsa_med": np.median(adbsa_fits),
            "adbsa_time": np.mean(adbsa_times),
            "p_val": p_val
        }

        print(f"    * L-SHADE : Mean = {np.mean(lshade_fits):.2e} | Time = {np.mean(lshade_times):.2f}s")
        print(f"    * AD-BSA 2: Mean = {np.mean(adbsa_fits):.2e} | Time = {np.mean(adbsa_times):.2f}s | p-val = {p_val:.4f}")

    # Guardar resultados
    import json
    out_path = os.path.join(os.path.dirname(__file__), "lshade_vs_adbsa_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados exitosamente en: {out_path}")

    return results


if __name__ == "__main__":
    run_head_to_head_comparison()
