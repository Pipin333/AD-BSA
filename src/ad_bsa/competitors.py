"""
================================================================================
  AD-BSA: Adaptive Differential Boogeyman Search Algorithm
  Module: src/ad_bsa/competitors.py
  State-of-the-Art Elite Competitors for IEEE CEC Benchmarks:
    1. jSO (Brest et al., IEEE CEC 2017 Champion)
    2. CMA-ES (Hansen et al., Covariance Matrix Adaptation)
    3. L-SHADE (Tanabe & Fukunaga, IEEE CEC 2014 Champion)
    4. Standard DE (DE/rand/1/bin, Storn & Price, 1997)
    5. Standard PSO (Eberhart & Kennedy, 1995 / Shi & Eberhart, 1998)
    6. Canonical Cuckoo Search (Yang & Deb, 2009)
    7. NL-SHADE-RSP (Stanovov et al., IEEE CEC 2021 Champion)
    8. NL-SHADE-LBC (Stanovov et al., IEEE CEC 2022 Top Performer)
    9. MadDE (Biswas et al., IEEE CEC 2021 Multi-Adaptation DE)
================================================================================
"""

import math
import time
from typing import Callable, List, Optional
import numpy as np

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

from .algorithm import OptimizationResult
from .utils import (
    bound_constraint_shade,
    bound_constraint_clamp,
)


# ==============================================================================
#  1. jSO (Brest et al., IEEE CEC 2017 Winner)
# ==============================================================================

class jSO:
    """
    jSO: Modified L-SHADE for Single Objective Real-Parameter Numerical Optimization.
    Referencia Canónica: Brest, J., Maucec, M. S., & Boskovic, B. (2017).
    Single objective real-parameter optimization: Algorithm jSO.
    Proc. IEEE CEC 2017, pp. 1311-1318.
    """

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        max_evaluations: int = 50000,
        memory_size: int = 5,
        pop_size_init: Optional[int] = None,
        seed: Optional[int] = None
    ):
        self.f = objective_func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.max_nfe = int(max_evaluations)
        self.H = int(memory_size)
        self.rng = np.random.default_rng(seed)

        self.N_min = 4
        if pop_size_init is not None:
            self.N_init = int(pop_size_init)
        else:
            # Fórmula canónica de jSO (Brest et al., IEEE CEC 2017):
            # N_init = round(25 * sqrt(D) * ln(D)) si D > 1
            if self.dim > 1:
                canonical_N = int(round(25.0 * math.sqrt(self.dim) * math.log(self.dim)))
            else:
                canonical_N = 18 * self.dim
            # Salvaguarda para pruebas con presupuesto reducido
            self.N_init = min(canonical_N, max(self.N_min + 2, int(self.max_nfe / 3)))

        self.N = self.N_init
        self.arc_rate = 2.6  # Ratio canónico de archivo en jSO (|A| = 2.6 * N)

    def optimize(self) -> OptimizationResult:
        start_time = time.perf_counter()
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        bound_range = high - low

        pop = low + self.rng.random((self.N, self.dim)) * bound_range
        fits = self.f(pop)
        nfe = self.N

        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = float(fits[best_idx])

        hist_best = [best_f]
        hist_evals = [nfe]

        M_F = np.full(self.H, 0.5, dtype=np.float64)
        M_CR = np.full(self.H, 0.8, dtype=np.float64)
        k_mem = 0

        archive: List[np.ndarray] = []
        max_arc = int(round(self.arc_rate * self.N_init))
        generation = 0

        while nfe < self.max_nfe:
            generation += 1
            nfe_ratio = nfe / self.max_nfe
            p_val = 0.25 - 0.125 * nfe_ratio
            p_num = max(2, int(round(p_val * self.N)))

            sort_idx = np.argsort(fits)
            pop = pop[sort_idx]
            fits = fits[sort_idx]

            S_F: List[float] = []
            S_CR: List[float] = []
            delta_f: List[float] = []
            trial_pop = np.empty_like(pop)
            trial_F = np.empty(self.N, dtype=np.float64)
            trial_CR = np.empty(self.N, dtype=np.float64)

            for i in range(self.N):
                r_k = self.rng.integers(0, self.H)

                if M_CR[r_k] < 0:
                    cr = 0.0
                else:
                    cr = np.clip(self.rng.normal(M_CR[r_k], 0.1), 0.0, 1.0)
                if nfe_ratio < 0.25 and cr < 0.7:
                    cr = 0.7
                elif nfe_ratio < 0.5 and cr < 0.6:
                    cr = 0.6

                while True:
                    f_val = float(M_F[r_k] + 0.1 * np.tan(np.pi * (self.rng.random() - 0.5)))
                    if f_val > 0.0:
                        break
                if f_val > 1.0:
                    f_val = 1.0
                if nfe_ratio < 0.6 and f_val > 0.7:
                    f_val = 0.7

                trial_F[i] = f_val
                trial_CR[i] = cr

                if nfe_ratio < 0.2:
                    Fw = 0.7 * f_val
                elif nfe_ratio < 0.4:
                    Fw = 0.8 * f_val
                else:
                    Fw = 1.2 * f_val

                p_idx = int(self.rng.integers(0, p_num))
                x_pbest = pop[p_idx]

                while True:
                    r1 = int(self.rng.integers(0, self.N))
                    if r1 != i:
                        break
                x_r1 = pop[r1]

                pool_size = self.N + len(archive)
                while True:
                    r2_idx = int(self.rng.integers(0, pool_size))
                    if r2_idx != i and r2_idx != r1:
                        break
                x_r2 = pop[r2_idx] if r2_idx < self.N else archive[r2_idx - self.N]

                v = pop[i] + Fw * (x_pbest - pop[i]) + f_val * (x_r1 - x_r2)
                # Manejo de fronteras canónico del punto medio (SHADE/jSO)
                v = bound_constraint_shade(v[np.newaxis, :], lb=low, ub=high, base=pop[i:i+1])[0]

                j_rand = int(self.rng.integers(0, self.dim))
                cross_mask = (self.rng.random(self.dim) < cr)
                cross_mask[j_rand] = True
                trial_pop[i] = np.where(cross_mask, v, pop[i])

            batch_size = min(self.N, self.max_nfe - nfe)
            trial_fits = self.f(trial_pop[:batch_size])
            nfe += batch_size

            for i in range(batch_size):
                if trial_fits[i] < fits[i]:
                    archive.append(pop[i].copy())
                    delta_f.append(fits[i] - trial_fits[i])
                    S_F.append(trial_F[i])
                    S_CR.append(trial_CR[i])
                    pop[i] = trial_pop[i].copy()
                    fits[i] = trial_fits[i]
                    if trial_fits[i] < best_f:
                        best_f = float(trial_fits[i])
                        best_x = trial_pop[i].copy()

            while len(archive) > max_arc:
                del archive[self.rng.integers(0, len(archive))]

            if len(S_F) > 0:
                w = np.array(delta_f, dtype=np.float64) / (np.sum(delta_f) + 1e-14)
                # Media de Lehmer ponderada para F
                M_F[k_mem] = np.sum(w * (np.array(S_F)**2)) / (np.sum(w * np.array(S_F)) + 1e-14)
                # Media aritmética ponderada canónica para CR
                if np.max(S_CR) > 0:
                    M_CR[k_mem] = float(np.sum(w * np.array(S_CR)))
                else:
                    M_CR[k_mem] = -1.0
                k_mem = (k_mem + 1) % self.H

            target_N = round(((self.N_min - self.N_init) / self.max_nfe) * nfe + self.N_init)
            target_N = max(self.N_min, target_N)
            if self.N > target_N:
                survivors = np.argsort(fits)[:target_N]
                pop = pop[survivors]
                fits = fits[survivors]
                self.N = target_N
                max_arc = int(round(self.arc_rate * self.N))
                while len(archive) > max_arc:
                    del archive[self.rng.integers(0, len(archive))]

            hist_best.append(best_f)
            hist_evals.append(nfe)

        return OptimizationResult(
            best_position=best_x,
            best_fitness=best_f,
            total_evaluations=nfe,
            generations=generation,
            history_best_fitness=hist_best,
            history_evaluations=hist_evals,
            execution_time=time.perf_counter() - start_time
        )

# ==============================================================================
#  2. CMA-ES (Hansen et al., Covariance Matrix Adaptation con IPOP Restarts)
# ==============================================================================

class CMA_ES:
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) con reinicios IPOP.
    Hansen, N. (2006). The CMA evolution strategy: a comparing review.
    Implementación oficial con reinicios de tamaño de población creciente (IPOP)
    utilizada en las evaluaciones oficiales de IEEE CEC.
    """

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        max_evaluations: int = 50000,
        enable_restarts: bool = True,
        seed: Optional[int] = None
    ):
        if not HAS_CMA:
            raise ImportError("El paquete 'cma' no está instalado. Instálalo con: pip install cma")
        self.f = objective_func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.max_nfe = int(max_evaluations)
        self.enable_restarts = bool(enable_restarts)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def optimize(self) -> OptimizationResult:
        start_time = time.perf_counter()
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        bound_range = high - low

        total_evals = 0
        total_iters = 0
        best_f = float("inf")
        best_x = np.empty(self.dim, dtype=np.float64)

        hist_best: List[float] = []
        hist_evals: List[int] = []

        # Parámetros canónicos de tamaño poblacional inicial y escalamiento de IPOP
        popsize = 4 + int(3 * math.log(self.dim))
        popsize_inc_factor = 2

        while total_evals < self.max_nfe:
            # Muestreo uniforme canónico en todo el espacio factible [low, high]
            x0 = low + self.rng.random(self.dim) * bound_range
            sigma0 = float(bound_range[0] * 0.3)

            remaining_evals = self.max_nfe - total_evals
            if remaining_evals <= 0:
                break

            opts = {
                "bounds": [low.tolist(), high.tolist()],
                "maxfevals": remaining_evals,
                "popsize": popsize,
                "seed": int(self.rng.integers(1, 1000000)),
                "verbose": -9,
            }

            try:
                es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            except Exception:
                break

            while not es.stop() and total_evals < self.max_nfe:
                solutions = es.ask()
                batch = np.asarray(solutions, dtype=np.float64)
                actual_batch = min(len(batch), self.max_nfe - total_evals)
                if actual_batch < len(batch):
                    batch = batch[:actual_batch]
                    solutions = solutions[:actual_batch]

                fits = self.f(batch)
                total_evals += actual_batch
                total_iters += 1

                try:
                    es.tell(solutions, fits.tolist())
                except Exception:
                    pass

                min_idx = np.argmin(fits)
                if fits[min_idx] < best_f:
                    best_f = float(fits[min_idx])
                    best_x = batch[min_idx].copy()

                hist_best.append(best_f)
                hist_evals.append(total_evals)

            if not self.enable_restarts:
                break

            popsize *= popsize_inc_factor

        return OptimizationResult(
            best_position=best_x,
            best_fitness=best_f,
            total_evaluations=total_evals,
            generations=total_iters,
            history_best_fitness=hist_best,
            history_evaluations=hist_evals,
            execution_time=time.perf_counter() - start_time
        )

# ==============================================================================
#  3. L-SHADE (Tanabe & Fukunaga, IEEE CEC 2014)
# ==============================================================================

class L_SHADE:
    """
    Linear Population Size Reduction SHADE (Tanabe & Fukunaga, IEEE CEC 2014).
    Implementación canónica estricta con reducción lineal de población,
    memoria histórica de Lehmer para F, media aritmética para CR y regla de frontera SHADE.
    """

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        max_evaluations: int = 50000,
        memory_size: int = 6,
        p_best_rate: float = 0.11,
        arc_rate: float = 1.4,
        pop_size_init: Optional[int] = None,
        seed: Optional[int] = None
    ):
        self.func = objective_func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.max_nfe = int(max_evaluations)
        self.H = int(memory_size)
        self.p_rate = float(p_best_rate)
        self.arc_rate = float(arc_rate)
        self.rng = np.random.default_rng(seed)

        self.N_min = 4
        if pop_size_init is not None:
            self.N_init = int(pop_size_init)
        else:
            # Fórmula canónica de L-SHADE (Tanabe & Fukunaga, 2014): N_init = 18 * D
            canonical_N = 18 * self.dim
            self.N_init = min(canonical_N, max(self.N_min + 2, int(self.max_nfe / 3)))

        self.N = self.N_init

    def optimize(self) -> OptimizationResult:
        start_time = time.perf_counter()
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        bound_range = high - low

        pop = low + self.rng.random((self.N, self.dim)) * bound_range
        fits = self.func(pop)
        nfe = self.N

        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = float(fits[best_idx])

        hist_best = [best_f]
        hist_evals = [nfe]

        M_F = np.full(self.H, 0.5, dtype=np.float64)
        M_CR = np.full(self.H, 0.5, dtype=np.float64)
        k_mem = 0

        archive: List[np.ndarray] = []
        max_archive_size = int(round(self.arc_rate * self.N_init))
        generation = 0

        while nfe < self.max_nfe:
            generation += 1
            S_F: List[float] = []
            S_CR: List[float] = []
            delta_f: List[float] = []

            sort_idx = np.argsort(fits)
            pop = pop[sort_idx]
            fits = fits[sort_idx]

            p_num = max(2, int(round(self.p_rate * self.N)))
            trial_pop = np.empty_like(pop)
            trial_F = np.empty(self.N, dtype=np.float64)
            trial_CR = np.empty(self.N, dtype=np.float64)

            for i in range(self.N):
                r_k = self.rng.integers(0, self.H)

                cr = np.clip(self.rng.normal(M_CR[r_k], 0.1), 0.0, 1.0)
                trial_CR[i] = cr

                while True:
                    f = M_F[r_k] + 0.1 * np.tan(np.pi * (self.rng.random() - 0.5))
                    if f > 0.0:
                        break
                trial_F[i] = min(f, 1.0)

                p_idx = int(self.rng.integers(0, p_num))
                x_pbest = pop[p_idx]

                while True:
                    r1 = int(self.rng.integers(0, self.N))
                    if r1 != i:
                        break

                pool_size = self.N + len(archive)
                while True:
                    r2_idx = int(self.rng.integers(0, pool_size))
                    if r2_idx != i and r2_idx != r1:
                        break
                x_r2 = pop[r2_idx] if r2_idx < self.N else archive[r2_idx - self.N]

                v = pop[i] + trial_F[i] * (x_pbest - pop[i]) + trial_F[i] * (pop[r1] - x_r2)
                # Manejo de fronteras canónico del punto medio (SHADE)
                v = bound_constraint_shade(v[np.newaxis, :], lb=low, ub=high, base=pop[i:i+1])[0]

                j_rand = self.rng.integers(0, self.dim)
                cross_mask = (self.rng.random(self.dim) < trial_CR[i])
                cross_mask[j_rand] = True
                u = np.where(cross_mask, v, pop[i])
                trial_pop[i] = u

            evals_batch = min(self.N, self.max_nfe - nfe)
            trial_fits = self.func(trial_pop[:evals_batch])
            nfe += evals_batch

            for i in range(evals_batch):
                if trial_fits[i] <= fits[i]:
                    if trial_fits[i] < fits[i]:
                        S_F.append(trial_F[i])
                        S_CR.append(trial_CR[i])
                        delta_f.append(fits[i] - trial_fits[i])
                        archive.append(pop[i].copy())
                    pop[i] = trial_pop[i].copy()
                    fits[i] = trial_fits[i]
                    if trial_fits[i] < best_f:
                        best_f = float(trial_fits[i])
                        best_x = trial_pop[i].copy()

            if len(archive) > max_archive_size:
                survivors = self.rng.choice(len(archive), size=max_archive_size, replace=False)
                archive = [archive[idx] for idx in survivors]

            if len(S_F) > 0:
                w = np.array(delta_f, dtype=np.float64) / (np.sum(delta_f) + 1e-14)
                # Media de Lehmer ponderada para F
                M_F[k_mem] = np.sum(w * (np.array(S_F)**2)) / (np.sum(w * np.array(S_F)) + 1e-14)
                # Media aritmética ponderada canónica para CR
                if np.max(S_CR) > 0:
                    M_CR[k_mem] = float(np.sum(w * np.array(S_CR)))
                else:
                    M_CR[k_mem] = -1.0
                k_mem = (k_mem + 1) % self.H

            if nfe < self.max_nfe:
                N_next = int(round(
                    ((self.N_min - self.N_init) / self.max_nfe) * nfe + self.N_init
                ))
                N_next = max(self.N_min, N_next)
                if N_next < self.N:
                    sort_idx = np.argsort(fits)
                    pop = pop[sort_idx[:N_next]]
                    fits = fits[sort_idx[:N_next]]
                    self.N = N_next
                    max_archive_size = int(round(self.arc_rate * self.N))
                    if len(archive) > max_archive_size:
                        survivors = self.rng.choice(len(archive), size=max_archive_size, replace=False)
                        archive = [archive[idx] for idx in survivors]

            hist_best.append(best_f)
            hist_evals.append(nfe)

        return OptimizationResult(
            best_position=best_x,
            best_fitness=best_f,
            total_evaluations=nfe,
            generations=generation,
            history_best_fitness=hist_best,
            history_evaluations=hist_evals,
            execution_time=time.perf_counter() - start_time
        )

# ==============================================================================
#  4. Standard Differential Evolution (DE/rand/1/bin)
# ==============================================================================

class StandardDE:
    """Evolución Diferencial canónica clásica (DE/rand/1/bin) vectorizada con clamping de fronteras."""

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        pop_size: int = 50,
        max_evaluations: int = 50000,
        F: float = 0.7,
        CR: float = 0.9,
        seed: Optional[int] = None
    ):
        self.func = objective_func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.pop_size = int(pop_size)
        self.max_evals = int(max_evaluations)
        self.F = float(F)
        self.CR = float(CR)
        self.rng = np.random.default_rng(seed)

    def optimize(self) -> OptimizationResult:
        start_time = time.perf_counter()
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        pop = lb + self.rng.random((self.pop_size, self.dim)) * (ub - lb)
        fitness = self.func(pop)
        evals = self.pop_size

        best_idx = np.argmin(fitness)
        best_pos = pop[best_idx].copy()
        best_fit = float(fitness[best_idx])

        hist_evals = [evals]
        hist_fit = [best_fit]
        generation = 0

        while evals < self.max_evals:
            generation += 1
            N = self.pop_size
            idx_matrix = np.tile(np.arange(N), (N, 1))
            mask = ~np.eye(N, dtype=bool)
            candidates = idx_matrix[mask].reshape(N, N - 1)
            r_cols = self.rng.random((N, N - 1)).argsort(axis=1)[:, :3]

            r1 = candidates[np.arange(N), r_cols[:, 0]]
            r2 = candidates[np.arange(N), r_cols[:, 1]]
            r3 = candidates[np.arange(N), r_cols[:, 2]]

            mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
            # Clamping canónico a fronteras
            mutant = bound_constraint_clamp(mutant, lb=lb, ub=ub)

            rand_j = self.rng.integers(0, self.dim, size=N)
            j_matrix = np.tile(np.arange(self.dim), (N, 1))
            j_mask = (j_matrix == rand_j[:, np.newaxis])
            cross_mask = (self.rng.random((N, self.dim)) <= self.CR) | j_mask
            trial = np.where(cross_mask, mutant, pop)

            eval_batch = min(N, self.max_evals - evals)
            trial_fit = self.func(trial[:eval_batch])
            evals += eval_batch

            better_mask = trial_fit <= fitness[:eval_batch]
            pop[:eval_batch][better_mask] = trial[:eval_batch][better_mask]
            fitness[:eval_batch][better_mask] = trial_fit[better_mask]

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fit:
                best_fit = float(fitness[best_idx])
                best_pos = pop[best_idx].copy()

            hist_evals.append(evals)
            hist_fit.append(best_fit)

        return OptimizationResult(
            best_position=best_pos,
            best_fitness=best_fit,
            total_evaluations=evals,
            generations=generation,
            history_best_fitness=hist_fit,
            history_evaluations=hist_evals,
            execution_time=time.perf_counter() - start_time
        )


# ==============================================================================
#  5. Standard PSO (Global Best with Inertia Weight Decay)
# ==============================================================================

class StandardPSO:
    """
    Particle Swarm Optimization canónico vectorizado con decaimiento lineal de inercia
    (Shi & Eberhart, 1998) y absorción de velocidad en las fronteras.
    """

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        pop_size: int = 50,
        max_evaluations: int = 50000,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        seed: Optional[int] = None
    ):
        self.func = objective_func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.pop_size = int(pop_size)
        self.max_evals = int(max_evaluations)
        self.w_max = float(w_max)
        self.w_min = float(w_min)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.rng = np.random.default_rng(seed)

    def optimize(self) -> OptimizationResult:
        start_time = time.perf_counter()
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        v_max = 0.2 * (ub - lb)

        pos = lb + self.rng.random((self.pop_size, self.dim)) * (ub - lb)
        vel = self.rng.uniform(-v_max, v_max, size=(self.pop_size, self.dim))

        pbest_pos = pos.copy()
        pbest_fit = self.func(pos)
        evals = self.pop_size

        gbest_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = float(pbest_fit[gbest_idx])

        hist_evals = [evals]
        hist_fit = [gbest_fit]
        generation = 0

        while evals < self.max_evals:
            generation += 1
            w = self.w_max - (evals / self.max_evals) * (self.w_max - self.w_min)
            r1 = self.rng.random((self.pop_size, self.dim))
            r2 = self.rng.random((self.pop_size, self.dim))

            vel = w * vel + self.c1 * r1 * (pbest_pos - pos) + self.c2 * r2 * (gbest_pos - pos)
            vel = np.clip(vel, -v_max, v_max)

            # Actualización de posición y absorción de velocidad canónica en paredes
            new_pos = pos + vel
            low_viol = new_pos < lb
            high_viol = new_pos > ub
            pos = np.clip(new_pos, lb, ub)
            vel[low_viol | high_viol] = 0.0

            eval_batch = min(self.pop_size, self.max_evals - evals)
            fit = self.func(pos[:eval_batch])
            evals += eval_batch

            improved = fit < pbest_fit[:eval_batch]
            pbest_pos[:eval_batch][improved] = pos[:eval_batch][improved]
            pbest_fit[:eval_batch][improved] = fit[improved]

            current_min_idx = np.argmin(pbest_fit)
            if pbest_fit[current_min_idx] < gbest_fit:
                gbest_fit = float(pbest_fit[current_min_idx])
                gbest_pos = pbest_pos[current_min_idx].copy()

            hist_evals.append(evals)
            hist_fit.append(gbest_fit)

        return OptimizationResult(
            best_position=gbest_pos,
            best_fitness=gbest_fit,
            total_evaluations=evals,
            generations=generation,
            history_best_fitness=hist_fit,
            history_evaluations=hist_evals,
            execution_time=time.perf_counter() - start_time
        )


# ==============================================================================
#  6. Canonical Cuckoo Search (CS - Yang & Deb, 2009)
# ==============================================================================

class CanonicalCuckooSearch:
    """Cuckoo Search auténtico con Vuelos de Lévy de Mantegna y abandono de nidos (pa)."""

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        pop_size: int = 30,
        max_evaluations: int = 50000,
        pa: float = 0.25,
        alpha: float = 0.01,
        seed: Optional[int] = None
    ):
        self.func = objective_func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.pop_size = int(pop_size)
        self.max_evals = int(max_evaluations)
        self.pa = float(pa)
        self.alpha = float(alpha)
        self.rng = np.random.default_rng(seed)

    def _levy_flight(self, beta: float = 1.5) -> np.ndarray:
        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                   (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        u = self.rng.normal(0, sigma_u, size=(self.pop_size, self.dim))
        v = self.rng.normal(0, 1, size=(self.pop_size, self.dim))
        return u / (np.abs(v) ** (1 / beta))

    def optimize(self) -> OptimizationResult:
        start_time = time.perf_counter()
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        scale = ub - lb

        nests = lb + self.rng.random((self.pop_size, self.dim)) * scale
        fitness = self.func(nests)
        evals = self.pop_size

        best_idx = np.argmin(fitness)
        best_pos = nests[best_idx].copy()
        best_fit = float(fitness[best_idx])

        hist_evals = [evals]
        hist_fit = [best_fit]
        generation = 0

        while evals < self.max_evals:
            generation += 1
            step = self._levy_flight()
            new_nests = nests + self.alpha * step * (nests - best_pos)
            new_nests = bound_constraint_clamp(new_nests, lb=lb, ub=ub)

            eval_batch = min(self.pop_size, self.max_evals - evals)
            new_fit = self.func(new_nests[:eval_batch])
            evals += eval_batch

            better_mask = new_fit < fitness[:eval_batch]
            nests[:eval_batch][better_mask] = new_nests[:eval_batch][better_mask]
            fitness[:eval_batch][better_mask] = new_fit[better_mask]

            if evals < self.max_evals:
                discover_mask = self.rng.random(self.pop_size) < self.pa
                perm1 = self.rng.permutation(self.pop_size)
                perm2 = self.rng.permutation(self.pop_size)
                step_abandon = self.rng.random((self.pop_size, self.dim)) * (nests[perm1] - nests[perm2])
                abandoned_nests = bound_constraint_clamp(
                    nests + step_abandon * discover_mask[:, np.newaxis],
                    lb=lb, ub=ub
                )

                abandon_batch = min(self.pop_size, self.max_evals - evals)
                abandon_fit = self.func(abandoned_nests[:abandon_batch])
                evals += abandon_batch

                better_ab_mask = abandon_fit < fitness[:abandon_batch]
                nests[:abandon_batch][better_ab_mask] = abandoned_nests[:abandon_batch][better_ab_mask]
                fitness[:abandon_batch][better_ab_mask] = abandon_fit[better_ab_mask]

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fit:
                best_fit = float(fitness[best_idx])
                best_pos = nests[best_idx].copy()

            hist_evals.append(evals)
            hist_fit.append(best_fit)

        return OptimizationResult(
            best_position=best_pos,
            best_fitness=best_fit,
            total_evaluations=evals,
            generations=generation,
            history_best_fitness=hist_fit,
            history_evaluations=hist_evals,
            execution_time=time.perf_counter() - start_time
        )


# ==============================================================================
#  7. NL-SHADE-RSP (Stanovov, Akhmedova, & Semenkin, IEEE CEC 2021 Winner)
# ==============================================================================

class NL_SHADE_RSP:
    """
    NL-SHADE-RSP: Non-Linear Population Size Reduction SHADE with Rank-Based
    Selective Pressure and Adaptive Archive Probability.
    Referencia Canónica: Stanovov, V., Akhmedova, S., & Semenkin, E. (2021).
    NL-SHADE-RSP algorithm with adaptive archive and selective pressure for
    CEC 2021 numerical optimization. Proc. IEEE CEC 2021, pp. 809-816.
    DOI: 10.1109/CEC45853.2021.9504959
    """

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        max_evaluations: int = 50000,
        memory_size: Optional[int] = None,
        pop_size_init: Optional[int] = None,
        arc_rate: float = 2.1,
        seed: Optional[int] = None
    ) -> None:
        self.func = objective_func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.max_nfe = int(max_evaluations)
        self.H = int(20 * self.dim) if memory_size is None else int(memory_size)
        self.arc_rate = float(arc_rate)
        self.rng = np.random.default_rng(seed)

        self.N_min = 4
        if pop_size_init is not None:
            self.N_init = int(pop_size_init)
        else:
            # Fórmula canónica de NL-SHADE-RSP (Stanovov et al., 2021): N_init = 30 * D
            canonical_N = 30 * self.dim
            self.N_init = min(canonical_N, max(self.N_min + 2, int(self.max_nfe / 3)))

        self.N = self.N_init
        self.pbest_max = 0.4
        self.pbest_min = 0.2
        self.p_archive_init = 0.5
        self.p_archive_min = 0.1
        self.p_archive_max = 0.9
        self.rank_pressure_coeff = 1.0
        self.binomial_prob = 0.5
        self.memory_mixing = 0.5

    @staticmethod
    def _weighted_lehmer(values: np.ndarray, weights: np.ndarray) -> float:
        denom = float(np.sum(weights * values))
        if denom <= 1e-30 or not np.isfinite(denom):
            return float(np.clip(np.average(values, weights=weights), 0.0, 1.0))
        return float(np.clip(np.sum(weights * (values ** 2)) / denom, 0.0, 1.0))

    def _sample_cauchy_batch(self, centers: np.ndarray) -> np.ndarray:
        n = len(centers)
        res = np.zeros(n, dtype=np.float64)
        unfilled = np.ones(n, dtype=bool)
        for _ in range(100):
            if not np.any(unfilled):
                break
            cnt = int(np.sum(unfilled))
            vals = centers[unfilled] + 0.1 * np.tan(np.pi * (self.rng.random(cnt) - 0.5))
            valid = vals >= 0.0
            valid_idx = np.where(unfilled)[0][valid]
            res[valid_idx] = np.minimum(vals[valid], 1.0)
            unfilled[valid_idx] = False
        if np.any(unfilled):
            res[unfilled] = np.clip(centers[unfilled], 1e-12, 1.0)
        return res

    def optimize(self) -> OptimizationResult:
        start_time = time.perf_counter()
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        bound_range = high - low

        pop = low + self.rng.random((self.N, self.dim)) * bound_range
        fits = self.func(pop)
        nfe = self.N

        best_idx = int(np.argmin(fits))
        best_x = pop[best_idx].copy()
        best_f = float(fits[best_idx])

        hist_best = [best_f]
        hist_evals = [nfe]

        M_F = np.full(self.H, 0.2, dtype=np.float64)
        M_CR = np.full(self.H, 0.2, dtype=np.float64)
        k_mem = 0
        p_archive = self.p_archive_init

        archive = np.empty((0, self.dim), dtype=np.float64)
        generation = 0

        while nfe < self.max_nfe:
            generation += 1
            progress = float(np.clip(nfe / self.max_nfe, 0.0, 1.0))

            # Ordenar población por fitness (mejor a peor)
            order = np.argsort(fits)
            pop = pop[order]
            fits = fits[order]

            p_val = self.pbest_max + (self.pbest_min - self.pbest_max) * progress
            pnum = int(np.clip(round(p_val * self.N), 2, max(2, self.N)))
            cr_bin = 0.0 if progress < 0.5 else float(np.clip(2.0 * (progress - 0.5), 0.0, 1.0))

            mem_indices = self.rng.integers(0, self.H, size=self.N)
            Fs = self._sample_cauchy_batch(M_F[mem_indices])
            raw_cr = np.clip(self.rng.normal(M_CR[mem_indices], 0.1), 0.0, 1.0)
            # Como pop ya está ordenada de mejor a peor, asignar CRs ordenados directamente
            CRs = np.sort(raw_cr)

            # Pesos de presión selectiva basada en rango (RSP): R_i = exp(-i / NP)
            rank_positions = np.arange(1, self.N + 1, dtype=np.float64)
            rsp_weights = np.exp(-self.rank_pressure_coeff * rank_positions / float(self.N))
            rsp_weights /= np.sum(rsp_weights)
            rsp_cdf = np.cumsum(rsp_weights)
            rsp_cdf[-1] = 1.0

            trial_pop = np.empty_like(pop)
            used_archive = np.zeros(self.N, dtype=bool)
            arc_len = len(archive)

            for i in range(self.N):
                # pbest != i dentro de los primeros pnum
                pbest_idx = int(self.rng.integers(0, pnum))
                if pbest_idx == i:
                    pbest_idx = (pbest_idx + 1) % pnum

                # r1 uniforme != {i, pbest_idx}
                while True:
                    r1_idx = int(self.rng.integers(0, self.N))
                    if r1_idx != i and r1_idx != pbest_idx:
                        break

                # r2 desde archivo con probabilidad p_archive o desde población con RSP
                if arc_len > 0 and self.rng.random() < p_archive:
                    r2_vec = archive[int(self.rng.integers(0, arc_len))]
                    used_archive[i] = True
                else:
                    # Muestreo exacto con presión selectiva por rango excluyendo {i, pbest_idx, r1_idx}
                    r2_idx = i
                    for _ in range(32):
                        cand = int(np.searchsorted(rsp_cdf, self.rng.random(), side="right"))
                        if cand >= self.N:
                            cand = self.N - 1
                        if cand != i and cand != pbest_idx and cand != r1_idx:
                            r2_idx = cand
                            break
                    else:
                        for cand in range(self.N):
                            if cand != i and cand != pbest_idx and cand != r1_idx:
                                r2_idx = cand
                                break
                    r2_vec = pop[r2_idx]

                F = Fs[i]
                donor = pop[i] + F * (pop[pbest_idx] - pop[i]) + F * (pop[r1_idx] - r2_vec)

                # Cruce híbrido: 50% Binomial (con cr_bin) / 50% Exponencial (con CRs[i])
                if self.rng.random() < self.binomial_prob:
                    mask = self.rng.random(self.dim) < cr_bin
                    mask[int(self.rng.integers(0, self.dim))] = True
                else:
                    start = int(self.rng.integers(0, self.dim))
                    length = 1
                    cr_i = CRs[i]
                    while length < self.dim and self.rng.random() < cr_i:
                        length += 1
                    mask = np.zeros(self.dim, dtype=bool)
                    indices = (start + np.arange(length)) % self.dim
                    mask[indices] = True

                u = np.where(mask, donor, pop[i])

                # Reparación aleatoria de fronteras canónica de NL-SHADE-RSP
                bad = (u < low) | (u > high)
                if np.any(bad):
                    u[bad] = low[bad] + self.rng.random(int(np.sum(bad))) * bound_range[bad]

                trial_pop[i] = u

            evals_batch = min(self.N, self.max_nfe - nfe)
            trial_fits = self.func(trial_pop[:evals_batch])
            nfe += evals_batch

            parent_fits_batch = fits[:evals_batch]
            strict_mask = trial_fits < parent_fits_batch
            survive_mask = trial_fits <= parent_fits_batch

            positive_gains = np.zeros(evals_batch, dtype=np.float64)
            positive_gains[strict_mask] = parent_fits_batch[strict_mask] - trial_fits[strict_mask]

            if np.any(strict_mask):
                capacity = max(1, int(round(self.arc_rate * self.N)))
                defeated = pop[:evals_batch][strict_mask]
                for old_x in defeated:
                    if len(archive) < capacity:
                        archive = np.vstack([archive, old_x]) if len(archive) > 0 else old_x.reshape(1, -1).copy()
                    else:
                        archive[int(self.rng.integers(0, len(archive)))] = old_x

                sf = Fs[:evals_batch][strict_mask]
                scr = CRs[:evals_batch][strict_mask]
                df = positive_gains[strict_mask]
                weights = df / (float(np.sum(df)) + 1e-30)

                new_f = self._weighted_lehmer(sf, weights)
                new_cr = self._weighted_lehmer(scr, weights)
                M_F[k_mem] = float(np.clip(self.memory_mixing * M_F[k_mem] + (1.0 - self.memory_mixing) * new_f, 0.0, 1.0))
                M_CR[k_mem] = float(np.clip(self.memory_mixing * M_CR[k_mem] + (1.0 - self.memory_mixing) * new_cr, 0.0, 1.0))
                k_mem = (k_mem + 1) % self.H

            pop[:evals_batch][survive_mask] = trial_pop[:evals_batch][survive_mask]
            fits[:evals_batch][survive_mask] = trial_fits[survive_mask]

            curr_best_idx = int(np.argmin(fits))
            if fits[curr_best_idx] < best_f:
                best_f = float(fits[curr_best_idx])
                best_x = pop[curr_best_idx].copy()

            # Adaptación de la probabilidad de uso de archivo p_archive
            used_arc_batch = used_archive[:evals_batch]
            n_arc = int(np.sum(used_arc_batch))
            n_pop = int(evals_batch - n_arc)
            avg_arc = float(np.sum(positive_gains[used_arc_batch]) / max(1, n_arc))
            avg_pop = float(np.sum(positive_gains[~used_arc_batch]) / max(1, n_pop))
            if avg_arc + avg_pop > 1e-30:
                p_archive = float(np.clip(
                    avg_arc / (avg_arc + avg_pop),
                    self.p_archive_min,
                    self.p_archive_max
                ))

            # Reducción No Lineal de Población (NLPSR)
            if nfe < self.max_nfe:
                prog_after = float(np.clip(nfe / self.max_nfe, 0.0, 1.0))
                target_N = int(round((self.N_min - self.N_init) * (prog_after ** (1.0 - prog_after)) + self.N_init))
                target_N = int(np.clip(target_N, self.N_min, self.N))
                if target_N < self.N:
                    keep = np.argsort(fits)[:target_N]
                    pop = pop[keep]
                    fits = fits[keep]
                    self.N = target_N

                max_arc = max(1, int(round(self.arc_rate * self.N)))
                if len(archive) > max_arc:
                    keep_arc = self.rng.choice(len(archive), size=max_arc, replace=False)
                    archive = archive[keep_arc]

            hist_best.append(best_f)
            hist_evals.append(nfe)

        return OptimizationResult(
            best_position=best_x,
            best_fitness=best_f,
            total_evaluations=nfe,
            generations=generation,
            history_best_fitness=hist_best,
            history_evaluations=hist_evals,
            execution_time=time.perf_counter() - start_time
        )

# ==============================================================================
#  8. NL-SHADE-LBC (Stanovov, Akhmedova, & Semenkin, IEEE CEC 2022)
# ==============================================================================

class NL_SHADE_LBC:
    """
    NL-SHADE-LBC: Non-Linear Population Size Reduction SHADE with Linear
    Parameter Adaptation Bias Change (Generalized Weighted Lehmer Mean).
    Referencia Canónica: Stanovov, V., Akhmedova, S., & Semenkin, E. (2022).
    NL-SHADE-LBC algorithm with linear parameter adaptation bias change for
    CEC 2022 Numerical Optimization. Proc. IEEE CEC 2022, pp. 01-08.
    DOI: 10.1109/CEC55065.2022.9870295
    """

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        max_evaluations: int = 50000,
        memory_size: Optional[int] = None,
        pop_size_init: Optional[int] = None,
        arc_rate: float = 1.0,
        seed: Optional[int] = None
    ) -> None:
        self.func = objective_func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.max_nfe = int(max_evaluations)
        self.H = int(20 * self.dim) if memory_size is None else int(memory_size)
        self.arc_rate = float(arc_rate)
        self.rng = np.random.default_rng(seed)

        self.N_min = 4
        if pop_size_init is not None:
            self.N_init = int(pop_size_init)
        else:
            # Fórmula canónica de NL-SHADE-LBC (Stanovov et al., 2022): N_init = 23 * D
            canonical_N = 23 * self.dim
            self.N_init = min(canonical_N, max(self.N_min + 2, int(self.max_nfe / 3)))

        self.N = self.N_init
        self.pbest_initial = 0.2
        self.pbest_final = 0.3
        self.p_archive = 0.5
        self.rank_pressure_coeff = 4.0
        self.lbc_m = 1.5
        self.lbc_p_f_initial = 3.5
        self.lbc_p_cr_initial = 1.0
        self.lbc_p_final = 1.5

    @staticmethod
    def _generalized_weighted_lehmer(
        values: np.ndarray,
        weights: np.ndarray,
        p: float,
        m: float = 1.5
    ) -> float:
        safe_vals = np.maximum(values, 1e-12)
        num = float(np.sum(weights * np.power(safe_vals, p)))
        den = float(np.sum(weights * np.power(safe_vals, p - m)))
        if abs(den) <= 1e-30 or not np.isfinite(den):
            return float(np.clip(np.average(values, weights=weights), 0.0, 1.0))
        return float(np.clip(num / den, 0.0, 1.0))

    def _sample_cauchy_scalar(self, center: float) -> float:
        for _ in range(100):
            val = float(center + 0.1 * np.tan(np.pi * (self.rng.random() - 0.5)))
            if val > 0.0:
                return min(val, 1.0)
        return min(max(center, 0.5), 1.0)

    def optimize(self) -> OptimizationResult:
        start_time = time.perf_counter()
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        bound_range = high - low

        pop = low + self.rng.random((self.N, self.dim)) * bound_range
        fits = self.func(pop)
        nfe = self.N

        best_idx = int(np.argmin(fits))
        best_x = pop[best_idx].copy()
        best_f = float(fits[best_idx])

        hist_best = [best_f]
        hist_evals = [nfe]

        M_F = np.full(self.H, 0.5, dtype=np.float64)
        M_CR = np.full(self.H, 0.9, dtype=np.float64)
        k_mem = 0

        archive = np.empty((0, self.dim), dtype=np.float64)
        archive_fits = np.empty(0, dtype=np.float64)
        generation = 0

        while nfe < self.max_nfe:
            generation += 1
            progress = float(np.clip(nfe / self.max_nfe, 0.0, 1.0))

            order = np.argsort(fits)
            pop = pop[order]
            fits = fits[order]

            p_val = self.pbest_initial + (self.pbest_final - self.pbest_initial) * progress
            pnum = int(np.clip(round(self.N * p_val), 2, max(2, self.N)))

            mem_indices = self.rng.integers(0, self.H, size=self.N)
            raw_cr = np.clip(self.rng.normal(M_CR[mem_indices], 0.1), 0.0, 1.0)
            # Crossover rate sorting: menores CR para individuos con mejor rango
            CRs = np.sort(raw_cr)

            rank_positions = np.arange(1, self.N + 1, dtype=np.float64)
            rsp_weights = np.exp(-self.rank_pressure_coeff * rank_positions / float(self.N))
            rsp_weights /= np.sum(rsp_weights)
            rsp_cdf = np.cumsum(rsp_weights)
            rsp_cdf[-1] = 1.0

            trial_pop = np.empty_like(pop)
            Fs = np.empty(self.N, dtype=np.float64)
            arc_len = len(archive)

            for i in range(self.N):
                cr_i = float(CRs[i])
                m_f_val = float(M_F[mem_indices[i]])

                # Regeneración acotada hasta 100 intentos con fallback de punto medio (canónico NL-SHADE-LBC)
                F = 0.5
                u = pop[i].copy()
                for _ in range(100):
                    F = self._sample_cauchy_scalar(m_f_val)
                    pbest_idx = int(self.rng.integers(0, pnum))
                    if pbest_idx == i:
                        pbest_idx = (pbest_idx + 1) % pnum

                    while True:
                        r1_idx = int(self.rng.integers(0, self.N))
                        if r1_idx != i and r1_idx != pbest_idx:
                            break

                    if arc_len > 0 and self.rng.random() < self.p_archive:
                        r2_vec = archive[int(self.rng.integers(0, arc_len))]
                    else:
                        r2_idx = i
                        for _ in range(32):
                            cand = int(np.searchsorted(rsp_cdf, self.rng.random(), side="right"))
                            if cand >= self.N:
                                cand = self.N - 1
                            if cand != i and cand != pbest_idx and cand != r1_idx:
                                r2_idx = cand
                                break
                        else:
                            for cand in range(self.N):
                                if cand != i and cand != pbest_idx and cand != r1_idx:
                                    r2_idx = cand
                                    break
                        r2_vec = pop[r2_idx]

                    donor = pop[i] + F * (pop[pbest_idx] - pop[i]) + F * (pop[r1_idx] - r2_vec)
                    cross = self.rng.random(self.dim) < cr_i
                    cross[int(self.rng.integers(0, self.dim))] = True
                    u = np.where(cross, donor, pop[i])
                    if np.all((u >= low) & (u <= high)):
                        break

                # Reparación por punto medio si aún excede fronteras tras regeneración
                below = u < low
                above = u > high
                if np.any(below):
                    u[below] = 0.5 * (low[below] + pop[i, below])
                if np.any(above):
                    u[above] = 0.5 * (high[above] + pop[i, above])

                trial_pop[i] = np.clip(u, low, high)
                Fs[i] = F

            evals_batch = min(self.N, self.max_nfe - nfe)
            trial_fits = self.func(trial_pop[:evals_batch])
            nfe += evals_batch

            parent_fits_batch = fits[:evals_batch]
            strict_mask = trial_fits < parent_fits_batch
            survive_mask = trial_fits <= parent_fits_batch

            if np.any(strict_mask):
                capacity = max(1, int(round(self.arc_rate * self.N)))
                for old_x, old_f in zip(pop[:evals_batch][strict_mask], parent_fits_batch[strict_mask]):
                    if len(archive) < capacity:
                        archive = np.vstack([archive, old_x]) if len(archive) > 0 else old_x.reshape(1, -1).copy()
                        archive_fits = np.append(archive_fits, float(old_f))
                    else:
                        # Reemplazo consciente del fitness en el archivo
                        worse_indices = np.where(old_f < archive_fits)[0]
                        if len(worse_indices) > 0:
                            rep_idx = int(self.rng.choice(worse_indices))
                        else:
                            rep_idx = int(self.rng.integers(0, len(archive)))
                        archive[rep_idx] = old_x
                        archive_fits[rep_idx] = float(old_f)

                sf = Fs[:evals_batch][strict_mask]
                scr = CRs[:evals_batch][strict_mask]
                df = parent_fits_batch[strict_mask] - trial_fits[strict_mask]
                weights = df / (float(np.sum(df)) + 1e-30)

                # Linear Bias Change (LBC) para los exponentes p_F y p_CR
                p_f = self.lbc_p_f_initial + (self.lbc_p_final - self.lbc_p_f_initial) * progress
                p_cr = self.lbc_p_cr_initial + (self.lbc_p_final - self.lbc_p_cr_initial) * progress

                M_F[k_mem] = self._generalized_weighted_lehmer(sf, weights, p_f, self.lbc_m)
                if np.max(scr) <= 0.0:
                    M_CR[k_mem] = 0.0
                else:
                    M_CR[k_mem] = self._generalized_weighted_lehmer(scr, weights, p_cr, self.lbc_m)
                k_mem = (k_mem + 1) % self.H

            pop[:evals_batch][survive_mask] = trial_pop[:evals_batch][survive_mask]
            fits[:evals_batch][survive_mask] = trial_fits[survive_mask]

            curr_best_idx = int(np.argmin(fits))
            if fits[curr_best_idx] < best_f:
                best_f = float(fits[curr_best_idx])
                best_x = pop[curr_best_idx].copy()

            # Reducción No Lineal de Población (NLPSR)
            if nfe < self.max_nfe:
                prog_after = float(np.clip(nfe / self.max_nfe, 0.0, 1.0))
                target_N = int(round((self.N_min - self.N_init) * (prog_after ** (1.0 - prog_after)) + self.N_init))
                target_N = int(np.clip(target_N, self.N_min, self.N))
                if target_N < self.N:
                    keep = np.argsort(fits)[:target_N]
                    pop = pop[keep]
                    fits = fits[keep]
                    self.N = target_N

                max_arc = max(1, int(round(self.arc_rate * self.N)))
                if len(archive) > max_arc:
                    keep_arc = self.rng.choice(len(archive), size=max_arc, replace=False)
                    archive = archive[keep_arc]
                    archive_fits = archive_fits[keep_arc]

            hist_best.append(best_f)
            hist_evals.append(nfe)

        return OptimizationResult(
            best_position=best_x,
            best_fitness=best_f,
            total_evaluations=nfe,
            generations=generation,
            history_best_fitness=hist_best,
            history_evaluations=hist_evals,
            execution_time=time.perf_counter() - start_time
        )

# ==============================================================================
#  9. MadDE (Biswas et al., IEEE CEC 2021 Multi-Adaptation DE)
# ==============================================================================

class MadDE:
    """
    MadDE: Multiple Adaptation Differential Evolution.
    Referencia Canónica: Biswas, S., Saha, D., De, S., Cobb, A. D., Das, S., &
    Jalaian, B. A. (2021). Improving Differential Evolution through Bayesian
    Hyperparameter Optimization. Proc. IEEE CEC 2021, pp. 832-840.
    DOI: 10.1109/CEC45853.2021.9504725
    """

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        max_evaluations: int = 50000,
        memory_size: Optional[int] = None,
        pop_size_init: Optional[int] = None,
        pbest_rate: float = 0.18,
        arc_rate: float = 2.3,
        qbx_probability: float = 0.01,
        seed: Optional[int] = None
    ) -> None:
        self.func = objective_func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.max_nfe = int(max_evaluations)
        self.H = int(10 * self.dim) if memory_size is None else int(memory_size)
        self.pbest_rate = float(pbest_rate)
        self.arc_rate = float(arc_rate)
        self.qbx_probability = float(qbx_probability)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.N_min = 4
        if pop_size_init is not None:
            self.N_init = int(pop_size_init)
        else:
            # Fórmula canónica de MadDE (Biswas et al., 2021): N_init = 2 * D^2
            canonical_N = 2 * (self.dim ** 2)
            self.N_init = min(canonical_N, max(self.N_min + 2, int(self.max_nfe / 3)))

        self.N = self.N_init

    @staticmethod
    def _weighted_lehmer(values: np.ndarray, weights: np.ndarray) -> float:
        denom = float(np.sum(weights * values))
        if denom <= 1e-30 or not np.isfinite(denom):
            return float(np.clip(np.average(values, weights=weights), 0.0, 1.0))
        return float(np.clip(np.sum(weights * (values ** 2)) / denom, 0.0, 1.0))

    def _init_positions(self, low: np.ndarray, bound_range: np.ndarray) -> np.ndarray:
        try:
            import warnings
            from scipy.stats import qmc
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.seed)
                unit = sampler.random(self.N)
            return low + unit * bound_range
        except Exception:
            return low + self.rng.random((self.N, self.dim)) * bound_range

    def _sample_cauchy_batch(self, centers: np.ndarray) -> np.ndarray:
        n = len(centers)
        res = np.zeros(n, dtype=np.float64)
        unfilled = np.ones(n, dtype=bool)
        while np.any(unfilled):
            cnt = int(np.sum(unfilled))
            vals = centers[unfilled] + 0.1 * np.tan(np.pi * (self.rng.random(cnt) - 0.5))
            valid = vals > 0.0
            valid_idx = np.where(unfilled)[0][valid]
            res[valid_idx] = np.minimum(vals[valid], 1.0)
            unfilled[valid_idx] = False
        return res

    def optimize(self) -> OptimizationResult:
        start_time = time.perf_counter()
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        bound_range = high - low

        pop = self._init_positions(low, bound_range)
        fits = self.func(pop)
        nfe = self.N

        best_idx = int(np.argmin(fits))
        best_x = pop[best_idx].copy()
        best_f = float(fits[best_idx])

        hist_best = [best_f]
        hist_evals = [nfe]

        M_F = np.full(self.H, 0.2, dtype=np.float64)
        M_CR = np.full(self.H, 0.2, dtype=np.float64)
        k_mem = 0
        mut_probs = np.full(3, 1.0 / 3.0, dtype=np.float64)

        archive = np.empty((0, self.dim), dtype=np.float64)
        archive_fits = np.empty(0, dtype=np.float64)
        generation = 0

        while nfe < self.max_nfe:
            generation += 1
            progress = float(np.clip(nfe / self.max_nfe, 0.0, 1.0))

            order = np.argsort(fits)
            pop = pop[order]
            fits = fits[order]

            combined_pop = np.vstack([pop, archive]) if len(archive) > 0 else pop
            combined_fits = np.concatenate([fits, archive_fits]) if len(archive) > 0 else fits
            combined_order = np.argsort(combined_fits)

            q_rate = float(np.clip(2.0 * self.pbest_rate - self.pbest_rate * progress, 1e-12, 1.0))
            attraction = 0.5 + 0.5 * progress
            pnum = min(self.N, max(2, int(round(self.pbest_rate * self.N))))
            qnum = min(self.N, max(2, int(round(q_rate * self.N))))
            qcombined_num = min(len(combined_pop), max(2, int(round(q_rate * len(combined_pop)))))

            mem_indices = self.rng.integers(0, self.H, size=self.N)
            Fs = self._sample_cauchy_batch(M_F[mem_indices])
            raw_m_cr = M_CR[mem_indices]
            CRs = np.where(
                raw_m_cr < 0.0,
                0.0,
                np.clip(self.rng.normal(np.maximum(raw_m_cr, 0.0), 0.1), 0.0, 1.0)
            )

            strategies = self.rng.choice(3, size=self.N, p=mut_probs / np.sum(mut_probs))
            trial_pop = np.empty_like(pop)
            total_pool = len(combined_pop)

            for i in range(self.N):
                F = Fs[i]
                CR = CRs[i]
                m = int(strategies[i])
                parent = pop[i]

                while True:
                    r1 = int(self.rng.integers(0, self.N))
                    if r1 != i:
                        break

                while True:
                    r2_comb = int(self.rng.integers(0, total_pool))
                    if r2_comb != i and r2_comb != r1:
                        break

                if m == 0:
                    # DE/current-to-pbest/1 + archive
                    pbest = pop[int(self.rng.integers(0, pnum))]
                    donor = parent + F * (pbest - parent + pop[r1] - combined_pop[r2_comb])
                elif m == 1:
                    # DE/current-to-rand/1 + archive
                    donor = parent + F * (pop[r1] - combined_pop[r2_comb])
                else:
                    # DE/weighted-rand-to-qbest/1
                    while True:
                        r3 = int(self.rng.integers(0, self.N))
                        if r3 != i and r3 != r1:
                            break
                    qbest = pop[int(self.rng.integers(0, qnum))]
                    donor = F * (pop[r1] + attraction * (qbest - pop[r3]))

                # Reparación de frontera del punto medio
                below = donor < low
                above = donor > high
                if np.any(below):
                    donor[below] = 0.5 * (parent[below] + low[below])
                if np.any(above):
                    donor[above] = 0.5 * (parent[above] + high[above])

                # Cruce Binomial con q-best Crossover (QBX) probabilístico
                if self.rng.random() <= self.qbx_probability:
                    base = combined_pop[combined_order[int(self.rng.integers(0, qcombined_num))]]
                else:
                    base = parent

                mask = self.rng.random(self.dim) <= CR
                mask[int(self.rng.integers(0, self.dim))] = True
                trial_pop[i] = np.where(mask, donor, base)

            evals_batch = min(self.N, self.max_nfe - nfe)
            trial_fits = self.func(trial_pop[:evals_batch])
            nfe += evals_batch

            parent_fits_batch = fits[:evals_batch]
            strict_mask = trial_fits < parent_fits_batch
            survive_mask = trial_fits <= parent_fits_batch

            gains = np.zeros(evals_batch, dtype=np.float64)
            gains[strict_mask] = parent_fits_batch[strict_mask] - trial_fits[strict_mask]

            if np.any(strict_mask):
                defeated_x = pop[:evals_batch][strict_mask]
                defeated_f = parent_fits_batch[strict_mask]
                archive = np.vstack([archive, defeated_x]) if len(archive) > 0 else defeated_x.copy()
                archive_fits = np.concatenate([archive_fits, defeated_f]) if len(archive_fits) > 0 else defeated_f.copy()

                weights = gains[strict_mask] / (float(np.sum(gains[strict_mask])) + 1e-30)
                good_f = Fs[:evals_batch][strict_mask]
                good_cr = CRs[:evals_batch][strict_mask]

                M_F[k_mem] = self._weighted_lehmer(good_f, weights)
                if M_CR[k_mem] < 0.0 or float(np.max(good_cr)) == 0.0:
                    M_CR[k_mem] = -1.0
                else:
                    M_CR[k_mem] = self._weighted_lehmer(good_cr, weights)
                k_mem = (k_mem + 1) % self.H
            else:
                M_F[k_mem] = 0.5
                M_CR[k_mem] = 0.5

            # Adaptación de probabilidades de las 3 estrategias de mutación
            scores = np.zeros(3, dtype=np.float64)
            rel_improvement = gains / np.maximum(np.abs(parent_fits_batch), 1e-30)
            strat_batch = strategies[:evals_batch]
            for m in range(3):
                m_mask = strat_batch == m
                scores[m] = float(np.mean(rel_improvement[m_mask])) if np.any(m_mask) else 0.0

            if float(np.sum(scores)) > 0.0 and np.all(np.isfinite(scores)):
                mut_probs = np.clip(scores / np.sum(scores), 0.1, 0.9)
                mut_probs /= np.sum(mut_probs)
            else:
                mut_probs[:] = 1.0 / 3.0

            pop[:evals_batch][survive_mask] = trial_pop[:evals_batch][survive_mask]
            fits[:evals_batch][survive_mask] = trial_fits[survive_mask]

            curr_best_idx = int(np.argmin(fits))
            if fits[curr_best_idx] < best_f:
                best_f = float(fits[curr_best_idx])
                best_x = pop[curr_best_idx].copy()

            # Reducción Lineal de Población (LPSR)
            if nfe < self.max_nfe:
                prog_after = float(np.clip(nfe / self.max_nfe, 0.0, 1.0))
                target_N = max(self.N_min, int(round(self.N_init + (self.N_min - self.N_init) * prog_after)))
                if target_N < self.N:
                    keep = np.argsort(fits)[:target_N]
                    pop = pop[keep]
                    fits = fits[keep]
                    self.N = target_N

                max_arc = max(0, int(round(self.arc_rate * self.N)))
                if len(archive) > max_arc:
                    if max_arc == 0:
                        archive = np.empty((0, self.dim), dtype=np.float64)
                        archive_fits = np.empty(0, dtype=np.float64)
                    else:
                        keep_arc = self.rng.choice(len(archive), size=max_arc, replace=False)
                        archive = archive[keep_arc]
                        archive_fits = archive_fits[keep_arc]

            hist_best.append(best_f)
            hist_evals.append(nfe)

        return OptimizationResult(
            best_position=best_x,
            best_fitness=best_f,
            total_evaluations=nfe,
            generations=generation,
            history_best_fitness=hist_best,
            history_evaluations=hist_evals,
            execution_time=time.perf_counter() - start_time
        )


