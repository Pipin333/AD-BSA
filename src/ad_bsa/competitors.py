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
================================================================================
"""

import math
import time
from typing import Callable, List, Optional, Tuple
import numpy as np
from scipy import stats

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

from .algorithm import OptimizationResult
from .utils import (
    reflect_boundaries,
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

                f_val = -1.0
                while f_val <= 0:
                    f_val = stats.cauchy.rvs(loc=M_F[r_k], scale=0.1, random_state=self.rng)
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

                p_idx = self.rng.integers(0, p_num)
                x_pbest = pop[p_idx]

                r1_choices = [idx for idx in range(self.N) if idx != i]
                r1 = self.rng.choice(r1_choices)
                x_r1 = pop[r1]

                pool = np.vstack([pop, archive]) if len(archive) > 0 else pop
                r2_choices = [idx for idx in range(len(pool)) if idx != i and idx != r1]
                r2 = self.rng.choice(r2_choices)
                x_r2 = pool[r2]

                v = pop[i] + Fw * (x_pbest - pop[i]) + f_val * (x_r1 - x_r2)
                # Manejo de fronteras canónico del punto medio (SHADE/jSO)
                v = bound_constraint_shade(v[np.newaxis, :], lb=low, ub=high, base=pop[i:i+1])[0]

                j_rand = self.rng.integers(0, self.dim)
                u = np.where((self.rng.random(self.dim) < cr) | (np.arange(self.dim) == j_rand), v, pop[i])
                trial_pop[i] = u

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

                p_idx = self.rng.integers(0, p_num)
                x_pbest = pop[p_idx]

                r1_candidates = [j for j in range(self.N) if j != i]
                r1 = self.rng.choice(r1_candidates)

                pool_size = self.N + len(archive)
                r2_candidates = [j for j in range(pool_size) if j != i and j != r1]
                r2_idx = self.rng.choice(r2_candidates)
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


