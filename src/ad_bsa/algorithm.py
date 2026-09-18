"""
================================================================================
  AD-BSA: Adaptive Differential Boogeyman Search Algorithm
  Module: src/ad_bsa/algorithm.py
  Core Algorithm with Bounded Cosecant Repulsion Barrier |csc(x)|,
  Multi-Boogeyman Anti-Attractors, Thermodynamic Annealing, Lehmer Memory,
  External Diversity Archive, and Linear Population Size Reduction (LPSR).
================================================================================
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from .utils import reflect_boundaries


@dataclass
class OptimizationResult:
    """Contenedor estructurado con los resultados de optimización de AD-BSA."""
    best_position: np.ndarray
    best_fitness: float
    total_evaluations: int
    generations: int
    history_best_fitness: List[float] = field(default_factory=list)
    history_evaluations: List[int] = field(default_factory=list)
    history_population_size: List[int] = field(default_factory=list)
    execution_time: float = 0.0

    @property
    def safe_house_position(self) -> np.ndarray:
        return self.best_position

    @property
    def safe_house_fitness(self) -> float:
        return self.best_fitness


class AD_BSA:
    """
    Adaptive Differential Boogeyman Search Algorithm (AD-BSA).

    Metaheurística continua de optimización global bioinspirada en la dinámica de
    aprendizaje negativo (Negative Learning / Anti-Attractors) y escape armónico.

    Componentes Arquitectónicos Clave:
      1. Multi-Boogeyman Clustering: Identificación de centroides anti-atractores
         en el k% de las peores soluciones (cuencas de estancamiento subóptimas).
      2. Operador de Repulsión Cosecante Acotada |csc(x)|: Barrera de potencial
         no lineal que expulsa agresivamente soluciones del epicentro de la trampa
         sin perturbar a individuos en cuencas prometedoras o valles profundos.
      3. Enfriamiento Termodinámico: Decaimiento temporal suave de la fuerza de
         escape F_escape(t) = F * (1 - t/MaxNFE)^gamma.
      4. Memorias Históricas de Lehmer: Auto-adaptación paramétrica guiada por
         mejoras reales de fitness (Delta f).
      5. Archivo Histórico de Diversidad (A): Selección de vectores diferenciales
         en P U A para preservar direcciones ortogonales de búsqueda.
      6. Reducción Lineal de Población (LPSR): Concentración progresiva del
         esfuerzo computacional desde exploración global hacia explotación fina.
    """

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        max_evaluations: int = 50000,
        pop_size_max: Optional[int] = None,
        pop_size_min: int = 4,
        memory_size: int = 6,
        k_boogeyman_ratio: float = 0.15,
        num_boogeymen: int = 2,
        p_safe_ratio: float = 0.11,
        annealing_power: float = 1.5,
        archive_factor: float = 1.4,
        max_repulsion_force: float = 4.0,
        seed: Optional[int] = None
    ) -> None:
        """
        Inicializa el algoritmo AD-BSA.

        :param objective_func: Función f(x) a minimizar (acepta vectores 1D o lotes 2D N x D).
        :param bounds: Array (D, 2) con cotas inferiores y superiores [(lb_0, ub_0), ...].
        :param max_evaluations: Presupuesto total de llamadas a la función objetivo (MaxNFE).
        :param pop_size_max: Población inicial (si es None, se auto-calibra a min(18*D, max(50, MaxNFE/200))).
        :param pop_size_min: Tamaño mínimo de población al final de LPSR (def: 4).
        :param memory_size: Tamaño de las memorias históricas de Lehmer H (def: 6).
        :param k_boogeyman_ratio: Proporción de peores individuos que forman el Cuco (def: 15%).
        :param num_boogeymen: Número de centroides anti-atractores simultáneos (def: 2).
        :param p_safe_ratio: Proporción de mejores soluciones (p-best) candidatas al Refugio (def: 11%).
        :param annealing_power: Exponente de enfriamiento termodinámico gamma (def: 1.5).
        :param archive_factor: Factor de capacidad del archivo externo |A| = archive_factor * N (def: 1.4).
        :param max_repulsion_force: Cota superior de la repulsión cosecante |csc| (def: 4.0).
        :param seed: Semilla pseudoaleatoria para reproducibilidad.
        """
        self.objective_func = objective_func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.lower_bounds = self.bounds[:, 0]
        self.upper_bounds = self.bounds[:, 1]
        self.bound_range = self.upper_bounds - self.lower_bounds
        self.max_evaluations = int(max_evaluations)
        self.pop_size_min = int(pop_size_min)
        self.memory_size = int(memory_size)
        self.k_boogeyman_ratio = float(k_boogeyman_ratio)
        self.num_boogeymen = int(num_boogeymen)
        self.p_safe_ratio = float(p_safe_ratio)
        self.annealing_power = float(annealing_power)
        self.archive_factor = float(archive_factor)
        self.max_repulsion_force = float(max_repulsion_force)

        self.rng = np.random.default_rng(seed)

        # Calibración dinámica de población inicial según el presupuesto
        if pop_size_max is None:
            calc_pop = int(min(18 * self.dim, max(50, self.max_evaluations / 200)))
            self.pop_size_max = max(self.pop_size_min + 2, calc_pop)
        else:
            self.pop_size_max = int(pop_size_max)

        self.current_pop_size = self.pop_size_max

        # Estructuras de datos poblacionales
        self.Children: np.ndarray = np.empty((0, self.dim), dtype=np.float64)
        self.fitness: np.ndarray = np.empty(0, dtype=np.float64)
        self.Archive: np.ndarray = np.empty((0, self.dim), dtype=np.float64)

        # Registro del óptimo global (Safe House)
        self.best_position: np.ndarray = np.empty(self.dim, dtype=np.float64)
        self.best_fitness: float = float("inf")

        # Memorias históricas adaptativas de Lehmer (H)
        self.Memory_F_safe = np.full(self.memory_size, 0.5, dtype=np.float64)
        self.Memory_F_escape = np.full(self.memory_size, 0.5, dtype=np.float64)
        self.Memory_F_diff = np.full(self.memory_size, 0.5, dtype=np.float64)
        self.Memory_CR = np.full(self.memory_size, 0.5, dtype=np.float64)
        self.memory_pointer = 0

        # Contadores de ejecución
        self.evaluations_count = 0
        self.generation = 0

    def _init_population(self) -> None:
        """Inicializa la población uniformemente en el hipercubo delimitado."""
        self.Children = self.lower_bounds + self.rng.random((self.current_pop_size, self.dim)) * self.bound_range
        self.fitness = self.objective_func(self.Children)
        self.evaluations_count = self.current_pop_size

        best_idx = np.argmin(self.fitness)
        self.best_fitness = float(self.fitness[best_idx])
        self.best_position = self.Children[best_idx].copy()

    def _awaken_multi_boogeymen(self) -> Tuple[np.ndarray, float]:
        """
        Calcula M anti-atractores agrupando los peores individuos (cuencas de estancamiento)
        y determina el radio medio de captura.
        """
        k_count = max(self.num_boogeymen * 2, int(np.ceil(self.k_boogeyman_ratio * self.current_pop_size)))
        worst_indices = np.argsort(self.fitness)[-k_count:]
        worst_individuals = self.Children[worst_indices]

        # Partición en clusters (Multi-Boogeyman)
        boogeymen = []
        cluster_size = int(np.ceil(k_count / self.num_boogeymen))
        for m in range(self.num_boogeymen):
            c_slice = worst_individuals[m * cluster_size : (m + 1) * cluster_size]
            if len(c_slice) > 0:
                boogeymen.append(np.mean(c_slice, axis=0))
            else:
                boogeymen.append(worst_individuals[self.rng.integers(0, k_count)])

        # Radio de captura como fracción de la dispersión de la población
        pop_std = np.mean(np.std(self.Children, axis=0))
        capture_radius = max(0.5 * pop_std, 1e-12)

        return np.array(boogeymen), capture_radius

    def _sample_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Genera parámetros F_safe, F_escape, F_diff y CR vía memorias de Lehmer y Cauchy."""
        mem_indices = self.rng.integers(0, self.memory_size, size=self.current_pop_size)

        def sample_cauchy(mu_arr: np.ndarray) -> np.ndarray:
            res = np.zeros(self.current_pop_size, dtype=np.float64)
            unfilled = np.ones(self.current_pop_size, dtype=bool)
            while np.any(unfilled):
                count = np.sum(unfilled)
                rand_cauchy = mu_arr[unfilled] + 0.1 * np.tan(np.pi * (self.rng.random(count) - 0.5))
                valid = rand_cauchy > 0.0
                res_indices = np.where(unfilled)[0][valid]
                res[res_indices] = np.minimum(rand_cauchy[valid], 1.0)
                unfilled[res_indices] = False
            return res

        F_safe = sample_cauchy(self.Memory_F_safe[mem_indices])
        raw_F_escape = sample_cauchy(self.Memory_F_escape[mem_indices])
        F_diff = sample_cauchy(self.Memory_F_diff[mem_indices])

        # Enfriamiento termodinámico
        progress = self.evaluations_count / self.max_evaluations
        annealing_factor = max(0.0, (1.0 - progress) ** self.annealing_power)
        F_escape = raw_F_escape * annealing_factor

        CR = np.clip(self.rng.normal(self.Memory_CR[mem_indices], 0.1), 0.0, 1.0)
        return F_safe, F_escape, F_diff, CR

    def _select_mutation_partners(
        self,
        boogeymen_pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Selecciona:
          - x_best_p: vector guía dentro del top p% (p-best).
          - S_closest: el Cuco más cercano a cada individuo.
          - x_r1: individuo aleatorio de la población actual (r1 != i).
          - x_r2: solución aleatoria del archivo extendido P U A (r2 != r1 != i).
        """
        N = self.current_pop_size

        # 1. P-Best Selection
        p_num = max(2, int(np.ceil(self.p_safe_ratio * N)))
        sorted_indices = np.argsort(self.fitness)
        top_p_indices = sorted_indices[:p_num]
        chosen_p_idx = top_p_indices[self.rng.integers(0, p_num, size=N)]
        x_best_p = self.Children[chosen_p_idx]

        # 2. Asignación del Cuco más cercano
        if len(boogeymen_pos) > 1:
            dists = np.linalg.norm(self.Children[:, np.newaxis, :] - boogeymen_pos[np.newaxis, :, :], axis=2)
            nearest_b_idx = np.argmin(dists, axis=1)
            S_closest = boogeymen_pos[nearest_b_idx]
        else:
            S_closest = np.tile(boogeymen_pos[0], (N, 1))

        # 3. Selección de r1 (de la población actual P, r1 != i)
        idx_matrix = np.tile(np.arange(N), (N, 1))
        mask = ~np.eye(N, dtype=bool)
        candidates_p = idx_matrix[mask].reshape(N, N - 1)
        r1_idx = candidates_p[np.arange(N), self.rng.integers(0, N - 1, size=N)]
        x_r1 = self.Children[r1_idx]

        # 4. Selección de r2 (de P U A, r2 != r1 != i)
        archive_size = len(self.Archive)
        if archive_size > 0:
            pool = np.vstack([self.Children, self.Archive])
        else:
            pool = self.Children

        total_pool_size = len(pool)
        i_arr = np.arange(N)
        low_forbid = np.minimum(i_arr, r1_idx)
        high_forbid = np.maximum(i_arr, r1_idx)

        r2_cand = self.rng.integers(0, total_pool_size - 2, size=N)
        r2_cand += (r2_cand >= low_forbid)
        r2_cand += (r2_cand >= high_forbid)
        x_r2 = pool[r2_cand]

        return x_best_p, S_closest, x_r1, x_r2

    @staticmethod
    def _lehmer_mean(values: np.ndarray, weights: np.ndarray) -> float:
        """Calcula la media de Lehmer ponderada sum(w * x^2) / sum(w * x)."""
        sum_wx = np.sum(weights * values)
        if sum_wx <= 1e-14:
            return float(np.mean(values))
        return float(np.sum(weights * (values**2)) / sum_wx)

    def _update_archive(self, defeated_parents: np.ndarray) -> None:
        """Agrega los padres derrotados al archivo histórico y trunca según capacidad."""
        if len(defeated_parents) == 0:
            return
        self.Archive = np.vstack([self.Archive, defeated_parents])
        max_archive_capacity = int(self.archive_factor * self.current_pop_size)
        if len(self.Archive) > max_archive_capacity:
            survivor_indices = self.rng.choice(len(self.Archive), size=max_archive_capacity, replace=False)
            self.Archive = self.Archive[survivor_indices]

    def _boogeyman_lpsr(self) -> None:
        """Reduce la población linealmente y trunca el archivo proporcionalmente."""
        target_size = int(np.round(
            self.pop_size_max - (self.evaluations_count / self.max_evaluations) * (self.pop_size_max - self.pop_size_min)
        ))
        target_size = max(self.pop_size_min, target_size)

        if target_size < self.current_pop_size:
            survival_order = np.argsort(self.fitness)
            survivors = survival_order[:target_size]
            eliminated = survival_order[target_size:]

            self._update_archive(self.Children[eliminated])

            self.Children = self.Children[survivors]
            self.fitness = self.fitness[survivors]
            self.current_pop_size = target_size

            max_archive_capacity = int(self.archive_factor * self.current_pop_size)
            if len(self.Archive) > max_archive_capacity:
                self.Archive = self.Archive[:max_archive_capacity]

    def optimize(self) -> OptimizationResult:
        """Ejecuta el ciclo de optimización de AD-BSA."""
        start_time = time.perf_counter()
        self._init_population()

        hist_best_fitness = [self.best_fitness]
        hist_evaluations = [self.evaluations_count]
        hist_population_size = [self.current_pop_size]

        while self.evaluations_count < self.max_evaluations:
            self.generation += 1

            # 1. Despertar al Séquito del Cuco y Radio de Captura
            boogeymen_pos, capture_radius = self._awaken_multi_boogeymen()

            # 2. Muestreo de hiperparámetros
            F_safe, F_escape, F_diff, CR = self._sample_parameters()

            # 3. Selección de parejas de mutación
            x_best_p, S_closest, x_r1, x_r2 = self._select_mutation_partners(boogeymen_pos)

            # Vector de atracción p-best
            safe_vector = (x_best_p - self.Children) * F_safe[:, np.newaxis]

            # Vector de perturbación diferencial con archivo
            diff_vector = (x_r1 - x_r2) * F_diff[:, np.newaxis]

            # ------------------------------------------------------------------
            # 4. OPERADOR DE REPULSIÓN COSECANTE ACOTADA |csc(x)|
            # ------------------------------------------------------------------
            diff_escape = self.Children - S_closest  # Dirección hacia afuera del Cuco
            r = np.linalg.norm(diff_escape, axis=1, keepdims=True) + 1e-8
            r_norm = r / (2.0 * capture_radius + 1e-8)

            # Argumento acotado en (0, pi) para evitar singularidades infinitas
            arg = np.clip(r_norm * np.pi * 0.5, 1e-3, 0.999 * np.pi)
            raw_csc = np.abs(1.0 / np.sin(arg))

            # Barrera acotada en [1.0, max_repulsion_force] con desplazamiento base - 1.0
            force_profile = np.clip(raw_csc, 1.0, self.max_repulsion_force) - 1.0

            # Corte estricto a cero cuando la solución está fuera del radio de peligro (r_norm >= 2.0)
            force_profile = np.where(r_norm < 2.0, force_profile, 0.0)

            unit_escape = diff_escape / r
            escape_vector = F_escape[:, np.newaxis] * force_profile * unit_escape * capture_radius

            # Mutante compuesto
            mutant_children = reflect_boundaries(
                self.Children + safe_vector + escape_vector + diff_vector,
                lb=self.lower_bounds,
                ub=self.upper_bounds,
                base=self.Children
            )

            # 5. Cruce Binomial Vectorizado
            rand_j = self.rng.integers(0, self.dim, size=self.current_pop_size)
            j_matrix = np.tile(np.arange(self.dim), (self.current_pop_size, 1))
            j_rand_mask = (j_matrix == rand_j[:, np.newaxis])
            crossover_mask = (self.rng.random((self.current_pop_size, self.dim)) <= CR[:, np.newaxis]) | j_rand_mask
            trial_children = np.where(crossover_mask, mutant_children, self.Children)

            evals_to_run = min(self.current_pop_size, self.max_evaluations - self.evaluations_count)
            if evals_to_run < self.current_pop_size:
                trial_children = trial_children[:evals_to_run]

            trial_fitness = self.objective_func(trial_children)
            self.evaluations_count += evals_to_run

            # 6. Selección Codiciosa y Archivo de Derrotados
            improvements_mask = trial_fitness <= self.fitness[:evals_to_run]
            successful_indices = np.where(improvements_mask)[0]

            fitness_deltas = np.zeros(evals_to_run, dtype=np.float64)
            fitness_deltas[successful_indices] = self.fitness[successful_indices] - trial_fitness[successful_indices]

            # Archivar a los padres superados
            self._update_archive(self.Children[successful_indices])

            self.Children[successful_indices] = trial_children[successful_indices]
            self.fitness[successful_indices] = trial_fitness[successful_indices]

            best_idx_now = np.argmin(self.fitness)
            if self.fitness[best_idx_now] < self.best_fitness:
                self.best_fitness = float(self.fitness[best_idx_now])
                self.best_position = self.Children[best_idx_now].copy()

            # 7. Actualización de Memorias Históricas de Lehmer
            if len(successful_indices) > 0:
                weights = fitness_deltas[successful_indices] / (np.sum(fitness_deltas[successful_indices]) + 1e-14)
                self.Memory_F_safe[self.memory_pointer] = self._lehmer_mean(F_safe[successful_indices], weights)
                self.Memory_F_escape[self.memory_pointer] = self._lehmer_mean(F_escape[successful_indices], weights)
                self.Memory_F_diff[self.memory_pointer] = self._lehmer_mean(F_diff[successful_indices], weights)
                self.Memory_CR[self.memory_pointer] = float(np.sum(weights * CR[successful_indices]))
                self.memory_pointer = (self.memory_pointer + 1) % self.memory_size

            # 8. Reducción Lineal de Población (LPSR)
            if self.evaluations_count < self.max_evaluations:
                self._boogeyman_lpsr()

            hist_best_fitness.append(self.best_fitness)
            hist_evaluations.append(self.evaluations_count)
            hist_population_size.append(self.current_pop_size)

        return OptimizationResult(
            best_position=self.best_position,
            best_fitness=self.best_fitness,
            total_evaluations=self.evaluations_count,
            generations=self.generation,
            history_best_fitness=hist_best_fitness,
            history_evaluations=hist_evaluations,
            history_population_size=hist_population_size,
            execution_time=time.perf_counter() - start_time
        )
