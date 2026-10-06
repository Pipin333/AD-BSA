"""AD-BSA: Adaptive Differential Boogeyman Search Algorithm.

Core algorithm implementation featuring:
  - Bounded Cosecant Repulsion Barrier (|csc(x)|)
  - Multi-Boogeyman Anti-Attractors (Fitness-Rank Stratified Partitioning)
  - Pure Thermodynamic Cooling Schedule
  - Historical Lehmer Parameter Memories
  - External Diversity Archive
  - Linear Population Size Reduction (LPSR)
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from .utils import reflect_boundaries


@dataclass
class OptimizationResult:
    """Optimization result container for AD-BSA runs.

    Attributes:
        best_position (np.ndarray): Best coordinate vector found in search space.
        best_fitness (float): Global objective function value at best_position.
        total_evaluations (int): Total number of objective function evaluations.
        generations (int): Total number of generations executed.
        history_best_fitness (List[float]): Best fitness history per generation.
        history_evaluations (List[int]): Cumulative evaluation counter history.
        history_population_size (List[int]): Population size trajectory.
        execution_time (float): Wall-clock optimization time in seconds.
    """

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
        """Alias returning the best position vector discovered."""
        return self.best_position

    @property
    def safe_house_fitness(self) -> float:
        """Alias returning the global best fitness value."""
        return self.best_fitness


class AD_BSA:
    """Adaptive Differential Boogeyman Search Algorithm (AD-BSA).

    Continuous global metaheuristic for high-dimensional multimodal landscapes,
    synthesizing negative learning anti-attractors with bounded cosecant repulsion.

    Key Architectural Mechanisms:
      1. Multi-Boogeyman Anti-Attractors: Stratified fitness-rank partitioning of
         the worst k% candidate solutions into M centroid epicenters in O(k * D).
      2. Bounded Cosecant Repulsion Barrier (|csc|): Non-linear potential field
         diverging near stagnation cores while smoothly vanishing outside 2 * Rc.
      3. Thermodynamic Annealing: Power-law damping (1 - t/MaxNFE)^gamma cleanly
         decoupled from historical parameter success memories.
      4. Historical Lehmer Memories: Parameter auto-adaptation driven by fitness
         improvement deltas (Delta f).
      5. External Diversity Archive: Differential perturbation vectors sampled
         from P U A to preserve orthogonal search directions.
      6. Linear Population Size Reduction (LPSR): Dynamic shrinkage from broad
         global exploration toward localized exploitation.
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
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the AD-BSA optimizer.

        Args:
            objective_func: Callable objective accepting 1D or 2D NumPy arrays.
            bounds: Boundary array of shape (D, 2) defining (lb, ub) per dimension.
            max_evaluations: Total evaluation budget (MaxNFE). Defaults to 50000.
            pop_size_max: Initial population size. Dynamically set if None.
            pop_size_min: Minimum terminal population size under LPSR. Defaults to 4.
            memory_size: Capacity of historical Lehmer memory buffers. Defaults to 6.
            k_boogeyman_ratio: Proportion of worst solutions forming anti-attractors.
            num_boogeymen: Number of stratified anti-attractor centroids.
            p_safe_ratio: Elite top-tier ratio for p-best positive attraction.
            annealing_power: Power-law exponent gamma for thermodynamic cooling.
            archive_factor: External archive size multiplier |A| = factor * N.
            max_repulsion_force: Upper clipping bound for the cosecant barrier.
            seed: Pseudo-random generator seed for deterministic reproducibility.
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

        if pop_size_max is None:
            calc_pop = int(
                min(18 * self.dim, max(50, self.max_evaluations / 200))
            )
            self.pop_size_max = max(self.pop_size_min + 2, calc_pop)
        else:
            self.pop_size_max = int(pop_size_max)

        self.current_pop_size = self.pop_size_max

        # Population structures
        self.Children: np.ndarray = np.empty((0, self.dim), dtype=np.float64)
        self.fitness: np.ndarray = np.empty(0, dtype=np.float64)
        self.Archive: np.ndarray = np.empty((0, self.dim), dtype=np.float64)

        # Global best record (Safe House)
        self.best_position: np.ndarray = np.empty(self.dim, dtype=np.float64)
        self.best_fitness: float = float("inf")

        # Historical Lehmer parameter memories (H)
        self.Memory_F_safe = np.full(self.memory_size, 0.5, dtype=np.float64)
        self.Memory_F_escape = np.full(self.memory_size, 0.5, dtype=np.float64)
        self.Memory_F_diff = np.full(self.memory_size, 0.5, dtype=np.float64)
        self.Memory_CR = np.full(self.memory_size, 0.5, dtype=np.float64)
        self.memory_pointer = 0

        # Runtime counters
        self.evaluations_count = 0
        self.generation = 0

    def _init_population(self) -> None:
        """Initialize the population uniformly inside the bounding box."""
        self.Children = (
            self.lower_bounds
            + self.rng.random((self.current_pop_size, self.dim)) * self.bound_range
        )
        self.fitness = self.objective_func(self.Children)
        self.evaluations_count = self.current_pop_size

        best_idx = np.argmin(self.fitness)
        self.best_fitness = float(self.fitness[best_idx])
        self.best_position = self.Children[best_idx].copy()

    def _awaken_multi_boogeymen(self) -> Tuple[np.ndarray, float]:
        """Compute M anti-attractors via Fitness-Rank Stratified Partitioning.

        Partitions the worst k% candidates into M contiguous sub-tiers to obtain
        centroid coordinates in O(k * D) without iterative clustering latency.

        Returns:
            Tuple[np.ndarray, float]: Tuple containing (boogeymen_centroids, Rc),
                where boogeymen_centroids has shape (num_boogeymen, D) and Rc
                is the spatial capture radius.
        """
        k_count = max(
            self.num_boogeymen * 2,
            int(np.ceil(self.k_boogeyman_ratio * self.current_pop_size)),
        )
        worst_indices = np.argsort(self.fitness)[-k_count:]
        worst_individuals = self.Children[worst_indices]

        boogeymen = []
        cluster_size = int(np.ceil(k_count / self.num_boogeymen))
        for m in range(self.num_boogeymen):
            c_slice = worst_individuals[
                m * cluster_size:(m + 1) * cluster_size
            ]
            if len(c_slice) > 0:
                boogeymen.append(np.mean(c_slice, axis=0))
            else:
                boogeymen.append(
                    worst_individuals[self.rng.integers(0, k_count)]
                )

        pop_std = np.mean(np.std(self.Children, axis=0))
        capture_radius = max(0.5 * pop_std, 1e-12)

        return np.array(boogeymen), capture_radius

    def _sample_parameters(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample control parameters via historical Cauchy/Gaussian distributions.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                Vectors of (F_safe, F_escape, raw_F_escape, F_diff, CR) for all
                current population members.
        """
        mem_indices = self.rng.integers(
            0, self.memory_size, size=self.current_pop_size
        )

        def sample_cauchy(mu_arr: np.ndarray) -> np.ndarray:
            res = np.zeros(self.current_pop_size, dtype=np.float64)
            unfilled = np.ones(self.current_pop_size, dtype=bool)
            while np.any(unfilled):
                count = np.sum(unfilled)
                rand_cauchy = mu_arr[unfilled] + 0.1 * np.tan(
                    np.pi * (self.rng.random(count) - 0.5)
                )
                valid = rand_cauchy > 0.0
                res_indices = np.where(unfilled)[0][valid]
                res[res_indices] = np.minimum(rand_cauchy[valid], 1.0)
                unfilled[res_indices] = False
            return res

        F_safe = sample_cauchy(self.Memory_F_safe[mem_indices])
        raw_F_escape = sample_cauchy(self.Memory_F_escape[mem_indices])
        F_diff = sample_cauchy(self.Memory_F_diff[mem_indices])

        progress = self.evaluations_count / self.max_evaluations
        annealing_factor = max(0.0, (1.0 - progress) ** self.annealing_power)
        F_escape = raw_F_escape * annealing_factor

        CR = np.clip(
            self.rng.normal(self.Memory_CR[mem_indices], 0.1), 0.0, 1.0
        )
        return F_safe, F_escape, raw_F_escape, F_diff, CR

    def _select_mutation_partners(
        self, boogeymen_pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Select mutation vector partners across population and external archive.

        Args:
            boogeymen_pos: Coordinate array of anti-attractors (M, D).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                Arrays of (x_best_p, S_closest, x_r1, x_r2).
        """
        num_pop = self.current_pop_size

        # 1. P-Best Selection
        p_num = max(2, int(np.ceil(self.p_safe_ratio * num_pop)))
        sorted_indices = np.argsort(self.fitness)
        top_p_indices = sorted_indices[:p_num]
        chosen_p_idx = top_p_indices[
            self.rng.integers(0, p_num, size=num_pop)
        ]
        x_best_p = self.Children[chosen_p_idx]

        # 2. Nearest Anti-Attractor Assignment
        if len(boogeymen_pos) > 1:
            dists = np.linalg.norm(
                self.Children[:, np.newaxis, :]
                - boogeymen_pos[np.newaxis, :, :],
                axis=2,
            )
            nearest_b_idx = np.argmin(dists, axis=1)
            S_closest = boogeymen_pos[nearest_b_idx]
        else:
            S_closest = np.tile(boogeymen_pos[0], (num_pop, 1))

        # 3. Selection of r1 from current population (r1 != i)
        idx_matrix = np.tile(np.arange(num_pop), (num_pop, 1))
        mask = ~np.eye(num_pop, dtype=bool)
        candidates_p = idx_matrix[mask].reshape(num_pop, num_pop - 1)
        r1_idx = candidates_p[
            np.arange(num_pop), self.rng.integers(0, num_pop - 1, size=num_pop)
        ]
        x_r1 = self.Children[r1_idx]

        # 4. Selection of r2 from P U A (r2 != r1 != i)
        archive_size = len(self.Archive)
        if archive_size > 0:
            pool = np.vstack([self.Children, self.Archive])
        else:
            pool = self.Children

        total_pool_size = len(pool)
        i_arr = np.arange(num_pop)
        low_forbid = np.minimum(i_arr, r1_idx)
        high_forbid = np.maximum(i_arr, r1_idx)

        r2_cand = self.rng.integers(0, total_pool_size - 2, size=num_pop)
        r2_cand += r2_cand >= low_forbid
        r2_cand += r2_cand >= high_forbid
        x_r2 = pool[r2_cand]

        return x_best_p, S_closest, x_r1, x_r2

    @staticmethod
    def _lehmer_mean(values: np.ndarray, weights: np.ndarray) -> float:
        """Compute the weighted Lehmer mean sum(w * x^2) / sum(w * x)."""
        sum_wx = np.sum(weights * values)
        if sum_wx <= 1e-14:
            return float(np.mean(values))
        return float(np.sum(weights * (values**2)) / sum_wx)

    def _update_archive(self, defeated_parents: np.ndarray) -> None:
        """Append defeated parents to external archive and enforce capacity."""
        if len(defeated_parents) == 0:
            return
        self.Archive = np.vstack([self.Archive, defeated_parents])
        max_archive_capacity = int(
            self.archive_factor * self.current_pop_size
        )
        if len(self.Archive) > max_archive_capacity:
            survivor_indices = self.rng.choice(
                len(self.Archive), size=max_archive_capacity, replace=False
            )
            self.Archive = self.Archive[survivor_indices]

    def _boogeyman_lpsr(self) -> None:
        """Linearly reduce population size and truncate archive proportionally."""
        target_size = int(
            np.round(
                self.pop_size_max
                - (self.evaluations_count / self.max_evaluations)
                * (self.pop_size_max - self.pop_size_min)
            )
        )
        target_size = max(self.pop_size_min, target_size)

        if target_size < self.current_pop_size:
            survival_order = np.argsort(self.fitness)
            survivors = survival_order[:target_size]
            eliminated = survival_order[target_size:]

            self._update_archive(self.Children[eliminated])

            self.Children = self.Children[survivors]
            self.fitness = self.fitness[survivors]
            self.current_pop_size = target_size

            max_archive_capacity = int(
                self.archive_factor * self.current_pop_size
            )
            if len(self.Archive) > max_archive_capacity:
                survivor_indices = self.rng.choice(
                    len(self.Archive), size=max_archive_capacity, replace=False
                )
                self.Archive = self.Archive[survivor_indices]

    def optimize(self) -> OptimizationResult:
        """Execute the primary AD-BSA optimization loop.

        Returns:
            OptimizationResult: Encapsulated trajectory and optimal solution.
        """
        start_time = time.perf_counter()
        self._init_population()

        hist_best_fitness = [self.best_fitness]
        hist_evaluations = [self.evaluations_count]
        hist_population_size = [self.current_pop_size]

        while self.evaluations_count < self.max_evaluations:
            self.generation += 1

            # 1. Awaken anti-attractor centroids
            boogeymen_pos, capture_radius = self._awaken_multi_boogeymen()

            # 2. Sample hyperparameters
            (
                F_safe,
                F_escape,
                raw_F_escape,
                F_diff,
                CR,
            ) = self._sample_parameters()

            # 3. Select mutation partners
            (
                x_best_p,
                S_closest,
                x_r1,
                x_r2,
            ) = self._select_mutation_partners(boogeymen_pos)

            safe_vector = (x_best_p - self.Children) * F_safe[:, np.newaxis]
            diff_vector = (x_r1 - x_r2) * F_diff[:, np.newaxis]

            # 4. Bounded Cosecant Repulsion Barrier |csc(x)|
            diff_escape = self.Children - S_closest
            r = np.linalg.norm(diff_escape, axis=1, keepdims=True) + 1e-8
            r_norm = r / (2.0 * capture_radius + 1e-8)

            arg = np.clip(r_norm * np.pi * 0.5, 1e-3, 0.999 * np.pi)
            raw_csc = np.abs(1.0 / np.sin(arg))
            force_profile = (
                np.clip(raw_csc, 1.0, self.max_repulsion_force) - 1.0
            )
            force_profile = np.where(r_norm < 2.0, force_profile, 0.0)

            unit_escape = diff_escape / r
            escape_vector = (
                F_escape[:, np.newaxis]
                * force_profile
                * unit_escape
                * capture_radius
            )

            # Composite mutant vector
            mutant_children = reflect_boundaries(
                self.Children + safe_vector + escape_vector + diff_vector,
                lb=self.lower_bounds,
                ub=self.upper_bounds,
                base=self.Children,
            )

            # 5. Binomial Crossover
            rand_j = self.rng.integers(0, self.dim, size=self.current_pop_size)
            j_matrix = np.tile(np.arange(self.dim), (self.current_pop_size, 1))
            j_rand_mask = j_matrix == rand_j[:, np.newaxis]
            crossover_mask = (
                self.rng.random((self.current_pop_size, self.dim))
                <= CR[:, np.newaxis]
            ) | j_rand_mask
            trial_children = np.where(
                crossover_mask, mutant_children, self.Children
            )

            evals_to_run = min(
                self.current_pop_size,
                self.max_evaluations - self.evaluations_count,
            )
            if evals_to_run < self.current_pop_size:
                trial_children = trial_children[:evals_to_run]

            trial_fitness = self.objective_func(trial_children)
            self.evaluations_count += evals_to_run

            # 6. Greedy Selection and Archiving
            improvements_mask = trial_fitness <= self.fitness[:evals_to_run]
            successful_indices = np.where(improvements_mask)[0]

            fitness_deltas = np.zeros(evals_to_run, dtype=np.float64)
            fitness_deltas[successful_indices] = (
                self.fitness[successful_indices]
                - trial_fitness[successful_indices]
            )

            self._update_archive(self.Children[successful_indices])
            self.Children[successful_indices] = trial_children[
                successful_indices
            ]
            self.fitness[successful_indices] = trial_fitness[successful_indices]

            best_idx_now = np.argmin(self.fitness)
            if self.fitness[best_idx_now] < self.best_fitness:
                self.best_fitness = float(self.fitness[best_idx_now])
                self.best_position = self.Children[best_idx_now].copy()

            # 7. Update Historical Lehmer Memories
            if len(successful_indices) > 0:
                weights = fitness_deltas[successful_indices] / (
                    np.sum(fitness_deltas[successful_indices]) + 1e-14
                )
                self.Memory_F_safe[self.memory_pointer] = self._lehmer_mean(
                    F_safe[successful_indices], weights
                )
                self.Memory_F_escape[self.memory_pointer] = self._lehmer_mean(
                    raw_F_escape[successful_indices], weights
                )
                self.Memory_F_diff[self.memory_pointer] = self._lehmer_mean(
                    F_diff[successful_indices], weights
                )
                self.Memory_CR[self.memory_pointer] = float(
                    np.sum(weights * CR[successful_indices])
                )
                self.memory_pointer = (
                    self.memory_pointer + 1
                ) % self.memory_size

            # 8. LPSR Population Reduction
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
            execution_time=time.perf_counter() - start_time,
        )
