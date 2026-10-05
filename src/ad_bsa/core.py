"""
================================================================================
  AD-BSA v2: Adaptive Differential Boogeyman Search Algorithm (Version 2.0)
  "The Nightmare & Sanctuary Edition" - Metaheurística Adaptativa de Élite
================================================================================
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
#  CLASE DE RESULTADOS DE OPTIMIZACIÓN v2
# ==============================================================================

@dataclass
class BoogeymanV2OptimizationResult:
    """Contenedor estructurado con los resultados del AD-BSA v2."""
    safe_house_position: np.ndarray
    safe_house_fitness: float
    total_evaluations: int
    generations: int
    history_best_fitness: List[float] = field(default_factory=list)
    history_evaluations: List[int] = field(default_factory=list)
    history_sack_captures: List[int] = field(default_factory=list)
    history_population_size: List[int] = field(default_factory=list)
    history_boogeyman_strength: List[float] = field(default_factory=list)
    execution_time_seconds: float = 0.0


# ==============================================================================
#  ALGORITMO AD-BSA v2 (ADAPTIVE DIFFERENTIAL BOOGEYMAN SEARCH v2)
# ==============================================================================

class AD_BSA_v2:
    """
    AD-BSA v2 (Adaptive Differential Boogeyman Search Algorithm - Version 2.0).

    Novedades Clave respecto a v1:
      1. Calibración Dinámica de Población Inicial (N_max dependiente del presupuesto MaxNFE).
      2. Multi-Boogeyman Clustering: Repulsión adaptativa desde múltiples anti-atractores locales.
      3. Enfriamiento Termodinámico del Cuco (Boogeyman Annealing): Decaimiento temporal de F_escape
         para deslizarse suavemente en valles estrechos (ej. Rosenbrock) y explotar el óptimo global.
      4. Archivo Histórico de Espíritus Supervivientes (External Archive A): Selección de r2 en P U A.
      5. Reeducación Guiada por P-Best: Proyección de renacimiento desde el Refugio p-best.
    """

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        max_evaluations: int = 100000,
        pop_size_max: Optional[int] = None,
        pop_size_min: int = 4,
        memory_size: int = 6,
        k_boogeyman_ratio: float = 0.15,
        num_boogeymen: int = 2,
        p_safe_ratio: float = 0.11,
        f_boost: float = 0.85,
        annealing_power: float = 1.5,
        archive_factor: float = 1.4,
        seed: Optional[int] = None
    ) -> None:
        """
        Inicializa AD-BSA v2 con todos sus operadores avanzados.

        :param objective_func: Función f(x) a minimizar (vectorizada N x D).
        :param bounds: Array con límites [(min_0, max_0), ..., (min_D, max_D)].
        :param max_evaluations: Presupuesto total de evaluaciones (MaxNFE).
        :param pop_size_max: Tamaño máximo inicial de población (si es None, se auto-calibra).
        :param pop_size_min: Población mínima permitida (N_min = 4).
        :param memory_size: Tamaño de las memorias históricas de Lehmer (H).
        :param k_boogeyman_ratio: Proporción de peores niños que componen los Cucos (15%).
        :param num_boogeymen: Número de anti-atractores simultáneos (Multi-Boogeyman).
        :param p_safe_ratio: Proporción de mejores niños candidatos al Refugio (p-best).
        :param f_boost: Factor de impulso para el salto de renacimiento.
        :param annealing_power: Exponente de enfriamiento de la fuerza de escape del Cuco.
        :param archive_factor: Factor de capacidad del archivo histórico (|A| = archive_factor * N).
        :param seed: Semilla aleatoria para reproducibilidad.
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
        self.f_boost = float(f_boost)
        self.annealing_power = float(annealing_power)
        self.archive_factor = float(archive_factor)

        # 1. Auto-calibración de N_max para garantizar cientos de generaciones de adaptación
        if pop_size_max is None:
            # Escalado óptimo: mínimo 4*D, máximo 18*D, adaptado al presupuesto NFE
            budget_based_pop = int(self.max_evaluations / 160)
            self.pop_size_max = max(4 * self.dim, min(18 * self.dim, budget_based_pop))
        else:
            self.pop_size_max = int(pop_size_max)

        self.rng = np.random.default_rng(seed)

        # Memorias históricas adaptativas de Lehmer (L-SHADE)
        self.Memory_F_safe = np.full(self.memory_size, 0.5, dtype=np.float64)
        self.Memory_F_escape = np.full(self.memory_size, 0.5, dtype=np.float64)
        self.Memory_F_diff = np.full(self.memory_size, 0.5, dtype=np.float64)
        self.Memory_CR = np.full(self.memory_size, 0.5, dtype=np.float64)
        self.memory_index = 0

        # Contadores de cacería
        self.evaluations_count = 0
        self.generation = 0
        self.current_pop_size = self.pop_size_max

        # Entidades: Población (Children), SafeHouse, Cucos y Archivo Histórico
        self.Children: np.ndarray = np.empty((0, self.dim), dtype=np.float64)
        self.fitness: np.ndarray = np.empty(0, dtype=np.float64)
        self.SafeHouse: np.ndarray = np.empty(self.dim, dtype=np.float64)
        self.SafeHouse_fitness: float = np.inf
        self.Boogeymen: np.ndarray = np.empty((0, self.dim), dtype=np.float64)

        # Archivo Externo de Espíritus Supervivientes
        self.Archive: np.ndarray = np.empty((0, self.dim), dtype=np.float64)

    # --------------------------------------------------------------------------
    #  INICIALIZACIÓN Y MANEJO DE FRONTERAS
    # --------------------------------------------------------------------------

    def _initialize_children(self) -> None:
        """Inicializa uniformemente la población de niños y detecta el Refugio inicial."""
        self.current_pop_size = self.pop_size_max
        self.Children = self.lower_bounds + self.rng.random((self.current_pop_size, self.dim)) * self.bound_range
        self.fitness = self.objective_func(self.Children)
        self.evaluations_count = len(self.Children)

        best_idx = np.argmin(self.fitness)
        self.SafeHouse = self.Children[best_idx].copy()
        self.SafeHouse_fitness = float(self.fitness[best_idx])
        self.Archive = np.empty((0, self.dim), dtype=np.float64)

    def _bound_reflection(self, u: np.ndarray, x_base: np.ndarray = None) -> np.ndarray:
        """
        Corrige desbordes mediante Midpoint Reflection formal del paper:
        Si u_j se sale del límite, se reubica en el punto medio entre el padre x_j y el límite.
        """
        clipped = np.copy(u)
        parent = x_base if x_base is not None else self.lower_bounds

        low_viol = clipped < self.lower_bounds
        if x_base is not None:
            clipped = np.where(low_viol, (parent + self.lower_bounds) / 2.0, clipped)
        else:
            clipped = np.where(low_viol, self.lower_bounds, clipped)

        high_viol = clipped > self.upper_bounds
        if x_base is not None:
            clipped = np.where(high_viol, (parent + self.upper_bounds) / 2.0, clipped)
        else:
            clipped = np.where(high_viol, self.upper_bounds, clipped)

        return np.clip(clipped, self.lower_bounds, self.upper_bounds)

    # --------------------------------------------------------------------------
    #  MULTI-BOOGEYMAN CLUSTERING & RADIO DE CAPTURA
    # --------------------------------------------------------------------------

    def _awaken_multi_boogeymen(self) -> Tuple[np.ndarray, float]:
        """
        Calcula M anti-atractores (El Séquito del Cuco) agrupando los peores
        individuos en clusters locales y calcula el radio dinámico de captura.
        """
        N = self.current_pop_size
        raw_k = max(self.num_boogeymen * 2, int(np.ceil(self.k_boogeyman_ratio * N)))
        k = min(N - 1, max(1, raw_k)) if N > 1 else 1
        worst_indices = np.argpartition(self.fitness, -k)[-k:]
        worst_children = self.Children[worst_indices]

        # Multi-Boogeymen clustering formal (K-means vectorizado de Lloyd)
        if self.num_boogeymen > 1 and len(worst_children) >= self.num_boogeymen:
            K = min(self.num_boogeymen, len(worst_children))
            # Inicialización uniforme de centroides
            init_idx = self.rng.choice(len(worst_children), size=K, replace=False)
            centroids = worst_children[init_idx].copy()
            # Iteraciones vectorizadas de Lloyd
            for _ in range(3):
                dists = np.linalg.norm(worst_children[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
                labels = np.argmin(dists, axis=1)
                for k_i in range(K):
                    mask = (labels == k_i)
                    if np.any(mask):
                        centroids[k_i] = np.mean(worst_children[mask], axis=0)
            boogeymen_pos = centroids
        else:
            boogeymen_pos = np.mean(worst_children, axis=0, keepdims=True)

        spatial_sigma = np.mean(np.std(self.Children, axis=0))
        capture_radius = 0.5 * spatial_sigma

        return boogeymen_pos, capture_radius

    # --------------------------------------------------------------------------
    #  MUESTREO ADAPTATIVO CON ENFRIAMIENTO TERMODINÁMICO (ANNEALING)
    # --------------------------------------------------------------------------

    def _sample_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Muestrea hiperparámetros por individuo modulando F_escape con un factor de
        enfriamiento termodinámico (1 - progress)^alpha.
        """
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

        # Factor de enfriamiento termodinámico del Cuco
        progress = self.evaluations_count / self.max_evaluations
        annealing_factor = max(0.0, (1.0 - progress) ** self.annealing_power)
        F_escape = raw_F_escape * annealing_factor

        CR = np.clip(self.rng.normal(self.Memory_CR[mem_indices], 0.1), 0.0, 1.0)

        return F_safe, F_escape, F_diff, CR, annealing_factor

    # --------------------------------------------------------------------------
    #  SELECCIÓN VECTORIZADA CON ARCHIVO HISTÓRICO (P U A)
    # --------------------------------------------------------------------------

    def _select_mutation_partners_v2(
        self,
        boogeymen_pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Selecciona vectorizadamente:
          - x_best_p: vector guía dentro del top p% (Safe House Candidates).
          - S_closest: el Cuco más cercano a cada niño individual.
          - x_r1: niño aleatorio de la población actual (r1 != i).
          - x_r2: solución aleatoria del espacio extendido P U A (r2 != r1 != i).
        """
        N = self.current_pop_size

        # 1. P-Best Selection
        p_num = max(2, int(np.ceil(self.p_safe_ratio * N)))
        sorted_indices = np.argsort(self.fitness)
        top_p_indices = sorted_indices[:p_num]
        chosen_p_idx = top_p_indices[self.rng.integers(0, p_num, size=N)]
        x_best_p = self.Children[chosen_p_idx]

        # 2. Asignación del Cuco más cercano por niño
        if len(boogeymen_pos) > 1:
            # Distancias N x M
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

        # 4. Selección de r2 (de la unión P U A, r2 != r1 != i)
        archive_size = len(self.Archive)
        total_pool_size = N + archive_size
        pool = np.vstack([self.Children, self.Archive]) if archive_size > 0 else self.Children

        # Muestreo uniforme vectorizado de r2 excluyendo {i, r1_idx[i]} sin bucles while
        i_arr = np.arange(N)
        low_forbid = np.minimum(i_arr, r1_idx)
        high_forbid = np.maximum(i_arr, r1_idx)

        r2_cand = self.rng.integers(0, total_pool_size - 2, size=N)
        r2_cand += (r2_cand >= low_forbid)
        r2_cand += (r2_cand >= high_forbid)
        x_r2 = pool[r2_cand]

        return x_best_p, S_closest, x_r1, x_r2
        
    def _compute_escape_vector(self, S_closest: np.ndarray, F_escape: np.ndarray, capture_radius: float) -> np.ndarray:
        """Calcula el vector de escape repulsivo (por defecto lineal de Hooke)."""
        return (self.Children - S_closest) * F_escape[:, np.newaxis]

    # --------------------------------------------------------------------------
    #  MEDIA DE LEHMER PONDERADA
    # --------------------------------------------------------------------------

    @staticmethod
    def _lehmer_mean(values: np.ndarray, weights: np.ndarray) -> float:
        """Calcula la media de Lehmer ponderada sum(w * x^2) / sum(w * x)."""
        sum_wx = np.sum(weights * values)
        if sum_wx <= 1e-14:
            return float(np.mean(values))
        return float(np.sum(weights * (values**2)) / sum_wx)

    # --------------------------------------------------------------------------
    #  GESTIÓN DEL ARCHIVO EXTERNO (A)
    # --------------------------------------------------------------------------

    def _update_archive(self, defeated_parents: np.ndarray) -> None:
        """Agrega los padres derrotados al archivo histórico y trunca según capacidad."""
        if len(defeated_parents) == 0:
            return
        self.Archive = np.vstack([self.Archive, defeated_parents])
        max_archive_capacity = int(self.archive_factor * self.current_pop_size)
        if len(self.Archive) > max_archive_capacity:
            survivor_indices = self.rng.choice(len(self.Archive), size=max_archive_capacity, replace=False)
            self.Archive = self.Archive[survivor_indices]

    # --------------------------------------------------------------------------
    #  REDUCCIÓN LINEAL DE POBLACIÓN (LPSR v2)
    # --------------------------------------------------------------------------

    def _boogeyman_lpsr_v2(self) -> None:
        """Reduce la población linealmente y trunca el archivo proporcionalmente."""
        target_size = int(np.round(
            self.pop_size_max - (self.evaluations_count / self.max_evaluations) * (self.pop_size_max - self.pop_size_min)
        ))
        target_size = max(self.pop_size_min, target_size)

        if target_size < self.current_pop_size:
            survival_order = np.argsort(self.fitness)
            survivors = survival_order[:target_size]
            eliminated = survival_order[target_size:]

            # Los niños eliminados por LPSR también enriquecen el archivo histórico
            self._update_archive(self.Children[eliminated])

            self.Children = self.Children[survivors]
            self.fitness = self.fitness[survivors]
            self.current_pop_size = target_size

            # Ajustar capacidad del archivo
            max_archive_capacity = int(self.archive_factor * self.current_pop_size)
            if len(self.Archive) > max_archive_capacity:
                self.Archive = self.Archive[:max_archive_capacity]

    # --------------------------------------------------------------------------
    #  CICLO PRINCIPAL DE OPTIMIZACIÓN v2 (HUNT & CONVERGE)
    # --------------------------------------------------------------------------

    def optimize(self) -> BoogeymanV2OptimizationResult:
        """Ejecuta el ciclo de optimización de AD-BSA v2."""
        start_time = time.perf_counter()
        self._initialize_children()

        history_best = [self.SafeHouse_fitness]
        history_evals = [self.evaluations_count]
        history_sack = [0]
        history_pop = [self.current_pop_size]
        history_strength = [1.0]

        while self.evaluations_count < self.max_evaluations:
            self.generation += 1

            # 1. Despertar al Séquito del Cuco (Multi-Boogeyman) y Radio de Captura
            boogeymen_pos, capture_radius = self._awaken_multi_boogeymen()

            # 2. Generar Hiperparámetros con Enfriamiento Termodinámico
            F_safe, F_escape, F_diff, CR, annealing_factor = self._sample_parameters()

            # 3. Mutación de Escape Multi-Anti-Atractor con Archivo Histórico
            x_best_p, S_closest, x_r1, x_r2 = self._select_mutation_partners_v2(boogeymen_pos)

            safe_vector = (x_best_p - self.Children) * F_safe[:, np.newaxis]
            escape_vector = self._compute_escape_vector(S_closest, F_escape, capture_radius)
            diff_vector = (x_r1 - x_r2) * F_diff[:, np.newaxis]

            mutant_children = self._bound_reflection(
                self.Children + safe_vector + escape_vector + diff_vector,
                x_base=self.Children
            )

            # 4. Cruce Binomial Vectorizado
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

            # 5. Selección y Operador El Saco (The Sack)
            improvements_mask = trial_fitness <= self.fitness[:evals_to_run]
            successful_indices = np.where(improvements_mask)[0]

            fitness_deltas = np.zeros(evals_to_run, dtype=np.float64)
            fitness_deltas[successful_indices] = self.fitness[successful_indices] - trial_fitness[successful_indices]

            # Archivar a los padres que fueron superados
            self._update_archive(self.Children[successful_indices])

            self.Children[successful_indices] = trial_children[successful_indices]
            self.fitness[successful_indices] = trial_fitness[successful_indices]

            best_idx_now = np.argmin(self.fitness)
            if self.fitness[best_idx_now] < self.SafeHouse_fitness:
                self.SafeHouse_fitness = float(self.fitness[best_idx_now])
                self.SafeHouse = self.Children[best_idx_now].copy()

            # Fracasos: Evaluar captura en El Saco
            failed_indices = np.where(~improvements_mask)[0]
            sack_captures = 0

            if len(failed_indices) > 0:
                dist_to_s = np.linalg.norm(trial_children[failed_indices] - S_closest[failed_indices], axis=1)
                captured_mask = dist_to_s < capture_radius
                captured_indices = failed_indices[captured_mask]
                sack_captures = len(captured_indices)

                if sack_captures > 0:
                    # Reeducación guiada hacia el SafeHouse formal del paper (Sección 2.5):
                    # u_reborn = x_pbest + N(0, 1) * (R_c / 2) * (x* - S) / ||x* - S||
                    direction = self.SafeHouse - S_closest[captured_indices]
                    norm = np.linalg.norm(direction, axis=1, keepdims=True)
                    unit_safe_dir = direction / np.maximum(norm, 1e-12)
                    noise = self.rng.normal(0.0, 1.0, size=(sack_captures, 1))

                    reborn_candidate = (
                        x_best_p[captured_indices] +
                        noise * (capture_radius / 2.0) * unit_safe_dir
                    )
                    reborn_children = self._bound_reflection(
                        reborn_candidate,
                        x_base=x_best_p[captured_indices]
                    )

                    if self.evaluations_count < self.max_evaluations:
                        reborn_to_eval = min(sack_captures, self.max_evaluations - self.evaluations_count)
                        reborn_fitness = self.objective_func(reborn_children[:reborn_to_eval])
                        self.evaluations_count += reborn_to_eval

                        reborn_target_idx = captured_indices[:reborn_to_eval]
                        better_reborn = reborn_fitness < self.fitness[reborn_target_idx]
                        if np.any(better_reborn):
                            acc_idx = reborn_target_idx[better_reborn]
                            self.Children[acc_idx] = reborn_children[:reborn_to_eval][better_reborn]
                            self.fitness[acc_idx] = reborn_fitness[better_reborn]

                            best_reb = np.argmin(self.fitness[acc_idx])
                            if self.fitness[acc_idx[best_reb]] < self.SafeHouse_fitness:
                                self.SafeHouse_fitness = float(self.fitness[acc_idx[best_reb]])
                                self.SafeHouse = self.Children[acc_idx[best_reb]].copy()

            # 6. Auto-Adaptación de Memorias de Lehmer
            if len(successful_indices) > 0 and np.sum(fitness_deltas[successful_indices]) > 0:
                weights = fitness_deltas[successful_indices] / np.sum(fitness_deltas[successful_indices])
                self.Memory_F_safe[self.memory_index] = self._lehmer_mean(F_safe[successful_indices], weights)
                self.Memory_F_escape[self.memory_index] = self._lehmer_mean(F_escape[successful_indices] / max(1e-4, annealing_factor), weights)
                self.Memory_F_diff[self.memory_index] = self._lehmer_mean(F_diff[successful_indices], weights)
                self.Memory_CR[self.memory_index] = float(np.sum(weights * CR[successful_indices]))
                self.memory_index = (self.memory_index + 1) % self.memory_size

            # 7. Reducción Lineal de Población (LPSR v2)
            self._boogeyman_lpsr_v2()

            history_best.append(self.SafeHouse_fitness)
            history_evals.append(self.evaluations_count)
            history_sack.append(sack_captures)
            history_pop.append(self.current_pop_size)
            history_strength.append(annealing_factor)

        exec_time = time.perf_counter() - start_time
        return BoogeymanV2OptimizationResult(
            safe_house_position=self.SafeHouse,
            safe_house_fitness=self.SafeHouse_fitness,
            total_evaluations=self.evaluations_count,
            generations=self.generation,
            history_best_fitness=history_best,
            history_evaluations=history_evals,
            history_sack_captures=history_sack,
            history_population_size=history_pop,
            history_boogeyman_strength=history_strength,
            execution_time_seconds=exec_time
        )



# ==============================================================================
#  AD-BSA Csc: Bounded Cosecant Repulsion Potential Profile
# ==============================================================================

class AD_BSA_Csc(AD_BSA_v2):
    """
    AD-BSA con Repulsión Cosecante Acotada (AD-BSA-Csc / AD-BSA v3).
    
    Implementa un perfil de potencial analítico no lineal tipo Pöschl-Teller:
      Phi(r) = clip( 1 / sin(clip(pi * r / (4 * R_c), 1e-3, 0.999 * pi / 2)), 1.0, 4.0 ) - 1.0
      
    Propiedades:
      - En r -> 0: Barrera infinita (acotada a M_max = 4.0) que expulsa inmediatamente de trampas.
      - En r -> 2*R_c: Decaimiento suave a 0 con derivada nula (dPhi/dr = 0), eliminando
        el ruido ortogonal en valles parabólicos y estrechos (ej. Bent Cigar F1).
    """
    def _compute_escape_vector(self, S_closest: np.ndarray, F_escape: np.ndarray, capture_radius: float) -> np.ndarray:
        diff = self.Children - S_closest
        r = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8
        r_norm = r / (capture_radius * 2.0 + 1e-8)
        
        # Cosecante acotada sin picos infinitos (techo M_max = 4.0)
        force_profile = np.clip(1.0 / (np.sin(np.clip(r_norm * np.pi * 0.5, 1e-3, np.pi * 0.999))), 1.0, 4.0) - 1.0
        force_profile = np.where(r_norm < 2.0, force_profile, 0.0)
        
        unit_escape = diff / r
        return unit_escape * (force_profile * F_escape[:, np.newaxis] * capture_radius)


# Alias de compatibilidad
AD_BSA = AD_BSA_Csc

# ==============================================================================
#  DEMO Y COMPARATIVA: AD-BSA v1 vs AD-BSA v2 vs COMPETIDORES
# ==============================================================================

def main():
    print("=" * 85)
    print("  COMPARATIVA DIRECTA: AD-BSA v1  vs  AD-BSA v2 (THE NIGHTMARE EDITION)")
    print("  Benchmarks en 30 Dimensiones | MaxNFE = 50,000 | 10 Corridas por Problema")
    print("=" * 85)

    from benchmark_suite import BENCHMARK_SUITE, StandardDE, StandardPSO, ackley, rastrigin, rosenbrock, sphere

    dim = 30
    max_evals = 50000
    runs = 10

    test_cases = [
        ("Sphere (Unimodal)", sphere, (-100.0, 100.0)),
        ("Rosenbrock (Valle Banana)", rosenbrock, (-30.0, 30.0)),
        ("Rastrigin (Multimodal Severo)", rastrigin, (-5.12, 5.12)),
        ("Ackley (Embudo Profundo)", ackley, (-32.768, 32.768))
    ]

    for name, func, b_range in test_cases:
        bounds = np.array([list(b_range)] * dim)
        fits_v1 = []
        fits_v2 = []
        fits_pso = []
        fits_de = []

        from ad_bsa import AD_BSA

        for r in range(runs):
            seed = 42 + r * 101

            # AD-BSA v1
            v1 = AD_BSA(func, bounds, pop_size_max=18 * dim, pop_size_min=4, max_evaluations=max_evals, seed=seed)
            res_v1 = v1.optimize()
            fits_v1.append(res_v1.safe_house_fitness)

            # AD-BSA v2
            v2 = AD_BSA_v2(func, bounds, max_evaluations=max_evals, num_boogeymen=2, annealing_power=1.5, seed=seed)
            res_v2 = v2.optimize()
            fits_v2.append(res_v2.safe_house_fitness)

            # PSO
            pso = StandardPSO(func, bounds, pop_size=50, max_evaluations=max_evals, seed=seed)
            res_pso = pso.optimize()
            fits_pso.append(res_pso.best_fitness)

            # DE
            de = StandardDE(func, bounds, pop_size=50, max_evaluations=max_evals, seed=seed)
            res_de = de.optimize()
            fits_de.append(res_de.best_fitness)

        print(f"\n>>> RESULTADOS EN {name}:")
        print(f"    AD-BSA v1 : Media = {np.mean(fits_v1):.3e} | Mediana = {np.median(fits_v1):.3e} | Min = {np.min(fits_v1):.3e}")
        print(f"    AD-BSA v2 : Media = {np.mean(fits_v2):.3e} | Mediana = {np.median(fits_v2):.3e} | Min = {np.min(fits_v2):.3e}  <--- [NUEVA VERSION]")
        print(f"    Can. PSO  : Media = {np.mean(fits_pso):.3e} | Mediana = {np.median(fits_pso):.3e} | Min = {np.min(fits_pso):.3e}")
        print(f"    Std. DE   : Media = {np.mean(fits_de):.3e} | Mediana = {np.median(fits_de):.3e} | Min = {np.min(fits_de):.3e}")

    print("\n[OK] Comparativa completada.")


if __name__ == "__main__":
    main()
