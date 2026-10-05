import sys
import time
import numpy as np
from scipy import stats
import cma
import opfunu.cec_based.cec2020 as cec20

sys.path.append(r"C:\Users\Petiso\Documents\ideas millonarias\papers")
sys.path.append(r"c:\Users\Petiso\Documents\antigravity\eager-faraday")

from ad_bsa_v2 import AD_BSA_v2

# ==============================================================================
#  1. AD-BSA Csc (Repulsión Cosecante Acotada / Pöschl-Teller)
# ==============================================================================
class AD_BSA_Csc(AD_BSA_v2):
    def _compute_escape_vector(self, S_closest: np.ndarray, F_escape: np.ndarray, capture_radius: float) -> np.ndarray:
        diff = self.Children - S_closest
        r = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8
        r_norm = r / (capture_radius * 2.0 + 1e-8)
        
        # Cosecante acotada sin picos infinitos (techo M_max = 4.0)
        force_profile = np.clip(1.0 / (np.sin(np.clip(r_norm * np.pi * 0.5, 1e-3, np.pi * 0.999))), 1.0, 4.0) - 1.0
        force_profile = np.where(r_norm < 2.0, force_profile, 0.0)
        
        unit_escape = diff / r
        return unit_escape * (force_profile * F_escape[:, np.newaxis] * capture_radius)


# ==============================================================================
#  2. jSO (IEEE CEC 2017 Winner - Brest et al.)
# ==============================================================================
class jSO:
    def __init__(self, objective_func, bounds: np.ndarray, max_evaluations: int = 50000, memory_size: int = 5, seed: int = None):
        self.f = objective_func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.max_nfe = max_evaluations
        self.H = memory_size
        self.rng = np.random.default_rng(seed)
        
        self.N_init = min(18 * self.dim, max(50, int(self.max_nfe / 200)))
        self.N_min = 4
        self.N = self.N_init

    def optimize(self):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        pop = self.rng.uniform(low, high, size=(self.N, self.dim))
        fits = self.f(pop)
        nfe = self.N
        
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = float(fits[best_idx])
        
        M_F = np.full(self.H, 0.5, dtype=np.float64)
        M_CR = np.full(self.H, 0.8, dtype=np.float64)
        k_mem = 0
        
        archive = []
        max_arc = int(1.4 * self.N_init)
        
        while nfe < self.max_nfe:
            nfe_ratio = nfe / self.max_nfe
            p_val = 0.25 - 0.125 * nfe_ratio
            p_num = max(2, int(round(p_val * self.N)))
            
            sort_idx = np.argsort(fits)
            pop = pop[sort_idx]
            fits = fits[sort_idx]
            
            S_F, S_CR, delta_f = [], [], []
            trial_pop = np.empty_like(pop)
            
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
                v = np.where(v < low, (pop[i] + low) / 2.0, v)
                v = np.where(v > high, (pop[i] + high) / 2.0, v)
                
                j_rand = self.rng.integers(0, self.dim)
                u = np.where((self.rng.random(self.dim) < cr) | (np.arange(self.dim) == j_rand), v, pop[i])
                trial_pop[i] = u
                
            batch_size = min(self.N, self.max_nfe - nfe)
            if batch_size < self.N:
                trial_pop = trial_pop[:batch_size]
            
            trial_fits = self.f(trial_pop)
            nfe += batch_size
            
            for i in range(batch_size):
                if trial_fits[i] < fits[i]:
                    archive.append(pop[i].copy())
                    delta_f.append(fits[i] - trial_fits[i])
                    S_F.append(f_val)
                    S_CR.append(cr)
                    pop[i] = trial_pop[i]
                    fits[i] = trial_fits[i]
                    if trial_fits[i] < best_f:
                        best_f = float(trial_fits[i])
                        best_x = trial_pop[i].copy()
                        
            while len(archive) > max_arc:
                del archive[self.rng.integers(0, len(archive))]
                
            if len(S_F) > 0:
                w = np.array(delta_f) / np.sum(delta_f)
                M_F[k_mem] = np.sum(w * (np.array(S_F)**2)) / np.sum(w * np.array(S_F))
                if np.max(S_CR) > 0:
                    M_CR[k_mem] = np.sum(w * (np.array(S_CR)**2)) / np.sum(w * np.array(S_CR))
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
                max_arc = int(1.4 * self.N)
                while len(archive) > max_arc:
                    del archive[self.rng.integers(0, len(archive))]

        return best_x, best_f


# ==============================================================================
#  3. CMA-ES (Hansen Official Framework)
# ==============================================================================
class CMA_ES_Optimizer:
    def __init__(self, objective_func, bounds: np.ndarray, max_evaluations: int = 50000, seed: int = None):
        self.f = objective_func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.max_nfe = max_evaluations
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def optimize(self):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        x0 = self.rng.uniform(low * 0.5, high * 0.5, size=self.dim)
        sigma0 = (high[0] - low[0]) * 0.25
        
        opts = {
            'bounds': [low.tolist(), high.tolist()],
            'maxfevals': self.max_nfe,
            'seed': self.seed if self.seed is not None else 42,
            'verbose': -9,
        }
        
        def _single_eval(x):
            return float(self.f(np.atleast_2d(x))[0])
            
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        es.optimize(_single_eval)
        
        return es.result.xbest, float(es.result.fbest)


CEC_FUNCS = {
    "F1": (cec20.F12020, "Unimodal (Bent Cigar)"),
    "F2": (cec20.F22020, "Multimodal (Schwefel)"),
    "F3": (cec20.F32020, "Multimodal (Lunacek bi-Rastrigin)"),
    "F4": (cec20.F42020, "Multimodal (Rosenbrock + Griewank)"),
    "F5": (cec20.F52020, "Hybrid 1"),
    "F6": (cec20.F62020, "Hybrid 2"),
    "F7": (cec20.F72020, "Hybrid 3"),
    "F8": (cec20.F82020, "Composition 1"),
    "F9": (cec20.F92020, "Composition 2"),
    "F10": (cec20.F102020, "Composition 3"),
}

def run_single_elite_task(task):
    func_name, algo_name, run_idx, seed, dim, max_nfe = task
    
    func_cls, _ = CEC_FUNCS[func_name]
    cec_obj = func_cls(ndim=dim)
    
    def _eval(x):
        x = np.atleast_2d(x)
        return np.array([cec_obj.evaluate(ind) for ind in x], dtype=np.float64)
    
    bounds = np.array([[-100.0, 100.0]] * dim)
    t0 = time.perf_counter()
    
    if algo_name == "AD-BSA-Csc":
        opt = AD_BSA_Csc(_eval, bounds=bounds, max_evaluations=max_nfe, seed=seed)
        res = opt.optimize()
        best_f = res.safe_house_fitness
    elif algo_name == "jSO":
        opt = jSO(_eval, bounds=bounds, max_evaluations=max_nfe, seed=seed)
        _, best_f = opt.optimize()
    elif algo_name == "CMA-ES":
        opt = CMA_ES_Optimizer(_eval, bounds=bounds, max_evaluations=max_nfe, seed=seed)
        _, best_f = opt.optimize()
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
        
    elapsed = time.perf_counter() - t0
    
    return {
        "func": func_name,
        "algo": algo_name,
        "run": run_idx,
        "seed": seed,
        "fitness": float(best_f),
        "time": float(elapsed)
    }
