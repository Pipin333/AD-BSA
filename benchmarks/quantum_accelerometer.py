"""
================================================================================
  BENCHMARK MULTIDIMENSIONAL CUÁNTICO: D = 10, 15, 20
  Control Cuántico Óptimo de Floquet: AD-BSA v2 vs. Standard DE
  Modelo: Acelerómetro Atomtrónico de Carmona-López et al. (Phys. Rev. Res. 2026)
================================================================================
"""

import sys
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# Importar AD-BSA v2 desde el workspace
from ad_bsa_v2 import AD_BSA_v2


# ==============================================================================
# 1. SIMULADOR CUÁNTICO EXACTO DE MUCHOS CUERPOS
# ==============================================================================

class MultiHarmonicBoseHubbardSimulator:
    def __init__(self, Ns=3, N=3, omega_B=1.0, J=1.0, tau=18.0):
        self.Ns = Ns
        self.N = N
        self.omega_B = omega_B
        self.J = J
        self.tau = tau
        
        # Base de Fock para N=3 en Ns=3 (10 estados)
        self.states = []
        for n1 in range(N + 1):
            for n2 in range(N + 1 - n1):
                n3 = N - n1 - n2
                self.states.append((n1, n2, n3))
        self.dim = len(self.states)
        
        # Operador de salto forward: sum_l a_{l+1}^dagger a_l
        self.hop_fwd = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for i, s in enumerate(self.states):
            for l in range(Ns):
                lp1 = (l + 1) % Ns
                if s[l] > 0:
                    s_new = list(s)
                    s_new[l] -= 1
                    s_new[lp1] += 1
                    for j, s2 in enumerate(self.states):
                        if s2 == tuple(s_new):
                            self.hop_fwd[j, i] += np.sqrt(s[l] * (s[lp1] + 1))
        self.hop_bwd = self.hop_fwd.conj().T
        
        # Operador de interacción
        diag_inter = np.zeros(self.dim, dtype=np.float64)
        for i, s in enumerate(self.states):
            diag_inter[i] = sum(s[l] * (s[l] - 1) for l in range(Ns))
        self.H_int = 0.5 * np.diag(diag_inter)

    def evaluate_pulse(self, x: np.ndarray, tau=None):
        """
        Decodifica un vector de dimensión D en un pulso multifrecuencia:
          x[0] = omega_fund in [0.5, 1.5]
          x[1] = U/J in [0.02, 0.20]
          Resto de parámetros: pares (A_k, theta_k) para los armónicos k=1..K
          Si sobra 1 parámetro (caso D impar): chirp rate beta
        """
        if tau is None:
            tau = self.tau
            
        omega = x[0]
        U = x[1]
        
        # Extraer armónicos
        rest = x[2:]
        D_rest = len(rest)
        K = D_rest // 2
        A_k = rest[:K]
        theta_k = rest[K:2*K]
        chirp = rest[2*K] if (D_rest % 2 == 1) else 0.0
        
        # Estado fundamental a t=0
        H0 = -self.J * (self.hop_fwd + self.hop_bwd) + U * self.H_int
        _, v = eigh(H0)
        psi0 = v[:, 0]
        y0 = np.concatenate([np.real(psi0), np.imag(psi0)])
        
        dim = self.dim
        J = self.J
        hop_fwd = self.hop_fwd
        hop_bwd = self.hop_bwd
        H_int = self.H_int
        omega_B = self.omega_B
        Ns = self.Ns
        
        harm_indices = np.arange(1, K + 1)
        
        def rhs(t, y):
            psi = y[:dim] + 1j * y[dim:]
            # Síntesis de Fourier de la fase de Peierls:
            # phi(t) = omega_B * t + sum_k A_k * sin(k * omega * t + theta_k) + chirp * t^2
            harmonics = np.sum(A_k * np.sin(harm_indices * omega * t + theta_k))
            phi = omega_B * t + harmonics + chirp * (t ** 1.5)
            phase = np.exp(1j * phi / Ns)
            H = -J * (phase * hop_fwd + np.conj(phase) * hop_bwd) + U * H_int
            dpsi = -1j * (H @ psi)
            return np.concatenate([np.real(dpsi), np.imag(dpsi)])
            
        t_eval = np.linspace(0, tau, 30)
        sol = solve_ivp(rhs, (0, tau), y0, t_eval=t_eval, method='RK23', rtol=1e-3, atol=1e-3)
        
        currents = []
        for idx, t in enumerate(sol.t):
            psi_t = sol.y[:dim, idx] + 1j * sol.y[dim:, idx]
            harmonics = np.sum(A_k * np.sin(harm_indices * omega * t + theta_k))
            phi_t = omega_B * t + harmonics + chirp * (t ** 1.5)
            phase_t = np.exp(1j * phi_t / Ns)
            J_op = -1j * (J / Ns) * (phase_t * hop_fwd - np.conj(phase_t) * hop_bwd)
            I_t = np.real(psi_t.conj().T @ J_op @ psi_t)
            currents.append(I_t)
            
        return float(np.trapezoid(currents, sol.t) / tau)


# Instancia del simulador
SIMULATOR = MultiHarmonicBoseHubbardSimulator(Ns=3, N=3, omega_B=1.0, tau=18.0)

def create_quantum_objective(D: int):
    """Crea límites físicos y función objetivo para dimensión D."""
    bounds = []
    # 0: omega
    bounds.append([0.6, 1.4])
    # 1: U/J
    bounds.append([0.02, 0.20])
    
    K = (D - 2) // 2
    for k in range(1, K + 1):
        # Amplitud A_k: decrece naturalmente con los armónicos superiores
        max_A = max(0.4, 1.2 / np.sqrt(k))
        bounds.append([0.0, max_A])
    for k in range(1, K + 1):
        # Fase theta_k
        bounds.append([0.0, 2.0 * np.pi])
    if (D - 2) % 2 == 1:
        # Chirp rate
        bounds.append([-0.05, 0.05])
        
    bounds = np.array(bounds, dtype=np.float64)
    
    def fitness_single(x: np.ndarray) -> float:
        # Evaluar corriente principal
        I_main = abs(SIMULATOR.evaluate_pulse(x))
        if I_main < 1e-4:
            return 20.0  # Fuerte penalización a trampas de corriente nula
            
        # Evaluar sensibilidad (gradiente ante desintonía delta_omega = 0.05)
        x_detuned = x.copy()
        x_detuned[0] += 0.05
        I_detuned = abs(SIMULATOR.evaluate_pulse(x_detuned))
        
        # Caída abrupta deseada
        drop = I_main - I_detuned
        sharpness = max(0.0, drop / 0.05)
        
        # Ponderación de calidad: señal neta + nitidez de la resonancia
        score = I_main * (1.0 + 4.0 * sharpness)
        return -float(score)
        
    def fitness_vectorized(X: np.ndarray) -> np.ndarray:
        res = np.zeros(len(X), dtype=np.float64)
        for i in range(len(X)):
            res[i] = fitness_single(X[i])
        return res
        
    return bounds, fitness_single, fitness_vectorized


# ==============================================================================
# 2. STANDARD DE CON MISMO MANEJO DE LÍMITES (FAIR COMPARISON)
# ==============================================================================

class FairStandardDE:
    def __init__(self, objective_func, bounds, max_evals, pop_size=24, F=0.5, CR=0.7, seed=42):
        self.func = objective_func
        self.bounds = np.asarray(bounds)
        self.max_evals = max_evals
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.rng = np.random.default_rng(seed)
        self.dim = len(bounds)
        
    def optimize(self):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        pop = low + self.rng.random((self.pop_size, self.dim)) * (high - low)
        fitness = np.array([self.func(ind) for ind in pop])
        evals = self.pop_size
        
        best_idx = np.argmin(fitness)
        best_fit = fitness[best_idx]
        history_evals = [evals]
        history_best = [best_fit]
        
        while evals < self.max_evals:
            for i in range(self.pop_size):
                if evals >= self.max_evals:
                    break
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = self.rng.choice(idxs, 3, replace=False)
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
                
                # Midpoint reflection para no tener sesgo de esquina artificial
                low_viol = mutant < low
                high_viol = mutant > high
                mutant = np.where(low_viol, (pop[i] + low) / 2.0, mutant)
                mutant = np.where(high_viol, (pop[i] + high) / 2.0, mutant)
                mutant = np.clip(mutant, low, high)
                
                cross_points = self.rng.random(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[self.rng.integers(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                f_trial = self.func(trial)
                evals += 1
                
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial
                    if f_trial < best_fit:
                        best_fit = f_trial
                        
            history_evals.append(evals)
            history_best.append(best_fit)
            
        best_idx = np.argmin(fitness)
        return pop[best_idx], best_fit, history_evals, history_best


# ==============================================================================
# 3. BENCHMARK EN D = 10, 15, 20
# ==============================================================================

def main():
    print("=" * 80)
    print("  EJECUTANDO BENCHMARK MULTIDIMENSIONAL DE CONTROL CUÁNTICO")
    print("  AD-BSA v2 (Boogeyman Anti-Attractors) vs. Standard DE (DE/rand/1/bin)")
    print("  Dimensiones: D = 10, D = 15, D = 20")
    print("=" * 80)
    
    dims = [10, 15, 20]
    budget = 350  # 350 evaluaciones por dimensión (rápido y suficiente)
    
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=120)
    
    for idx, D in enumerate(dims):
        print(f"\n>>> INICIANDO TORNEO EN D = {D} ({ (D-2)//2 } Armónicos de Shaking)...")
        bounds, f_single, f_vec = create_quantum_objective(D)
        
        pop_size = max(18, min(40, 2 * D))
        
        # 1. Standard DE
        t0 = time.time()
        de = FairStandardDE(f_single, bounds, max_evals=budget, pop_size=pop_size, seed=100 + D)
        best_x_de, best_fit_de, evals_de, hist_de = de.optimize()
        t_de = time.time() - t0
        print(f"  [Standard DE] Finalizado en {t_de:.1f}s | Mejor Score: {-best_fit_de:.4f}")
        
        # 2. AD-BSA v2
        t0 = time.time()
        bsa = AD_BSA_v2(
            objective_func=f_vec,
            bounds=bounds,
            max_evaluations=budget,
            pop_size_max=pop_size,
            pop_size_min=4,
            num_boogeymen=3,          # 3 Cucos acechando en alta dimensión
            k_boogeyman_ratio=0.18,
            annealing_power=1.6,
            seed=100 + D
        )
        res_bsa = bsa.optimize()
        t_bsa = time.time() - t0
        best_fit_bsa = res_bsa.safe_house_fitness
        print(f"  [AD-BSA v2]   Finalizado en {t_bsa:.1f}s | Mejor Score: {-best_fit_bsa:.4f}")
        
        gain = (-best_fit_bsa) / max(1e-5, -best_fit_de)
        print(f"  --> Ventaja AD-BSA v2 en D={D}: {gain:.2f}x ({'+' if gain > 1 else ''}{(gain-1)*100:.1f}%)")
        
        results[D] = {
            'de_score': -best_fit_de,
            'bsa_score': -best_fit_bsa,
            'gain': gain,
            't_de': t_de,
            't_bsa': t_bsa
        }
        
        # Graficar en el panel correspondiente
        ax = axes[idx]
        ax.plot(evals_de, -np.array(hist_de), 'r--', label='Standard DE', lw=2.0)
        ax.plot(res_bsa.history_evaluations, -np.array(res_ad_fitness := res_bsa.history_best_fitness), 'g-', label='AD-BSA v2 (Boogeyman)', lw=2.5)
        ax.set_title(f'Dimensión D = {D} ({ (D-2)//2 } Armónicos)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Evaluaciones Cuánticas (NFE)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Sensibilidad Cuántica (Score)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
    plt.tight_layout()
    plot_file = "carmona_ad_bsa_multidim_benchmark.png"
    plt.savefig(plot_file)
    print(f"\n[OK] Gráfico comparativo multidimensional guardado en: {plot_file}")
    
    print("\n" + "=" * 80)
    print("  TABLA FINAL DE RESULTADOS: BENCHMARK D = 10, 15, 20")
    print("=" * 80)
    print(f"{'Dimensión':<12} | {'Standard DE Score':<18} | {'AD-BSA v2 Score':<18} | {'Ventaja Cuco':<15} | {'Vencedor'}")
    print("-" * 80)
    for D in dims:
        r = results[D]
        winner = "AD-BSA v2 (GANA)" if r['gain'] > 1.05 else ("EMPATE TÉCNICO" if r['gain'] >= 0.95 else "DE (GANA)")
        print(f"D = {D:<8} | {r['de_score']:<18.4f} | {r['bsa_score']:<18.4f} | {r['gain']:<6.2f}x ({'+' if r['gain']>1 else ''}{(r['gain']-1)*100:.1f}%) | {winner}")
    print("=" * 80)

if __name__ == '__main__':
    main()
