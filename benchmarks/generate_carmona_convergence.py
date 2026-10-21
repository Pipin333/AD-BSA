"""
================================================================================
  AD-BSA Quantum Benchmark: Multi-Harmonic Floquet Convergence Curves
  Simulates Many-Body Bose-Hubbard Hamiltonian (Carmona-Lopez et al. PRResearch 2026)
  across dimensions D = 10, D = 15, D = 20 with Mean ± Std Shaded Error Bands:
    1. AD-BSA (Proposed - Bounded Cosecant Repulsion |csc|)
    2. jSO (Canonical CEC 2017 Winner - Brest et al.)
    3. L-SHADE (Canonical CEC 2014 Winner - Tanabe & Fukunaga)
    4. Standard-DE (Canonical DE/rand/1/bin)
================================================================================
"""

import os
import sys
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from ad_bsa import (
    AD_BSA,
    jSO,
    L_SHADE,
    StandardDE,
    OptimizationResult,
)


class MultiHarmonicBoseHubbardSimulator:
    """Exact Many-Body Quantum Floquet Simulator for 3-Site Ring."""

    def __init__(self, Ns=3, N=3, omega_B=1.0, J=1.0, tau=18.0):
        self.Ns = Ns
        self.N = N
        self.omega_B = omega_B
        self.J = J
        self.tau = tau

        # Fock basis for N=3 bosons in Ns=3 sites (10 states)
        self.states = []
        for n1 in range(N + 1):
            for n2 in range(N + 1 - n1):
                n3 = N - n1 - n2
                self.states.append((n1, n2, n3))
        self.dim = len(self.states)

        # Forward hopping operator
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

        # On-site interaction operator
        diag_inter = np.zeros(self.dim, dtype=np.float64)
        for i, s in enumerate(self.states):
            diag_inter[i] = sum(s[l] * (s[l] - 1) for l in range(Ns))
        self.H_int = 0.5 * np.diag(diag_inter)

    def evaluate_pulse(self, x: np.ndarray, tau=None):
        if tau is None:
            tau = self.tau

        omega = x[0]
        U = x[1]
        rest = x[2:]
        D_rest = len(rest)
        K = D_rest // 2
        A_k = rest[:K]
        theta_k = rest[K:2*K]
        chirp = rest[2*K] if (D_rest % 2 == 1) else 0.0

        # Ground state at t=0
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


SIMULATOR = MultiHarmonicBoseHubbardSimulator(Ns=3, N=3, omega_B=1.0, tau=18.0)


def create_quantum_objective(D: int):
    """Generates parameter bounds and cost function (negative sensitivity score)."""
    bounds = []
    bounds.append([0.6, 1.4])   # omega
    bounds.append([0.02, 0.20]) # U/J

    K = (D - 2) // 2
    for k in range(1, K + 1):
        max_A = max(0.4, 1.2 / np.sqrt(k))
        bounds.append([0.0, max_A])
    for k in range(1, K + 1):
        bounds.append([0.0, 2.0 * np.pi])
    if (D - 2) % 2 == 1:
        bounds.append([-0.05, 0.05])  # chirp rate

    bounds = np.array(bounds, dtype=np.float64)

    def fitness_single(x: np.ndarray) -> float:
        I_main = abs(SIMULATOR.evaluate_pulse(x))
        if I_main < 1e-4:
            return 20.0  # Heavy penalty for zero-current traps

        x_detuned = x.copy()
        x_detuned[0] += 0.05
        I_detuned = abs(SIMULATOR.evaluate_pulse(x_detuned))

        drop = I_main - I_detuned
        sharpness = max(0.0, drop / 0.05)
        score = I_main * (1.0 + 4.0 * sharpness)
        return -float(score)

    def fitness_vectorized(X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return fitness_single(X)
        res = np.zeros(len(X), dtype=np.float64)
        for i in range(len(X)):
            res[i] = fitness_single(X[i])
        return res

    return bounds, fitness_vectorized


ALGORITHMS = {
    "AD-BSA (Proposed)": {
        "cls": AD_BSA,
        "kwargs": {
            "num_boogeymen": 3,
            "k_boogeyman_ratio": 0.18,
            "annealing_power": 1.6,
            "pop_size_max": 24,
            "pop_size_min": 4,
        },
        "color": "#1D4ED8",  # Vibrant Deep Blue
        "linestyle": "-",
        "linewidth": 2.5,
        "zorder": 10,
    },
    "jSO": {
        "cls": jSO,
        "kwargs": {
            "pop_size_init": 24,
        },
        "color": "#059669",  # Emerald Green
        "linestyle": "-.",
        "linewidth": 1.8,
        "zorder": 8,
    },
    "L-SHADE": {
        "cls": L_SHADE,
        "kwargs": {
            "pop_size_init": 24,
        },
        "color": "#D97706",  # Amber
        "linestyle": "--",
        "linewidth": 1.8,
        "zorder": 6,
    },
    "Standard-DE": {
        "cls": StandardDE,
        "kwargs": {
            "pop_size": 24,
        },
        "color": "#DC2626",  # Red
        "linestyle": ":",
        "linewidth": 1.8,
        "zorder": 4,
    },
}


def run_carmona_worker(args):
    D, aname, run_idx, budget, seed = args
    bounds, f_vec = create_quantum_objective(D)
    ainfo = ALGORITHMS[aname]

    opt = ainfo["cls"](
        f_vec,
        bounds=bounds,
        max_evaluations=budget,
        seed=seed,
        **ainfo["kwargs"],
    )
    res = opt.optimize()

    raw_evals = np.array(res.history_evaluations, dtype=np.float64)
    # Fitness is negative score -> score = -fitness
    raw_scores = -np.array(res.history_best_fitness, dtype=np.float64)

    # Interpolate to 100 uniform evaluation points
    target_evals = np.linspace(min(raw_evals[0], 24), budget, 100)
    interp_scores = np.zeros_like(target_evals)

    for idx, e in enumerate(target_evals):
        mask = raw_evals <= e
        if np.any(mask):
            interp_scores[idx] = np.max(raw_scores[mask])
        else:
            interp_scores[idx] = raw_scores[0]

    return {
        "D": D,
        "aname": aname,
        "run_idx": run_idx,
        "evals": target_evals.tolist(),
        "scores": interp_scores.tolist(),
        "best_score": float(interp_scores[-1]),
    }


def main():
    dims = [10, 15, 20]
    budget = 350
    n_runs = 5
    workers = min(8, os.cpu_count() or 4)

    tasks = []
    for D in dims:
        for aname in ALGORITHMS.keys():
            for r in range(n_runs):
                seed = 1000 * (r + 1) + D * 17
                tasks.append((D, aname, r, budget, seed))

    print(f"=== Starting Carmona Quantum Floquet Convergence Runner ({len(tasks)} tasks across {workers} workers) ===")
    t_start = time.time()

    runs_data = {f"D{D}": {a: [] for a in ALGORITHMS.keys()} for D in dims}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_carmona_worker, t): t for t in tasks}
        completed = 0
        for fut in as_completed(futures):
            res = fut.result()
            D = res["D"]
            aname = res["aname"]
            runs_data[f"D{D}"][aname].append(res)
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(f"[{completed:02d}/{len(tasks):02d}] D={D} | {aname:<18} | Peak: {res['best_score']:.4f}")

    total_time = time.time() - t_start
    print(f"\nAll quantum simulations completed in {total_time:.1f}s.")

    # Calculate mean and std per algorithm per dimension
    consolidated = {}
    for D in dims:
        d_key = f"D{D}"
        consolidated[d_key] = {}
        for aname in ALGORITHMS.keys():
            all_scores = np.array([r["scores"] for r in runs_data[d_key][aname]])
            evals = runs_data[d_key][aname][0]["evals"]
            mean_s = np.mean(all_scores, axis=0)
            std_s = np.std(all_scores, axis=0)
            peak_s = np.max([r["best_score"] for r in runs_data[d_key][aname]])
            final_mean = float(mean_s[-1])
            final_std = float(std_s[-1])

            consolidated[d_key][aname] = {
                "evals": evals,
                "mean_scores": mean_s.tolist(),
                "std_scores": std_s.tolist(),
                "final_mean": final_mean,
                "final_std": final_std,
                "peak": float(peak_s),
            }

    # Save JSON
    json_path = os.path.join(os.path.dirname(__file__), "carmona_quantum_convergence_data.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(consolidated, jf, indent=2)
    print(f"Saved quantum convergence data to: {json_path}")

    # Plotting 3 panels with shaded error bands
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), dpi=300)

    for idx, D in enumerate(dims):
        ax = axes[idx]
        d_key = f"D{D}"

        for aname, astyle in ALGORITHMS.items():
            entry = consolidated[d_key][aname]
            evs = np.array(entry["evals"])
            m = np.array(entry["mean_scores"])
            s = np.array(entry["std_scores"])

            line, = ax.plot(
                evs,
                m,
                color=astyle["color"],
                linestyle=astyle["linestyle"],
                linewidth=astyle["linewidth"],
                label=f"{aname} ({entry['final_mean']:.2f} ± {entry['final_std']:.2f})",
                zorder=astyle["zorder"],
            )
            ax.fill_between(
                evs,
                np.maximum(0, m - s),
                m + s,
                color=astyle["color"],
                alpha=0.15,
                zorder=astyle["zorder"] - 1,
            )

        harmonics_str = f"{(D-2)//2} Harmonics" + (" + Chirp" if (D-2)%2==1 else "")
        ax.set_title(f"Dimension D = {D} ({harmonics_str})", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Quantum Evaluations (NFE)", fontsize=10)
        ax.set_ylabel("Quantum Sagnac Sensitivity (Mean ± Std)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlim(0, budget)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    plt.suptitle(
        "Optimal Multi-Harmonic Floquet Control in Atomtronic Sagnac Accelerometers\nConvergence Trajectories (5 Independent Runs, Carmona-López et al., PRResearch 2026)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    out_png1 = os.path.join(os.path.dirname(__file__), "carmona_quantum_convergence.png")
    out_png2 = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "carmona_quantum_convergence.png")

    fig.savefig(out_png1, dpi=300, bbox_inches="tight")
    fig.savefig(out_png2, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Upgraded Carmona Quantum Convergence generated:\n  1. {out_png1}\n  2. {out_png2}")


if __name__ == "__main__":
    main()
