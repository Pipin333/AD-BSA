"""
Phase 2: AD-BSA Dynamic Instrumentation Script
Monitors generation-by-generation:
  1. Fraction of individuals with phi > 0
  2. Mean value of phi across population
  3. Magnitude of v_escape relative to positive attraction term ||v_escape|| / ||v_safe||
Evaluates D in [5, 20, 50] on F3, F4, F10 for 1 run per configuration.
"""

import os
import sys
import numpy as np
from opfunu.cec_based import cec2020

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from ad_bsa import AD_BSA, EvaluatorWrapper
from ad_bsa.utils import reflect_boundaries

def run_instrumented_ad_bsa(func_class_name, f_bias, dim, max_evals=20000, seed=42):
    f_cls = getattr(cec2020, func_class_name)
    inst = f_cls(ndim=dim)
    evaluator = EvaluatorWrapper(inst.evaluate, f_bias=f_bias)
    bounds = np.column_stack([inst.lb, inst.ub])

    opt = AD_BSA(evaluator, bounds=bounds, max_evaluations=max_evals, seed=seed)
    
    # Custom instrumented loop
    opt._init_population()
    
    records = []
    
    while opt.evaluations_count < opt.max_evaluations:
        opt.generation += 1

        # 1. Anti-attractors
        boogeymen_pos, capture_radius = opt._awaken_multi_boogeymen()

        # 2. Hyperparameters
        (F_safe, F_escape, raw_F_escape, F_diff, CR) = opt._sample_parameters()

        # 3. Mutation partners
        (x_best_p, S_closest, x_r1, x_r2) = opt._select_mutation_partners(boogeymen_pos)

        safe_vector = (x_best_p - opt.Children) * F_safe[:, np.newaxis]
        diff_vector = (x_r1 - x_r2) * F_diff[:, np.newaxis]

        # 4. Repulsion barrier calculation
        diff_escape = opt.Children - S_closest
        r = np.linalg.norm(diff_escape, axis=1, keepdims=True) + 1e-8
        r_norm = r / (2.0 * capture_radius + 1e-8)

        arg = np.clip(r_norm * np.pi * 0.5, 1e-3, 0.999 * np.pi)
        raw_csc = np.abs(1.0 / np.sin(arg))
        force_profile = (
            np.clip(raw_csc, 1.0, opt.max_repulsion_force) - 1.0
        )
        force_profile = np.where(r_norm < 2.0, force_profile, 0.0)

        unit_escape = diff_escape / r
        escape_vector = (
            F_escape[:, np.newaxis]
            * force_profile
            * unit_escape
            * capture_radius
        )

        # Metrics collection
        phi_vals = force_profile.flatten()
        frac_active = float(np.mean(phi_vals > 0.0))
        mean_phi = float(np.mean(phi_vals))
        
        norm_escape = np.linalg.norm(escape_vector, axis=1)
        norm_safe = np.linalg.norm(safe_vector, axis=1)
        mean_norm_escape = float(np.mean(norm_escape))
        mean_norm_safe = float(np.mean(norm_safe))
        ratio_escape_safe = float(mean_norm_escape / (mean_norm_safe + 1e-12))

        records.append({
            "gen": opt.generation,
            "frac_active": frac_active,
            "mean_phi": mean_phi,
            "mean_norm_escape": mean_norm_escape,
            "mean_norm_safe": mean_norm_safe,
            "ratio_escape_safe": ratio_escape_safe,
            "pop_size": opt.current_pop_size,
            "capture_radius": float(capture_radius),
        })

        # Remainder of optimization step to maintain authentic dynamics
        mutant_children = reflect_boundaries(
            opt.Children + safe_vector + escape_vector + diff_vector,
            lb=opt.lower_bounds,
            ub=opt.upper_bounds,
            base=opt.Children,
        )

        rand_j = opt.rng.integers(0, opt.dim, size=opt.current_pop_size)
        j_matrix = np.tile(np.arange(opt.dim), (opt.current_pop_size, 1))
        j_rand_mask = j_matrix == rand_j[:, np.newaxis]
        crossover_mask = (
            opt.rng.random((opt.current_pop_size, opt.dim))
            <= CR[:, np.newaxis]
        ) | j_rand_mask
        trial_children = np.where(
            crossover_mask, mutant_children, opt.Children
        )

        evals_to_run = min(
            opt.current_pop_size,
            opt.max_evaluations - opt.evaluations_count,
        )
        if evals_to_run < opt.current_pop_size:
            trial_children = trial_children[:evals_to_run]

        trial_fitness = opt.objective_func(trial_children)
        opt.evaluations_count += evals_to_run

        improvements_mask = trial_fitness <= opt.fitness[:evals_to_run]
        successful_indices = np.where(improvements_mask)[0]

        opt._update_archive(opt.Children[successful_indices])
        opt.Children[successful_indices] = trial_children[successful_indices]
        opt.fitness[successful_indices] = trial_fitness[successful_indices]

        best_cur = np.min(opt.fitness)
        if best_cur < opt.best_fitness:
            opt.best_fitness = float(best_cur)
            opt.best_position = opt.Children[np.argmin(opt.fitness)].copy()

        opt._boogeyman_lpsr()

    return records

def analyze_records(records, label):
    fracs = [r["frac_active"] for r in records]
    phis = [r["mean_phi"] for r in records]
    ratios = [r["ratio_escape_safe"] for r in records]
    
    print(f"\n--- {label} ---")
    print(f"  Total generations: {len(records)}")
    print(f"  Active phi (>0) fraction: Mean={np.mean(fracs)*100:.1f}%, Min={np.min(fracs)*100:.1f}%, Max={np.max(fracs)*100:.1f}%")
    print(f"  Mean phi value:           Mean={np.mean(phis):.4f}, Max={np.max(phis):.4f}")
    print(f"  ||v_esc|| / ||v_safe||:   Mean={np.mean(ratios):.4f}, Max={np.max(ratios):.4f}")

if __name__ == "__main__":
    configs = [
        ("F32020", 700.0, "F3 (Lunacek)"),
        ("F42020", 1900.0, "F4 (Rosenbrock+Griewank)"),
        ("F102020", 2500.0, "F10 (Composition 3)"),
    ]
    dims = [5, 20, 50]

    for f_cls, bias, name in configs:
        for d in dims:
            max_evals = 10000 if d <= 20 else 20000
            recs = run_instrumented_ad_bsa(f_cls, bias, dim=d, max_evals=max_evals, seed=42)
            analyze_records(recs, f"{name} in D={d}")
