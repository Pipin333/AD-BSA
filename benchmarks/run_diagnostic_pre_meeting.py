"""
Diagnostic Pre-Meeting Script for AD-BSA v2:
Combines Task 2 (Identity Test) & Task 3 (Minimal Instrumentation)
Evaluates F3 in D=20 with seed=101000, MaxNFE=200,000, pop_size=360, paired P_0.
Saves detailed generation-by-generation telemetry to benchmarks/diagnostic_pre_meeting_f3_20d.json.
"""

import json
import os
import sys
import time
import numpy as np
from opfunu.cec_based import cec2020

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from ad_bsa import AD_BSA, EvaluatorWrapper
from ad_bsa.utils import reflect_boundaries

def run_diagnostic():
    dim = 20
    max_evals = 200000
    base_seed = 101000
    f_bias = 700.0

    print("=" * 80)
    print("AD-BSA v2: DIAGNOSTICO MINIMO PRE-REUNION (F3, D=20)")
    print(f"Semilla: {base_seed} | MaxNFE: {max_evals:,} | Dimension: {dim}")
    print("=" * 80)

    # 1. Instanciar función F3 y generar P_0 pareada
    inst_full = cec2020.F32020(ndim=dim)
    inst_wout = cec2020.F32020(ndim=dim)

    lb, ub = inst_full.lb, inst_full.ub
    pop_size = 18 * dim # 360
    prng = np.random.default_rng(base_seed)
    p0 = lb + prng.random((pop_size, dim)) * (ub - lb)

    eval_full = EvaluatorWrapper(inst_full.evaluate, f_bias=f_bias)
    eval_wout = EvaluatorWrapper(inst_wout.evaluate, f_bias=f_bias)

    bounds = np.column_stack([lb, ub])

    opt_full = AD_BSA(
        eval_full,
        bounds=bounds,
        max_evaluations=max_evals,
        initial_population=p0.copy(),
        repulsion_mode="cosecant",
        seed=base_seed,
    )

    opt_wout = AD_BSA(
        eval_wout,
        bounds=bounds,
        max_evaluations=max_evals,
        initial_population=p0.copy(),
        repulsion_mode="none",
        seed=base_seed,
    )

    opt_full._init_population()
    opt_wout._init_population()

    # Comprobar identidad inicial
    assert np.all(opt_full.Children == opt_wout.Children), "P0 difiere"
    assert np.all(opt_full.fitness == opt_wout.fitness), "Fitness inicial difiere"

    telemetry = []
    first_divergence_gen = None
    divergence_info = {}

    total_generations = 0
    t0 = time.perf_counter()

    while opt_full.evaluations_count < opt_full.max_evaluations:
        opt_full.generation += 1
        opt_wout.generation += 1
        g = opt_full.generation
        total_generations = g

        # 1. Anti-attractors
        bm_full, rc_full = opt_full._awaken_multi_boogeymen()
        bm_wout, rc_wout = opt_wout._awaken_multi_boogeymen()

        # 2. Parameters
        F_safe_f, F_esc_f, _, F_diff_f, CR_f = opt_full._sample_parameters()
        F_safe_w, F_esc_w, _, F_diff_w, CR_w = opt_wout._sample_parameters()

        # 3. Partners
        xbp_f, S_cl_f, xr1_f, xr2_f = opt_full._select_mutation_partners(bm_full)
        xbp_w, S_cl_w, xr1_w, xr2_w = opt_wout._select_mutation_partners(bm_wout)

        # 4. Vectors
        v_safe_f = (xbp_f - opt_full.Children) * F_safe_f[:, np.newaxis]
        v_diff_f = (xr1_f - xr2_f) * F_diff_f[:, np.newaxis]

        v_safe_w = (xbp_w - opt_wout.Children) * F_safe_w[:, np.newaxis]
        v_diff_w = (xr1_w - xr2_w) * F_diff_w[:, np.newaxis]

        # Repulsion calculation in Full
        diff_esc_f = opt_full.Children - S_cl_f
        r_f = np.linalg.norm(diff_esc_f, axis=1, keepdims=True) + 1e-8
        r_norm_f = r_f / (2.0 * rc_full + 1e-8)

        arg_f = np.clip(r_norm_f * np.pi * 0.5, 1e-3, 0.999 * np.pi)
        raw_csc_f = np.abs(1.0 / np.sin(arg_f))
        fp_f = np.clip(raw_csc_f, 1.0, opt_full.max_repulsion_force) - 1.0
        fp_f = np.where(r_norm_f < 2.0, fp_f, 0.0)

        unit_esc_f = diff_esc_f / r_f
        v_esc_f = F_esc_f[:, np.newaxis] * fp_f * unit_esc_f * rc_full

        # In w/o Repulsion:
        v_esc_w = np.zeros_like(v_safe_w)

        # Telemetry measurements on Full
        r_norm_flat = r_norm_f.flatten()
        norm_esc = np.linalg.norm(v_esc_f, axis=1)
        norm_safe = np.linalg.norm(v_safe_f, axis=1)

        active_mask = r_norm_flat < 2.0
        frac_active = float(np.mean(active_mask))
        is_esc_zero = bool(np.all(v_esc_f == 0.0))

        gen_record = {
            "gen": g,
            "pop_size": int(opt_full.current_pop_size),
            "evals": int(opt_full.evaluations_count),
            "capture_radius": float(rc_full),
            "frac_r_norm_lt_2": frac_active,
            "count_active": int(np.sum(active_mask)),
            "is_v_esc_identically_zero": is_esc_zero,
            "mean_norm_v_esc": float(np.mean(norm_esc)),
            "max_norm_v_esc": float(np.max(norm_esc)),
            "mean_norm_v_safe": float(np.mean(norm_safe)),
            "min_r_norm": float(np.min(r_norm_flat)),
            "med_r_norm": float(np.median(r_norm_flat)),
            "max_r_norm": float(np.max(r_norm_flat)),
        }
        telemetry.append(gen_record)

        # Mutants
        mut_f = reflect_boundaries(opt_full.Children + v_safe_f + v_esc_f + v_diff_f, opt_full.lower_bounds, opt_full.upper_bounds, opt_full.Children)
        mut_w = reflect_boundaries(opt_wout.Children + v_safe_w + v_esc_w + v_diff_w, opt_wout.lower_bounds, opt_wout.upper_bounds, opt_wout.Children)

        # Crossover
        rand_j_f = opt_full.rng.integers(0, opt_full.dim, size=opt_full.current_pop_size)
        j_mat_f = np.tile(np.arange(opt_full.dim), (opt_full.current_pop_size, 1))
        cross_mask_f = (opt_full.rng.random((opt_full.current_pop_size, opt_full.dim)) <= CR_f[:, np.newaxis]) | (j_mat_f == rand_j_f[:, np.newaxis])
        trial_f = np.where(cross_mask_f, mut_f, opt_full.Children)

        rand_j_w = opt_wout.rng.integers(0, opt_wout.dim, size=opt_wout.current_pop_size)
        j_mat_w = np.tile(np.arange(opt_wout.dim), (opt_wout.current_pop_size, 1))
        cross_mask_w = (opt_wout.rng.random((opt_wout.current_pop_size, opt_wout.dim)) <= CR_w[:, np.newaxis]) | (j_mat_w == rand_j_w[:, np.newaxis])
        trial_w = np.where(cross_mask_w, mut_w, opt_wout.Children)

        # Check divergence
        if first_divergence_gen is None:
            if not np.allclose(opt_full.Children, opt_wout.Children, atol=1e-15):
                first_divergence_gen = g
                divergence_info = {
                    "gen": g,
                    "reason": "Children coordinates differed",
                    "max_coord_diff": float(np.max(np.abs(opt_full.Children - opt_wout.Children))),
                }
            elif not is_esc_zero:
                # escape vector was non-zero
                first_divergence_gen = g
                divergence_info = {
                    "gen": g,
                    "reason": "Non-zero escape vector applied",
                    "max_v_esc": float(np.max(norm_esc)),
                }

        # Objective eval
        evals_f = min(opt_full.current_pop_size, opt_full.max_evaluations - opt_full.evaluations_count)
        evals_w = min(opt_wout.current_pop_size, opt_wout.max_evaluations - opt_wout.evaluations_count)

        tfits_f = opt_full.objective_func(trial_f[:evals_f])
        tfits_w = opt_wout.objective_func(trial_w[:evals_w])

        opt_full.evaluations_count += evals_f
        opt_wout.evaluations_count += evals_w

        # Selection
        succ_f = np.where(tfits_f <= opt_full.fitness[:evals_f])[0]
        succ_w = np.where(tfits_w <= opt_wout.fitness[:evals_w])[0]

        opt_full._update_archive(opt_full.Children[succ_f])
        opt_full.Children[succ_f] = trial_f[succ_f]
        opt_full.fitness[succ_f] = tfits_f[succ_f]

        opt_wout._update_archive(opt_wout.Children[succ_w])
        opt_wout.Children[succ_w] = trial_w[succ_w]
        opt_wout.fitness[succ_w] = tfits_w[succ_w]

        # Best tracking
        best_f_cur = np.min(opt_full.fitness)
        if best_f_cur < opt_full.best_fitness:
            opt_full.best_fitness = float(best_f_cur)
            opt_full.best_position = opt_full.Children[np.argmin(opt_full.fitness)].copy()

        best_w_cur = np.min(opt_wout.fitness)
        if best_w_cur < opt_wout.best_fitness:
            opt_wout.best_fitness = float(best_w_cur)
            opt_wout.best_position = opt_wout.Children[np.argmin(opt_wout.fitness)].copy()

        opt_full._boogeyman_lpsr()
        opt_wout._boogeyman_lpsr()

    elapsed = time.perf_counter() - t0

    # Consolidate analysis
    active_gens = [t for t in telemetry if not t["is_v_esc_identically_zero"]]
    all_esc_norms = [t["mean_norm_v_esc"] for t in telemetry]
    all_safe_norms = [t["mean_norm_v_safe"] for t in telemetry]
    all_fracs = [t["frac_r_norm_lt_2"] for t in telemetry]
    all_min_r = [t["min_r_norm"] for t in telemetry]
    all_med_r = [t["med_r_norm"] for t in telemetry]
    all_max_r = [t["max_r_norm"] for t in telemetry]

    print("\n" + "=" * 80)
    print("RESULTADOS DEL TEST DE IDENTIDAD E INSTRUMENTACION")
    print("=" * 80)
    print(f"Tiempo transcurrido: {elapsed:.2f}s | Generaciones totales: {total_generations}")
    print(f"AD-BSA (Full) Error Final:           {max(0.0, opt_full.best_fitness - f_bias):.10f}")
    print(f"AD-BSA (w/o Repulsion) Error Final:   {max(0.0, opt_wout.best_fitness - f_bias):.10f}")
    print(f"Diferencia final de fitness:         {abs(opt_full.best_fitness - opt_wout.best_fitness):.10e}")
    print(f"Generaciones con v_esc != 0:         {len(active_gens)} de {total_generations} ({len(active_gens)/total_generations*100:.2f}%)")
    print(f"Primer evento de divergencia:        {first_divergence_gen} ({divergence_info})")
    print(f"Fraccion media de individuos r_norm<2: {np.mean(all_fracs)*100:.3f}% (Max: {np.max(all_fracs)*100:.2f}%)")
    print(f"r_norm a lo largo de la corrida:     Min global={np.min(all_min_r):.4f}, Mediana global={np.median(all_med_r):.4f}, Max global={np.max(all_max_r):.4f}")
    print(f"Norma media v_esc:                   {np.mean(all_esc_norms):.6e}")
    print(f"Norma media v_safe:                  {np.mean(all_safe_norms):.6e}")

    # Guardar JSON
    output_json = "benchmarks/diagnostic_pre_meeting_f3_20d.json"
    summary_data = {
        "metadata": {
            "experiment": "AD-BSA v2 Pre-Meeting Diagnostic",
            "function": "F3",
            "dimension": dim,
            "max_evaluations": max_evals,
            "base_seed": base_seed,
            "pop_size_init": pop_size,
            "elapsed_seconds": elapsed,
        },
        "identity_test": {
            "total_generations": total_generations,
            "full_best_fitness": float(opt_full.best_fitness),
            "wout_best_fitness": float(opt_wout.best_fitness),
            "final_fitness_delta": float(abs(opt_full.best_fitness - opt_wout.best_fitness)),
            "first_divergence_generation": first_divergence_gen,
            "divergence_details": divergence_info,
            "generations_with_nonzero_escape": len(active_gens),
            "percentage_generations_active": float(len(active_gens) / total_generations * 100.0),
        },
        "aggregate_telemetry": {
            "mean_fraction_r_norm_lt_2": float(np.mean(all_fracs)),
            "max_fraction_r_norm_lt_2": float(np.max(all_fracs)),
            "min_r_norm_observed": float(np.min(all_min_r)),
            "median_r_norm_observed": float(np.median(all_med_r)),
            "max_r_norm_observed": float(np.max(all_max_r)),
            "mean_norm_v_esc": float(np.mean(all_esc_norms)),
            "max_norm_v_esc": float(np.max([t["max_norm_v_esc"] for t in telemetry])),
            "mean_norm_v_safe": float(np.mean(all_safe_norms)),
        },
        "generation_telemetry": telemetry,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[OK] Telemetria completa guardada en: {output_json}")

if __name__ == "__main__":
    run_diagnostic()
