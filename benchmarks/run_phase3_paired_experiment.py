"""
================================================================================
  AD-BSA Phase 3: Definitive Paired PoC Experiment (20D, 10k*D Budget, 15 Runs)
  Variants:
    1. AD-BSA (Full: Cosecant Repulsion + Anti-Attractors)
    2. AD-BSA (w/o Repulsion: v_esc = 0)
    3. AD-BSA (Random Direction: identical ||v_esc||, isotropic random direction)
    4. L-SHADE (State-of-the-art Reference)
  Protocol:
    - Exactly paired initial population P_0 and random seed per run and function
    - Dimension D = 20, MaxNFE = 200,000 (10,000 * D)
    - 15 independent runs per function
    - Informative continuous benchmark functions: F3, F4, F5, F6, F7, F10
================================================================================
"""

import os
import sys

# Prevent OpenBLAS/OMP thread contention across parallel workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
from opfunu.cec_based import cec2020

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from ad_bsa import AD_BSA, L_SHADE, EvaluatorWrapper

INFORMATIVE_FUNCS = [
    ("F3", "F32020", "Shifted and Rotated Lunacek bi-Rastrigin", 700.0),
    ("F4", "F42020", "Expanded Rosenbrock + Griewank", 1900.0),
    ("F5", "F52020", "Hybrid Function 1 (N=3)", 1700.0),
    ("F6", "F62020", "Hybrid Function 2 (N=4)", 1600.0),
    ("F7", "F72020", "Hybrid Function 3 (N=5)", 2100.0),
    ("F10", "F102020", "Composition Function 3 (N=5)", 2500.0),
]

VARIANTS = [
    "AD-BSA (Full)",
    "AD-BSA (w/o Repulsion)",
    "AD-BSA (Random Direction)",
    "L-SHADE (Reference)",
]

def generate_paired_task_spec(func_tuple, run_idx, dim, max_evals):
    """Generate paired task configurations with identical initial population P_0."""
    func_id, func_class_name, desc, f_bias = func_tuple
    f_num = int(func_id.replace("F", ""))
    base_seed = 100000 * (run_idx + 1) + 1000 * f_num

    # Initial population generation with base seed
    f_class = getattr(cec2020, func_class_name)
    inst = f_class(ndim=dim)
    lb, ub = inst.lb, inst.ub
    bound_range = ub - lb

    # Both L-SHADE and AD-BSA in 20D use canonical 18 * D = 360 individuals
    pop_size = 18 * dim
    prng = np.random.default_rng(base_seed)
    p0 = lb + prng.random((pop_size, dim)) * bound_range

    return {
        "func_id": func_id,
        "func_class_name": func_class_name,
        "desc": desc,
        "f_bias": f_bias,
        "run_idx": run_idx,
        "dim": dim,
        "max_evals": max_evals,
        "base_seed": base_seed,
        "initial_population": p0,
    }

def run_single_variant_experiment(spec: Dict, variant: str) -> Dict:
    """Execute one optimization run for a specific variant with paired setup."""
    f_class = getattr(cec2020, spec["func_class_name"])
    inst = f_class(ndim=spec["dim"])
    evaluator = EvaluatorWrapper(inst.evaluate, f_bias=spec["f_bias"])
    bounds = np.column_stack([inst.lb, inst.ub])
    p0 = spec["initial_population"].copy()
    seed = spec["base_seed"]

    t0 = time.perf_counter()
    if variant == "AD-BSA (Full)":
        opt = AD_BSA(
            evaluator,
            bounds=bounds,
            max_evaluations=spec["max_evals"],
            initial_population=p0,
            repulsion_mode="cosecant",
            seed=seed,
        )
    elif variant == "AD-BSA (w/o Repulsion)":
        opt = AD_BSA(
            evaluator,
            bounds=bounds,
            max_evaluations=spec["max_evals"],
            initial_population=p0,
            repulsion_mode="none",
            seed=seed,
        )
    elif variant == "AD-BSA (Random Direction)":
        opt = AD_BSA(
            evaluator,
            bounds=bounds,
            max_evaluations=spec["max_evals"],
            initial_population=p0,
            repulsion_mode="random_direction",
            seed=seed,
        )
    elif variant == "L-SHADE (Reference)":
        opt = L_SHADE(
            evaluator,
            bounds=bounds,
            max_evaluations=spec["max_evals"],
            initial_population=p0,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    res = opt.optimize()
    elapsed = time.perf_counter() - t0
    error = float(max(0.0, res.best_fitness - spec["f_bias"]))

    return {
        "variant": variant,
        "func": spec["func_id"],
        "run": spec["run_idx"],
        "error": error,
        "best_fitness": float(res.best_fitness),
        "evaluations": int(res.total_evaluations),
        "time": elapsed,
    }

def bootstrap_median_difference(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Compute difference in medians with bootstrap confidence intervals."""
    diff_orig = float(np.median(x) - np.median(y))
    rng = np.random.default_rng(42)
    n = len(x)
    boot_diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[b] = np.median(x[idx]) - np.median(y[idx])

    alpha = 1.0 - ci
    low = float(np.percentile(boot_diffs, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_diffs, 100.0 * (1.0 - alpha / 2.0)))
    return diff_orig, low, high

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Definitive Paired PoC Experiment (20D)")
    parser.add_argument("--runs", type=int, default=15, help="Number of independent paired runs (def: 15)")
    parser.add_argument("--max-evals", type=int, default=200000, help="MaxNFE budget (def: 200,000 = 10,000 * D)")
    parser.add_argument("--dim", type=int, default=20, help="Problem dimension (def: 20)")
    parser.add_argument("--workers", type=int, default=6, help="ProcessPool workers (def: 6)")
    parser.add_argument("--output", type=str, default="benchmarks/phase3_poc_results.json", help="Output JSON path")
    args = parser.parse_args()

    checkpoint_file = args.output.replace(".json", "_checkpoint.json")

    print("=" * 85, flush=True)
    print("  AD-BSA PHASE 3: DEFINITIVE PAIRED POC BENCHMARK", flush=True)
    print(f"  Dimension: {args.dim} | MaxNFE: {args.max_evals:,} (10,000 * D)", flush=True)
    print(f"  Runs: {args.runs} | Functions: {[f[0] for f in INFORMATIVE_FUNCS]}", flush=True)
    print(f"  Variants: {VARIANTS}", flush=True)
    print(f"  Workers: {args.workers} | Checkpoint: {checkpoint_file}", flush=True)
    print("=" * 85, flush=True)

    completed_runs = []
    completed_keys = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as cf:
                completed_runs = json.load(cf)
                completed_keys = {(r["variant"], r["func"], r["run"]) for r in completed_runs if r["run"] < args.runs}
                completed_runs = [r for r in completed_runs if r["run"] < args.runs]
            print(f"[+] Reanudando desde checkpoint: {len(completed_keys)} corridas ya procesadas.", flush=True)
        except Exception as e:
            print(f"[!] Checkpoint corrupto o ilegible ({e}), comenzando de nuevo.", flush=True)
            completed_runs = []
            completed_keys = set()

    # Build work queue
    tasks = []
    for f_tuple in INFORMATIVE_FUNCS:
        for r_idx in range(args.runs):
            spec = generate_paired_task_spec(f_tuple, r_idx, args.dim, args.max_evals)
            for v in VARIANTS:
                if (v, spec["func_id"], r_idx) in completed_keys:
                    continue
                tasks.append((spec, v))

    total_tasks = len(INFORMATIVE_FUNCS) * args.runs * len(VARIANTS)
    pending = len(tasks)
    print(f"[+] Total tareas requeridas: {total_tasks} ({len(completed_keys)} completadas, {pending} pendientes).\n", flush=True)

    if pending > 0:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(run_single_variant_experiment, s, v): (v, s["func_id"], s["run_idx"]) for s, v in tasks}
            done_count = len(completed_keys)

            for future in as_completed(future_to_task):
                try:
                    res = future.result()
                    completed_runs.append(res)
                    done_count += 1

                    if done_count % 5 == 0 or done_count == total_tasks:
                        elapsed = time.perf_counter() - t0
                        pct = (done_count / total_tasks) * 100.0
                        eta = (elapsed / (done_count - len(completed_keys))) * (total_tasks - done_count) if (done_count > len(completed_keys)) else 0
                        print(f"  [{done_count:3d}/{total_tasks}] Progreso: {pct:5.1f}% | Transcurrido: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m | {res['variant'][:15]} {res['func']} run {res['run']}: err={res['error']:.2e}", flush=True)
                        
                        # Save checkpoint
                        with open(checkpoint_file, "w", encoding="utf-8") as cf:
                            json.dump(completed_runs, cf, indent=2)
                except Exception as e:
                    task_info = future_to_task[future]
                    print(f"[!] Error ejecutando tarea {task_info}: {e}", flush=True)

    # Save final consolidated JSON
    with open(checkpoint_file, "w", encoding="utf-8") as cf:
        json.dump(completed_runs, cf, indent=2)

    # Perform Statistical Analysis
    analyze_and_report_phase3(completed_runs, args.runs, args.output)

def analyze_and_report_phase3(completed_runs: List[Dict], num_runs: int, output_path: str):
    funcs = [f[0] for f in INFORMATIVE_FUNCS]
    data_by_func = {f: {v: [] for v in VARIANTS} for f in funcs}

    for r in completed_runs:
        if r["func"] in data_by_func and r["variant"] in data_by_func[r["func"]]:
            data_by_func[r["func"]][r["variant"]].append((r["run"], r["error"]))

    # Sort runs by run index to enforce pairing
    paired_errors = {f: {} for f in funcs}
    for f in funcs:
        for v in VARIANTS:
            runs_list = sorted(data_by_func[f][v], key=lambda x: x[0])
            paired_errors[f][v] = np.array([x[1] for x in runs_list], dtype=np.float64)

    print("\n" + "=" * 90)
    print("  PHASE 3 STATISTICAL ANALYSIS & DECISION RULE EVALUATION")
    print("=" * 90)

    # 1. Table of Medians & Means
    print(f"\n{'Func':5s} | {'AD-BSA (Full)':20s} | {'w/o Repulsion':20s} | {'Random Dir':20s} | {'L-SHADE':20s}")
    print("-" * 95)
    for f in funcs:
        row_str = f"{f:5s} | "
        for v in VARIANTS:
            arr = paired_errors[f][v]
            row_str += f"{np.median(arr):10.2e} (±{np.std(arr):.1e}) | "
        print(row_str)

    # 2. Paired Wilcoxon Signed-Rank Tests vs AD-BSA (Full)
    comparisons = [
        ("AD-BSA (w/o Repulsion)", "Repulsion Removal Check"),
        ("AD-BSA (Random Direction)", "Noise vs Negative Information Check"),
        ("L-SHADE (Reference)", "Benchmark State of the Art Check"),
    ]

    summary_stats = {}
    p_values_all = []

    for comp_var, label in comparisons:
        print(f"\n--- Comparative Analysis: AD-BSA (Full) vs {comp_var} [{label}] ---")
        p_vals_per_f = []
        for f in funcs:
            full_arr = paired_errors[f]["AD-BSA (Full)"]
            comp_arr = paired_errors[f][comp_var]

            diffs = full_arr - comp_arr
            med_diff, ci_low, ci_high = bootstrap_median_difference(full_arr, comp_arr)

            if np.allclose(diffs, 0.0, atol=1e-12):
                pval = 1.0
                sign = "="
            else:
                try:
                    res = stats.wilcoxon(full_arr, comp_arr, alternative="two-sided")
                    pval = float(res.pvalue)
                except Exception:
                    pval = 1.0

                if pval < 0.05:
                    sign = "+" if med_diff < 0 else "-"
                else:
                    sign = "="

            p_vals_per_f.append(pval)
            p_values_all.append(pval)
            print(f"  {f:5s}: Wilcoxon p={pval:8.4e} ({sign}) | Median Diff={med_diff:10.2e} [95% CI: {ci_low:10.2e}, {ci_high:10.2e}]")

    # Save summary dictionary
    out_dict = {
        "metadata": {
            "dimension": 20,
            "max_evaluations": 200000,
            "num_runs": num_runs,
            "functions": funcs,
            "variants": VARIANTS,
        },
        "paired_errors": {f: {v: paired_errors[f][v].tolist() for v in VARIANTS} for f in funcs},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\n[OK] Resultados finales guardados en: {output_path}")

if __name__ == "__main__":
    main()
