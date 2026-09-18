"""
================================================================================
  AD-BSA: Component Contribution & Ablation Study Runner
  Isolates individual contributions of:
    1. Bounded Cosecant Repulsion (|csc| Operator)
    2. Multi-Tier Anti-Attractor Stratification (M=2 vs M=1)
    3. Thermodynamic Annealing Schedule (gamma=1.5 vs gamma=0)
================================================================================
"""

import os
import sys

# Limit worker BLAS/OMP threads to prevent CPU lockup
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from ad_bsa import AD_BSA, EvaluatorWrapper
from opfunu.cec_based import cec2020

ABLATION_FUNCS = [
    ("F1", "F12020", 100.0, "Unimodal (Bent Cigar)"),
    ("F2", "F22020", 1100.0, "Multimodal (Schwefel)"),
    ("F4", "F42020", 1900.0, "Multimodal (Rosenbrock+Griewank)"),
    ("F8", "F82020", 2200.0, "Composition 1 (Multi-Basin)"),
]

VARIANTS = {
    "Full AD-BSA (Proposed)": {},
    "w/o Cosecant Repulsion (v_esc = 0)": {"max_repulsion_force": 1.0},
    "w/o Thermodynamic Annealing (gamma = 0)": {"annealing_power": 0.0},
    "w/o Stratification (Single Centroid M = 1)": {"num_boogeymen": 1},
}

def run_single_ablation(args):
    v_name, f_id, f_class_name, f_bias, run_idx, max_evals, seed = args
    f_cls = getattr(cec2020, f_class_name)
    inst = f_cls(ndim=50)
    evaluator = EvaluatorWrapper(inst.evaluate, f_bias=f_bias)
    bounds = np.column_stack([inst.lb, inst.ub])

    kwargs = VARIANTS[v_name].copy()
    kwargs["max_evaluations"] = max_evals
    kwargs["seed"] = seed

    opt = AD_BSA(evaluator, bounds=bounds, **kwargs)
    res = opt.optimize()
    err = float(max(0.0, res.best_fitness - f_bias))
    return v_name, f_id, run_idx, err

def main():
    parser = argparse.ArgumentParser(description="Ejecutar Estudio de Ablación de AD-BSA (CEC 2020 50D)")
    parser.add_argument("--runs", type=int, default=5, help="Corridas independientes por función (def: 5)")
    parser.add_argument("--max-evals", type=int, default=50000, help="Presupuesto MaxNFE (def: 50000)")
    parser.add_argument("--workers", type=int, default=4, help="Procesos paralelos (def: 4)")
    args = parser.parse_args()

    print("=" * 85, flush=True)
    print("  AD-BSA: ABLATION STUDY & COMPONENT CONTRIBUTION ANALYSIS", flush=True)
    print(f"  Dimension: 50 | MaxNFE: {args.max_evals:,} | Runs: {args.runs} | Workers: {args.workers}", flush=True)
    print("=" * 85, flush=True)

    tasks = []
    for f_id, f_class, f_bias, desc in ABLATION_FUNCS:
        f_num = int(f_id.replace("F", ""))
        for v_idx, v_name in enumerate(VARIANTS):
            for r_idx in range(args.runs):
                seed = 50000 * (r_idx + 1) + 500 * f_num + v_idx
                tasks.append((v_name, f_id, f_class, f_bias, r_idx, args.max_evals, seed))

    total = len(tasks)
    print(f"[+] Total tareas a ejecutar: {total} ({len(VARIANTS)} variantes x {len(ABLATION_FUNCS)} funciones x {args.runs} corridas)\n", flush=True)

    results = {v: {f[0]: [] for f in ABLATION_FUNCS} for v in VARIANTS}
    t0 = time.perf_counter()
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_ablation, t): t for t in tasks}
        for fut in as_completed(futures):
            v_name, f_id, r_idx, err = fut.result()
            results[v_name][f_id].append(err)
            done += 1
            if done % 10 == 0 or done == total:
                elapsed = time.perf_counter() - t0
                print(f"  [{done:2d}/{total}] Progreso: {done/total*100:5.1f}% | Transcurrido: {elapsed:.1f}s", flush=True)

    print("\n" + "=" * 110)
    print(f"{'Ablation Variant':45s} | " + " | ".join(f"{f[0]:12s}" for f in ABLATION_FUNCS) + " | Friedman Rank")
    print("-" * 110)

    # Calcular rankings promedio
    ranks = {v: [] for v in VARIANTS}
    for f_id, _, _, _ in ABLATION_FUNCS:
        means = {v: float(np.mean(results[v][f_id])) for v in VARIANTS}
        sorted_v = sorted(VARIANTS.keys(), key=lambda x: means[x])
        for r_idx, v in enumerate(sorted_v, 1):
            ranks[v].append(r_idx)

    for v_name in VARIANTS:
        row = [f"{v_name:45s}"]
        for f_id, _, _, _ in ABLATION_FUNCS:
            m = float(np.mean(results[v_name][f_id]))
            row.append(f"{m:12.2e}")
        row.append(f"{np.mean(ranks[v_name]):13.2f}")
        print(" | ".join(row))
    print("=" * 110)

if __name__ == "__main__":
    main()
