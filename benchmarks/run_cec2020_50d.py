"""
================================================================================
  AD-BSA Benchmark Suite: Official IEEE CEC 2020 (50 Dimensions) Runner
  Evaluates:
    1. AD-BSA (Proposed Algorithm with Bounded Cosecant Repulsion |csc|)
    2. jSO (IEEE CEC 2017 Winner)
    3. CMA-ES (Hansen Covariance Matrix Adaptation)
    4. L-SHADE (IEEE CEC 2014 Winner)
    5. Standard DE (DE/rand/1/bin)
    6. Standard PSO (Global Best PSO)
    7. Canonical Cuckoo Search (CS)
================================================================================
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np
from opfunu.cec_based import cec2020
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from ad_bsa import (
    AD_BSA,
    jSO,
    CMA_ES,
    L_SHADE,
    CanonicalCuckooSearch,
    StandardDE,
    StandardPSO,
    EvaluatorWrapper,
    compute_wilcoxon,
)

ALGORITHMS = {
    "AD-BSA": AD_BSA,
    "jSO": jSO,
    "CMA-ES": CMA_ES,
    "L-SHADE": L_SHADE,
    "Cuckoo-Search": CanonicalCuckooSearch,
    "Standard-DE": StandardDE,
    "Standard-PSO": StandardPSO,
}

CEC2020_FUNCS = [
    ("F1", "F12020", "Shifted and Rotated Bent Cigar", 100.0),
    ("F2", "F22020", "Shifted and Rotated Schwefel", 1100.0),
    ("F3", "F32020", "Shifted and Rotated Lunacek bi-Rastrigin", 700.0),
    ("F4", "F42020", "Expanded Rosenbrock + Griewank", 1900.0),
    ("F5", "F52020", "Hybrid Function 1", 1700.0),
    ("F6", "F62020", "Hybrid Function 2", 1600.0),
    ("F7", "F72020", "Hybrid Function 3", 2100.0),
    ("F8", "F82020", "Composition Function 1", 2200.0),
    ("F9", "F92020", "Composition Function 2", 2400.0),
    ("F10", "F102020", "Composition Function 3", 2500.0),
]


def run_single_experiment(args_tuple):
    algo_name, func_id, func_class_name, f_bias, run_idx, max_evals, seed = args_tuple
    f_class = getattr(cec2020, func_class_name)
    inst = f_class(ndim=50)

    evaluator = EvaluatorWrapper(inst.evaluate, f_bias=f_bias)
    bounds = np.column_stack([inst.lb, inst.ub])

    algo_class = ALGORITHMS[algo_name]
    opt = algo_class(evaluator, bounds=bounds, max_evaluations=max_evals, seed=seed)
    res = opt.optimize()

    error = float(max(0.0, res.best_fitness - f_bias))
    return {
        "algo": algo_name,
        "func": func_id,
        "run": run_idx,
        "best_fitness": float(res.best_fitness),
        "error": error,
        "evaluations": int(res.total_evaluations),
        "time": float(res.execution_time),
    }


def consolidate_benchmark_data(raw_runs: List[Dict], num_runs: int, max_evals: int) -> Dict:
    """Calcula estadísticas consolidadas, tests de Wilcoxon y rankings de Friedman."""
    funcs = [f[0] for f in CEC2020_FUNCS]
    algos = list(ALGORITHMS.keys())

    # Agrupar errores por función y algoritmo
    by_func = {f: {a: [] for a in algos} for f in funcs}
    for r in raw_runs:
        by_func[r["func"]][r["algo"]].append(r["error"])

    summary = {}
    wilcoxon_results = {}

    for f in funcs:
        summary[f] = {}
        wilcoxon_results[f] = {}
        ad_bsa_scores = by_func[f]["AD-BSA"]

        for a in algos:
            scores = np.asarray(by_func[f][a], dtype=np.float64)
            summary[f][a] = {
                "mean_error": float(np.mean(scores)),
                "std_error": float(np.std(scores)),
                "median_error": float(np.median(scores)),
                "best_error": float(np.min(scores)),
                "worst_error": float(np.max(scores)),
            }
            if a != "AD-BSA":
                pval, sign = compute_wilcoxon(ad_bsa_scores, scores)
                wilcoxon_results[f][a] = {"p_value": pval, "sign": sign}

    # Rankings de Friedman
    ranks = {a: [] for a in algos}
    for f in funcs:
        means = {a: summary[f][a]["mean_error"] for a in algos}
        sorted_algos = sorted(algos, key=lambda x: means[x])
        for rank, a in enumerate(sorted_algos, 1):
            ranks[a].append(rank)

    friedman_ranks = {a: float(np.mean(ranks[a])) for a in algos}

    return {
        "metadata": {
            "benchmark_suite": "IEEE CEC 2020",
            "dimension": 50,
            "bounds": [-100.0, 100.0],
            "max_evaluations": max_evals,
            "num_runs": num_runs,
            "algorithms": algos,
            "functions": [f[0] for f in CEC2020_FUNCS],
        },
        "friedman_ranks": friedman_ranks,
        "summary_statistics": summary,
        "wilcoxon_tests_vs_ad_bsa": wilcoxon_results,
        "raw_runs": raw_runs,
    }


def main():
    parser = argparse.ArgumentParser(description="Ejecutar Benchmark CEC 2020 en 50 Dimensiones")
    parser.add_argument("--runs", type=int, default=5, help="Número de ejecuciones independientes por función")
    parser.add_argument("--max-evals", type=int, default=50000, help="Presupuesto MaxNFE por corrida")
    parser.add_argument("--workers", type=int, default=8, help="Número de procesos paralelos")
    parser.add_argument("--output", type=str, default="benchmarks/cec2020_50d_results.json", help="Ruta de guardado")
    args = parser.parse_args()

    print("=" * 85)
    print(f"  IEEE CEC 2020 BENCHMARK SUITE - 50 DIMENSIONS")
    print(f"  Algoritmos: {list(ALGORITHMS.keys())}")
    print(f"  Configuración: {args.runs} corridas x 10 funciones | MaxNFE: {args.max_evals:,}")
    print("=" * 85)

    tasks = []
    base_seed = 2026
    for func_id, func_class, desc, bias in CEC2020_FUNCS:
        for algo in ALGORITHMS.keys():
            for run_idx in range(args.runs):
                seed = base_seed + run_idx * 100 + hash(algo) % 1000
                tasks.append((algo, func_id, func_class, bias, run_idx, args.max_evals, seed))

    print(f"\n[+] Iniciando {len(tasks)} experimentos distribuidos en {args.workers} workers...")
    t0 = time.perf_counter()

    raw_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for idx, result in enumerate(executor.map(run_single_experiment, tasks), 1):
            raw_results.append(result)
            if idx % 10 == 0 or idx == len(tasks):
                elapsed = time.perf_counter() - t0
                print(f"  [{idx}/{len(tasks)}] Progreso: {idx/len(tasks)*100:.1f}% | Tiempo transcurrido: {elapsed:.1f}s")

    consolidated = consolidate_benchmark_data(raw_results, num_runs=args.runs, max_evals=args.max_evals)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)

    print(f"\n[+] Resultados guardados en: {args.output}")
    print("\n--- RANKING PROMEDIO DE FRIEDMAN (1 a 7, menor es mejor) ---")
    for algo, rank in sorted(consolidated["friedman_ranks"].items(), key=lambda x: x[1]):
        print(f"  {algo:18s}: {rank:.2f}")


if __name__ == "__main__":
    main()
