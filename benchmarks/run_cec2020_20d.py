"""
================================================================================
  AD-BSA Benchmark Suite: Official IEEE CEC 2020 (20 Dimensions) Runner
  Evaluates 10 Algorithms across all 10 CEC 2020 Benchmark Functions in 20D:
    1. AD-BSA (Proposed Algorithm with Bounded Cosecant Repulsion |csc|)
    2. CMA-ES (Hansen Covariance Matrix Adaptation with IPOP Restarts)
    3. L-SHADE (IEEE CEC 2014 Winner, Tanabe & Fukunaga)
    4. NL-SHADE-RSP (IEEE CEC 2021 Winner, Stanovov et al.)
    5. NL-SHADE-LBC (IEEE CEC 2022 Top Performer, Stanovov et al.)
    6. MadDE (IEEE CEC 2021 Multi-Adaptation DE, Biswas et al.)
    7. Standard PSO (Global Best PSO)
    8. Standard DE (DE/rand/1/bin)
    9. Canonical Cuckoo Search (CS)
   10. jSO (IEEE CEC 2017 Winner, Brest et al.)
================================================================================
"""

import os
import sys

# Prevent OpenBLAS/MKL/OMP thread explosion across workers on multi-core systems
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from opfunu.cec_based import cec2020

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from ad_bsa import (
    AD_BSA,
    jSO,
    CMA_ES,
    L_SHADE,
    NL_SHADE_RSP,
    NL_SHADE_LBC,
    MadDE,
    CanonicalCuckooSearch,
    StandardDE,
    StandardPSO,
    EvaluatorWrapper,
    compute_wilcoxon,
)

ALGORITHMS = {
    "AD-BSA": AD_BSA,
    "CMA-ES": CMA_ES,
    "L-SHADE": L_SHADE,
    "NL-SHADE-RSP": NL_SHADE_RSP,
    "NL-SHADE-LBC": NL_SHADE_LBC,
    "MadDE": MadDE,
    "Standard-PSO": StandardPSO,
    "Standard-DE": StandardDE,
    "Cuckoo-Search": CanonicalCuckooSearch,
    "jSO": jSO,
}

CEC2020_FUNCS = [
    ("F1", "F12020", "Shifted and Rotated Bent Cigar", "Unimodal", 100.0),
    ("F2", "F22020", "Shifted and Rotated Schwefel", "Multimodal", 1100.0),
    ("F3", "F32020", "Shifted and Rotated Lunacek bi-Rastrigin", "Multimodal", 700.0),
    ("F4", "F42020", "Expanded Rosenbrock + Griewank", "Multimodal", 1900.0),
    ("F5", "F52020", "Hybrid Function 1", "Hybrid", 1700.0),
    ("F6", "F62020", "Hybrid Function 2", "Hybrid", 1600.0),
    ("F7", "F72020", "Hybrid Function 3", "Hybrid", 2100.0),
    ("F8", "F82020", "Composition Function 1", "Composition", 2200.0),
    ("F9", "F92020", "Composition Function 2", "Composition", 2400.0),
    ("F10", "F102020", "Composition Function 3", "Composition", 2500.0),
]


def run_single_experiment(args_tuple):
    algo_name, func_id, func_class_name, f_bias, run_idx, max_evals, seed = args_tuple
    f_class = getattr(cec2020, func_class_name)
    inst = f_class(ndim=20)

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
    funcs = [f[0] for f in CEC2020_FUNCS]
    meta_dict = {f[0]: f for f in CEC2020_FUNCS}
    algos = list(ALGORITHMS.keys())

    by_func = {f: {a: [] for a in algos} for f in funcs}
    raw_by_func = {f: {a: [] for a in algos} for f in funcs}
    for r in raw_runs:
        if r["func"] in by_func and r["algo"] in by_func[r["func"]]:
            by_func[r["func"]][r["algo"]].append(r["error"])
            raw_by_func[r["func"]][r["algo"]].append(r["error"])

    summary = {}
    wilcoxon_results = {}

    for f in funcs:
        f_id, f_class, f_desc, f_cat, f_bias = meta_dict[f]
        summary[f] = {
            "name": f_desc,
            "category": f_cat,
            "bias": f_bias,
            "algorithms": {}
        }
        wilcoxon_results[f] = {}
        ad_bsa_scores = by_func[f]["AD-BSA"]

        for a in algos:
            scores = np.asarray(by_func[f][a], dtype=np.float64)
            summary[f]["algorithms"][a] = {
                "mean_error": float(np.mean(scores)) if len(scores) > 0 else 0.0,
                "std_error": float(np.std(scores)) if len(scores) > 0 else 0.0,
                "median_error": float(np.median(scores)) if len(scores) > 0 else 0.0,
                "best_error": float(np.min(scores)) if len(scores) > 0 else 0.0,
                "worst_error": float(np.max(scores)) if len(scores) > 0 else 0.0,
            }
            if a != "AD-BSA" and len(scores) > 0 and len(ad_bsa_scores) > 0:
                pval, sign = compute_wilcoxon(ad_bsa_scores, scores)
                wilcoxon_results[f][a] = {"p_value": pval, "sign": sign}

    # Rankings de Friedman
    ranks = {a: [] for a in algos}
    for f in funcs:
        means = {a: summary[f]["algorithms"][a]["mean_error"] for a in algos}
        sorted_algos = sorted(algos, key=lambda x: means[x])
        for rank, a in enumerate(sorted_algos, 1):
            ranks[a].append(rank)

    friedman_ranks = {a: float(np.mean(ranks[a])) for a in algos}

    return {
        "metadata": {
            "benchmark_suite": "IEEE CEC 2020",
            "dimension": 20,
            "bounds": [-100.0, 100.0],
            "max_evaluations": max_evals,
            "num_runs": num_runs,
            "algorithms": algos,
            "functions": funcs,
        },
        "friedman_ranks": friedman_ranks,
        "summary_statistics": summary,
        "wilcoxon_tests_vs_ad_bsa": wilcoxon_results,
        "raw_runs": raw_by_func,
    }


def generate_report_and_plot(data: dict, output_md: str, output_png: str):
    n_runs = data.get("metadata", {}).get("num_runs", 5)
    dim = data.get("metadata", {}).get("dimension", 20)
    max_evals = data.get("metadata", {}).get("max_evaluations", 200000)

    lines = []
    lines.append(f"# IEEE CEC 2020 Benchmark Results ({dim} Dimensions, {n_runs} Runs)\n")
    lines.append(f"Evaluación comparativa rigurosa sobre las 10 funciones canónicas de la **IEEE CEC 2020** en **$D = {dim}$**, bajo el estándar oficial de $\\text{{MaxNFE}} = 10,000 \\cdot D = {max_evals:,}$ evaluaciones por corrida y ${n_runs}$ ejecuciones independientes controladas con semillas pseudoaleatorias.\n")
    lines.append("### Ranking Promedio de Friedman (Menor es Mejor):\n")
    lines.append("| Posición | Algoritmo | Rango Friedman | Tipo / Linaje |")
    lines.append("| :---: | :--- | :---: | :--- |")

    sorted_ranks = sorted(data["friedman_ranks"].items(), key=lambda x: x[1])
    descriptions = {
        "AD-BSA": r"**Propuesto: Repulsión Cosecante Acotada $|\csc(x)|$ + Anti-Atractores**",
        "NL-SHADE-LBC": "Top IEEE CEC 2022 (NLPSR + Generalized Lehmer Linear Bias Change, Stanovov et al.)",
        "NL-SHADE-RSP": "Campeón IEEE CEC 2021 (NLPSR + Rank Selective Pressure + Hybrid Crossover, Stanovov et al.)",
        "MadDE": "Top IEEE CEC 2021 (Multi-Adaptation DE + QBX + Sobol QMC, Biswas et al.)",
        "CMA-ES": "Hansen Covariance Matrix Adaptation con IPOP Restarts",
        "jSO": "Campeón IEEE CEC 2017 (Extensión de iL-SHADE, Brest et al.)",
        "L-SHADE": "Campeón IEEE CEC 2014 (DE Adaptativa con LPSR, Tanabe & Fukunaga)",
        "Standard-PSO": "Particle Swarm Optimization canónico con decaimiento de inercia",
        "Standard-DE": "Differential Evolution clásica DE/rand/1/bin",
        "Cuckoo-Search": "Cuckoo Search canónico con Vuelos de Lévy",
    }
    for pos, (algo, rank) in enumerate(sorted_ranks, 1):
        lines.append(f"| **#{pos}** | `{algo}` | **{rank:.2f}** | {descriptions.get(algo, '')} |")

    lines.append("\n---\n")
    lines.append(r"### Tabla Detallada de Rendimiento: Media ± Desv. Estándar (Error $\Delta f = f(x) - f^*$) y Test de Wilcoxon:" + "\n")
    lines.append(r"Signos de Wilcoxon vs `AD-BSA`: `+` (AD-BSA supera significativamente al competidor, $p < 0.05$), `=` (estadísticamente equivalente, $p \ge 0.05$), `-` (competidor supera significativamente a AD-BSA, $p < 0.05$)." + "\n")

    algos_order = [
        "AD-BSA", "NL-SHADE-LBC", "NL-SHADE-RSP", "MadDE", "jSO", "CMA-ES", "L-SHADE", "Standard-DE", "Standard-PSO", "Cuckoo-Search"
    ]
    present_algos = [a for a in algos_order if a in data["summary_statistics"]["F1"]["algorithms"]]

    cols_header = " | ".join([f"`{a}` (Propuesto)" if a == "AD-BSA" else f"`{a}`" for a in present_algos])
    lines.append(f"| Función | Categoría | {cols_header} |")
    lines.append("| :--- | :--- | " + " | ".join([":---:" for _ in present_algos]) + " |")

    for fid, _, name, cat, _ in CEC2020_FUNCS:
        st = data["summary_statistics"][fid]["algorithms"]
        wt = data["wilcoxon_tests_vs_ad_bsa"].get(fid, {})

        def fmt(a):
            m = st[a]["mean_error"]
            s = st[a]["std_error"]
            w_sign = f" ({wt[a]['sign']})" if a != "AD-BSA" and a in wt else ""
            if m < 1e-4:
                return "0.00e+00" + w_sign
            elif m >= 1e4:
                return f"{m:.2e} ± {s:.1e}" + w_sign
            else:
                return f"{m:.2f} ± {s:.1f}" + w_sign

        cells = [f"**{fmt(a)}**" if a == "AD-BSA" else fmt(a) for a in present_algos]
        lines.append(f"| **{fid}** ({name[:16]}...) | {cat} | " + " | ".join(cells) + " |")

    lines.append("\n---\n")
    lines.append(f"### Resumen Estadístico Global de Wilcoxon (`AD-BSA` vs Competidores en $D={dim}$):\n")
    lines.append("| Competidor | Año / Referencia | `AD-BSA` Mejor (`+`) | Empate Estadístico (`=`) | Competidor Mejor (`-`) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: |")
    for a in present_algos:
        if a == "AD-BSA":
            continue
        signs = [data["wilcoxon_tests_vs_ad_bsa"].get(fid, {}).get(a, {}).get("sign", "=") for fid, _, _, _, _ in CEC2020_FUNCS]
        plus_c = signs.count("+")
        eq_c = signs.count("=")
        minus_c = signs.count("-")
        lines.append(f"| `{a}` | {descriptions.get(a, '')} | **{plus_c}** | {eq_c} | {minus_c} |")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Gráfico
    funcs = [f[0] for f in CEC2020_FUNCS]
    colors = [
        "#2563EB", "#0D9488", "#0891B2", "#4F46E5", "#059669",
        "#7C3AED", "#D97706", "#DC2626", "#4B5563", "#9333EA"
    ]
    fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300)
    width = 0.08
    x = np.arange(len(funcs))

    for i, (algo, color) in enumerate(zip(present_algos, colors[:len(present_algos)])):
        errors = [max(1e-8, data["summary_statistics"][f]["algorithms"][algo]["mean_error"]) for f in funcs]
        log_errors = np.log10(errors)
        offset = (i - len(present_algos) / 2) * width + width / 2
        ax.bar(x + offset, log_errors, width, label=algo, color=color, alpha=0.9, edgecolor="black", linewidth=0.4)

    ax.set_title(f"IEEE CEC 2020 Benchmark (20 Dimensions, {n_runs} Runs) - Log10 Mean Error across 10 Algorithms", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Funciones de Prueba CEC 2020 (20D)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Log10(Error Delta f) [Menor es Mejor]", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(funcs, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Ejecutar Benchmark CEC 2020 en 20 Dimensiones")
    parser.add_argument("--runs", type=int, default=5, help="Número de corridas independientes (def: 5)")
    parser.add_argument("--max-evals", type=int, default=200000, help="Presupuesto MaxNFE (10000*D = 200000)")
    parser.add_argument("--workers", type=int, default=7, help="Procesos paralelos (def: 7)")
    parser.add_argument("--output", type=str, default="benchmarks/cec2020_20d_results.json", help="Ruta de guardado JSON")
    args = parser.parse_args()

    algo_list = list(ALGORITHMS.keys())

    print("=" * 85, flush=True)
    print(f"  IEEE CEC 2020 BENCHMARK SUITE - 20 DIMENSIONS", flush=True)
    print(f"  Algoritmos: {algo_list}", flush=True)
    print(f"  Configuración: {args.runs} corridas x {len(CEC2020_FUNCS)} funciones | MaxNFE: {args.max_evals:,} (10,000 * D)", flush=True)
    print(f"  Workers paralelos: {args.workers}", flush=True)
    print("=" * 85, flush=True)

    checkpoint_file = os.path.join(os.path.dirname(args.output), "cec2020_20d_checkpoint.json")
    completed_runs = []
    completed_keys = set()

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as cf:
                loaded_cp = json.load(cf)
                for r in loaded_cp:
                    if r.get("evaluations", args.max_evals) == args.max_evals and r["run"] < args.runs and (r["algo"], r["func"], r["run"]) not in completed_keys:
                        completed_runs.append(r)
                        completed_keys.add((r["algo"], r["func"], r["run"]))
            if len(completed_keys) > 0:
                print(f"[+] Reanudando desde checkpoint: {len(completed_keys)} corridas ya completadas.", flush=True)
        except Exception as e:
            print(f"[!] Error leyendo checkpoint 20D ({e}), iniciando desde cero.", flush=True)

    tasks = []
    for func_id, func_class, desc, cat, bias in CEC2020_FUNCS:
        func_num = int(func_id.replace("F", ""))
        for algo in algo_list:
            algo_idx = algo_list.index(algo)
            for run_idx in range(args.runs):
                if (algo, func_id, run_idx) in completed_keys:
                    continue
                seed = 20000 * (run_idx + 1) + 100 * func_num + algo_idx
                tasks.append((algo, func_id, func_class, bias, run_idx, args.max_evals, seed))

    total_tasks = len(CEC2020_FUNCS) * len(algo_list) * args.runs
    pending = len(tasks)
    print(f"\n[+] Total requerido: {total_tasks} experimentos. Completados: {len(completed_keys)}. Pendientes: {pending}.\n", flush=True)

    if pending > 0:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_single_experiment, t): t for t in tasks}
            done_count = len(completed_keys)

            for future in as_completed(futures):
                res = future.result()
                completed_runs.append(res)
                done_count += 1

                if done_count % 5 == 0 or done_count == total_tasks:
                    elapsed = time.perf_counter() - t0
                    rate = (done_count - len(completed_keys)) / (elapsed + 1e-6)
                    eta = (total_tasks - done_count) / (rate + 1e-6)
                    print(f"  [{done_count:4d}/{total_tasks}] Progreso: {done_count/total_tasks*100:5.1f}% | "
                          f"Transcurrido: {elapsed/60:4.1f}m | ETA: {eta/60:4.1f}m | "
                          f"Ultimo: {res['algo']:14s} {res['func']} (err: {res['error']:.2e})", flush=True)

                    with open(checkpoint_file, "w", encoding="utf-8") as cf:
                        json.dump(completed_runs, cf)

    consolidated = consolidate_benchmark_data(completed_runs, num_runs=args.runs, max_evals=args.max_evals)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)

    output_md = os.path.join(os.path.dirname(args.output), "cec2020_20d_report.md")
    output_png = os.path.join(os.path.dirname(args.output), "cec2020_20d_comparison.png")
    generate_report_and_plot(consolidated, output_md, output_png)

    print(f"\n[+] Resultados consolidados guardados en: {args.output}")
    print(f"[+] Reporte Markdown guardado en: {output_md}")
    print(f"[+] Grafico guardado en: {output_png}")
    print("\n--- RANKING PROMEDIO DE FRIEDMAN EN 20D (1 a 10, menor es mejor) ---")
    for algo, rank in sorted(consolidated["friedman_ranks"].items(), key=lambda x: x[1]):
        print(f"  {algo:18s}: {rank:.2f}")


if __name__ == "__main__":
    main()
