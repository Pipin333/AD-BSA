"""
================================================================================
  ELITE TRIAD BENCHMARK IEEE CEC 2020 (50D, 30 RUNS)
  Duelo de Titanes entre:
    1. AD-BSA-Csc (Nueva propuesta con repulsión cosecante acotada)
    2. jSO (Ganador absoluto de IEEE CEC 2017 - Janez Brest)
    3. CMA-ES (Estándar de oro mundial de Hansen)
================================================================================
"""

import os
import sys
import time
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(r"C:\Users\Petiso\Documents\ideas millonarias")
import elite_worker

def main():
    print("=" * 95)
    print("  INICIANDO DUELO ÉLITE IEEE CEC 2020 (50D, 30 CORRIDAS): AD-BSA-Csc vs jSO vs CMA-ES")
    print("  Evaluación oficial: 10 Funciones (F1-F10) | 50,000 NFE por corrida | 6 Cores")
    print("=" * 95)

    algos = ["AD-BSA-Csc", "jSO", "CMA-ES"]
    funcs = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]
    n_runs = 30
    dim = 50
    max_nfe = 50000
    workers = 6

    json_path = r"C:\Users\Petiso\Documents\ideas millonarias\papers\cec2020_elite_triad_results.json"
    plot_path = r"C:\Users\Petiso\Documents\ideas millonarias\papers\cec2020_elite_triad_comparison.png"

    # Cargar progreso previo si existe
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                loaded = json.load(jf)
                # Si es formato final con "raw_runs", extraerlo
                if "raw_runs" in loaded:
                    results = loaded["raw_runs"]
                else:
                    results = loaded
        except Exception:
            results = {f: {a: [] for a in algos} for f in funcs}
    else:
        results = {f: {a: [] for a in algos} for f in funcs}

    # Asegurar estructura
    for f in funcs:
        if f not in results:
            results[f] = {}
        for a in algos:
            if a not in results[f]:
                results[f][a] = []

    tasks = []
    for f in funcs:
        for a in algos:
            existing_count = len(results[f][a])
            for r in range(existing_count, n_runs):
                seed = 2000 * (r + 1) + int(f.replace("F", ""))
                tasks.append((f, a, r, seed, dim, max_nfe))

    total_tasks = 900
    start_total = time.time()

    while True:
        tasks = []
        for f in funcs:
            for a in algos:
                existing_count = len(results[f][a])
                for r in range(existing_count, n_runs):
                    seed = 2000 * (r + 1) + int(f.replace("F", ""))
                    tasks.append((f, a, r, seed, dim, max_nfe))

        pending_tasks = len(tasks)
        completed = total_tasks - pending_tasks
        if pending_tasks == 0:
            break

        print(f"Progreso actual: {completed}/{total_tasks} ({completed/total_tasks*100:5.1f}%). Pendientes: {pending_tasks} corridas...")

        try:
            with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as executor:
                futures = {executor.submit(elite_worker.run_single_elite_task, t): t for t in tasks}
                
                for future in as_completed(futures):
                    res = future.result()
                    f_name = res["func"]
                    a_name = res["algo"]
                    results[f_name][a_name].append(res["fitness"])
                    
                    completed += 1
                    if completed % 5 == 0 or completed == total_tasks:
                        elapsed = time.time() - start_total
                        rate = completed / max(1e-4, elapsed)
                        remaining = (total_tasks - completed) / max(1e-4, rate)
                        print(f"  [Progreso: {completed:3d}/{total_tasks} ({completed/total_tasks*100:5.1f}%)] "
                              f"Velocidad: {rate:4.1f} runs/s | Restante estimado: {remaining/60:4.1f}m")
                        
                        with open(json_path, "w", encoding="utf-8") as jf:
                            json.dump(results, jf, indent=2)
        except Exception as err:
            print(f"Aviso: reconectando pool de procesos ({err}). Reanudando automáticamente...")
            time.sleep(1)

    total_time = time.time() - start_total
    print("\n" + "=" * 95)
    print(f"  DUELO ÉLITE COMPLETADO EN {total_time/60:.2f} MINUTOS")
    print("=" * 95)

    # 1. ANÁLISIS ESTADÍSTICO DETALLADO
    print("\n" + "=" * 125)
    print(f"  {'Función':<10} | {'Algoritmo':<15} | {'Media':<15} | {'Std':<15} | {'Mediana':<15} | {'IQR':<15}")
    print("=" * 125)

    summary_stats = {}
    for f in funcs:
        summary_stats[f] = {}
        for a in algos:
            arr = np.array(results[f][a])
            mean_v = float(np.mean(arr))
            std_v = float(np.std(arr))
            med_v = float(np.median(arr))
            iqr_v = float(stats.iqr(arr))
            summary_stats[f][a] = {
                "mean": mean_v, "std": std_v, "median": med_v, "iqr": iqr_v,
                "min": float(np.min(arr)), "max": float(np.max(arr))
            }
            print(f"  {f:<10} | {a:<15} | {mean_v:14.4e} | {std_v:14.4e} | {med_v:14.4e} | {iqr_v:14.4e}")
        print("-" * 125)

    # 2. PRUEBA NO PARAMÉTRICA DE WILCOXON (AD-BSA-Csc vs jSO y vs CMA-ES)
    print("\n" + "=" * 95)
    print("  TESTS DE WILCOXON PAREADOS (p-values) DE AD-BSA-Csc FRENTE A jSO Y CMA-ES")
    print("=" * 95)
    print(f"  {'Función':<10} | {'vs jSO':<25} | {'vs CMA-ES':<25}")
    print("-" * 95)

    wilcoxon_results = {}
    for f in funcs:
        csc_arr = np.array(results[f]["AD-BSA-Csc"])
        wilcoxon_results[f] = {}
        row_str = f"  {f:<10} | "
        for a in ["jSO", "CMA-ES"]:
            other_arr = np.array(results[f][a])
            if np.allclose(csc_arr, other_arr):
                p_val = 1.0
            else:
                try:
                    _, p_val = stats.wilcoxon(csc_arr, other_arr)
                except Exception:
                    p_val = 1.0
            wilcoxon_results[f][a] = float(p_val)
            p_fmt = f"{p_val:12.4e}" if p_val < 0.05 else f"{p_val:12.4f} (Tie)"
            row_str += f"{p_fmt:<25} | "
        print(row_str)
    print("=" * 95)

    # 3. RANKING DE FRIEDMAN OFICIAL
    ranks = np.zeros((len(funcs), len(algos)))
    for i, f in enumerate(funcs):
        means = [summary_stats[f][a]["mean"] for a in algos]
        ranks[i] = stats.rankdata(means)

    mean_ranks = np.mean(ranks, axis=0)
    print("\n" + "=" * 80)
    print("  RANKING PROMEDIO OFICIAL DE FRIEDMAN: DUELO ÉLITE (10 FUNCIONES, 50D, 30 RUNS)")
    print("=" * 80)
    sorted_rank_idx = np.argsort(mean_ranks)
    for pos, idx in enumerate(sorted_rank_idx):
        print(f"  #{pos+1:<2} : {algos[idx]:<18} -> Rango Friedman Promedio: {mean_ranks[idx]:.2f}")
    print("=" * 80)

    final_output = {
        "metadata": {
            "suite": "IEEE CEC 2020 Elite Triad",
            "dimension": dim,
            "max_evaluations": max_nfe,
            "num_runs": n_runs,
            "total_execution_time_seconds": total_time,
            "algorithms": algos
        },
        "friedman_ranks": {algos[i]: float(mean_ranks[i]) for i in range(len(algos))},
        "summary_statistics": summary_stats,
        "wilcoxon_tests": wilcoxon_results,
        "raw_runs": results
    }

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(final_output, jf, indent=2)
    print(f"\nResultados consolidados guardados en:\n{json_path}")

    # 4. FIGURA DE BOXPLOTS
    print("\nGenerando figura de boxplots de alta resolución...")
    plt.figure(figsize=(16, 12), dpi=300)
    plt.suptitle("IEEE CEC 2020 Elite Triad Benchmark (50 Dimensions, 30 Runs)\nAD-BSA-Csc vs. jSO (CEC 2017 Winner) vs. CMA-ES (Gold Standard)",
                 fontsize=15, fontweight="bold", y=0.98)

    sample_funcs = ["F1", "F3", "F4", "F8"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    for sp_idx, sf in enumerate(sample_funcs):
        ax = plt.subplot(2, 2, sp_idx + 1)
        data = [results[sf][a] for a in algos]
        
        box = ax.boxplot(data, tick_labels=algos, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            
        ax.set_yscale('log')
        ax.set_title(f"{sf}: {elite_worker.CEC_FUNCS[sf][1]} (50D)", fontweight="bold", fontsize=11)
        ax.set_ylabel("Fitness (Log Scale)")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_path)
    plt.close()
    print(f"Figura guardada en:\n{plot_path}")

if __name__ == "__main__":
    main()
