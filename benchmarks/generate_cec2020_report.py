"""
================================================================================
  AD-BSA Report Generator: Consolidate IEEE CEC 2020 (50D) Results & Visuals
  Produces:
    - benchmarks/cec2020_50d_report.md
    - benchmarks/cec2020_50d_comparison.png
================================================================================
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "cec2020_50d_results.json")
OUTPUT_MD = os.path.join(os.path.dirname(__file__), "cec2020_50d_report.md")
OUTPUT_PNG = os.path.join(os.path.dirname(__file__), "cec2020_50d_comparison.png")

CEC2020_META = [
    ("F1", "Shifted & Rotated Bent Cigar", "Unimodal", 100.0),
    ("F2", "Shifted & Rotated Schwefel", "Multimodal", 1100.0),
    ("F3", "Shifted & Rotated Lunacek bi-Rastrigin", "Multimodal", 700.0),
    ("F4", "Expanded Rosenbrock + Griewank", "Multimodal", 1900.0),
    ("F5", "Hybrid Function 1 (N=3)", "Hybrid", 1700.0),
    ("F6", "Hybrid Function 2 (N=4)", "Hybrid", 1600.0),
    ("F7", "Hybrid Function 3 (N=5)", "Hybrid", 2100.0),
    ("F8", "Composition Function 1 (N=3)", "Composition", 2200.0),
    ("F9", "Composition Function 2 (N=4)", "Composition", 2400.0),
    ("F10", "Composition Function 3 (N=5)", "Composition", 2500.0),
]

ALGOS = ["AD-BSA", "jSO", "CMA-ES", "L-SHADE", "Standard-DE", "Standard-PSO", "Cuckoo-Search"]


def generate_markdown_report(data: dict):
    n_runs = data.get("metadata", {}).get("num_runs", 10)
    lines = []
    lines.append(f"# IEEE CEC 2020 Benchmark Results (50 Dimensions, {n_runs} Runs)\n")
    lines.append(f"Comparativa exhaustiva y estadísticamente rigurosa sobre las 10 funciones canónicas de la **IEEE CEC 2020** en **$D = 50$**, con un presupuesto de $50,000$ evaluaciones por corrida y ${n_runs}$ ejecuciones independientes con semillas pseudoaleatorias controladas.\n")
    lines.append("### Ranking Promedio de Friedman (Menor es Mejor):\n")
    lines.append("| Posición | Algoritmo | Rango Friedman | Tipo / Linaje |")
    lines.append("| :---: | :--- | :---: | :--- |")

    sorted_ranks = sorted(data["friedman_ranks"].items(), key=lambda x: x[1])
    descriptions = {
        "AD-BSA": r"**Propuesto: Repulsión Cosecante Acotada $|\csc(x)|$ + Anti-Atractores**",
        "CMA-ES": "Hansen Covariance Matrix Adaptation con IPOP Restarts",
        "jSO": "Campeón IEEE CEC 2017 (Extensión de iL-SHADE)",
        "L-SHADE": "Campeón IEEE CEC 2014 (DE Adaptativa con LPSR)",
        "Standard-PSO": "Particle Swarm Optimization canónico con decaimiento de inercia",
        "Standard-DE": "Differential Evolution clásica DE/rand/1/bin",
        "Cuckoo-Search": "Cuckoo Search canónico con Vuelos de Lévy",
    }
    for pos, (algo, rank) in enumerate(sorted_ranks, 1):
        lines.append(f"| **#{pos}** | `{algo}` | **{rank:.2f}** | {descriptions.get(algo, '')} |")

    lines.append("\n---\n")
    lines.append(r"### Tabla Detallada de Rendimiento: Media ± Desv. Estándar (Error $\Delta f = f(x) - f^*$) y Test de Wilcoxon:" + "\n")
    lines.append(r"Signos de Wilcoxon vs `AD-BSA`: `+` (AD-BSA supera con $p < 0.05$), `-` (AD-BSA es superado con $p < 0.05$), `=` (empate estadístico)." + "\n")

    # Header
    header = "| Función | Categoría | `AD-BSA` (Propuesto) | `jSO` | `CMA-ES` | `L-SHADE` | `Standard-DE` | `Standard-PSO` | `Cuckoo-Search` |"
    lines.append(header)
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for fid, name, cat, _ in CEC2020_META:
        st = data["summary_statistics"][fid]["algorithms"]
        wt = data["wilcoxon_tests_vs_ad_bsa"][fid]

        def fmt(a):
            m = st[a]["mean_error"]
            s = st[a]["std_error"]
            w_sign = f" ({wt[a]['sign']})" if a != "AD-BSA" else ""
            if m < 1e-4:
                return f"**0.00e+00**" + w_sign if a == "AD-BSA" else "0.00e+00" + w_sign
            elif m >= 1e4:
                return f"{m:.2e} ± {s:.1e}" + w_sign
            else:
                return f"{m:.2f} ± {s:.1f}" + w_sign

        row = f"| **{fid}** ({name[:16]}...) | {cat} | **{fmt('AD-BSA')}** | {fmt('jSO')} | {fmt('CMA-ES')} | {fmt('L-SHADE')} | {fmt('Standard-DE')} | {fmt('Standard-PSO')} | {fmt('Cuckoo-Search')} |"
        lines.append(row)

    md_content = "\n".join(lines) + "\n"
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    return md_content


def generate_comparison_plot(data: dict):
    funcs = [f[0] for f in CEC2020_META]
    algos = ALGOS
    colors = ["#2563EB", "#059669", "#7C3AED", "#D97706", "#DC2626", "#4B5563", "#9333EA"]
    n_runs = data.get("metadata", {}).get("num_runs", 10)

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    width = 0.11
    x = np.arange(len(funcs))

    for i, (algo, color) in enumerate(zip(algos, colors)):
        errors = [max(1e-8, data["summary_statistics"][f]["algorithms"][algo]["mean_error"]) for f in funcs]
        log_errors = np.log10(errors)
        offset = (i - len(algos) / 2) * width + width / 2
        ax.bar(x + offset, log_errors, width, label=algo, color=color, alpha=0.9, edgecolor="black", linewidth=0.5)

    ax.set_title(f"IEEE CEC 2020 Benchmark (50 Dimensions, {n_runs} Runs) - Log10 Mean Error", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Funciones de Prueba CEC 2020", fontsize=12, fontweight="bold")
    ax.set_ylabel("Log10(Error Delta f) [Menor es Mejor]", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(funcs, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG)
    plt.close()


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_JSON):
        print(f"[!] No se encontró {OUTPUT_JSON}. Ejecuta primero 'python benchmarks/run_cec2020_50d.py'.")
        sys.exit(1)

    print(f"[+] Cargando resultados oficiales desde: {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    md = generate_markdown_report(data)
    generate_comparison_plot(data)
    print(f"[OK] Reporte Markdown actualizado en: {OUTPUT_MD}")
    print(f"[OK] Gráfico comparativo actualizado en: {OUTPUT_PNG}")
