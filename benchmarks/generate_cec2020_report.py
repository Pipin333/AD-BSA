"""
================================================================================
  AD-BSA Report Generator: Consolidate IEEE CEC 2020 (50D) Results & Visuals
  Produces:
    - benchmarks/cec2020_50d_results.json
    - benchmarks/cec2020_50d_report.md
    - benchmarks/cec2020_50d_comparison.png
================================================================================
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SOURCE_PATH_1 = r"C:\Users\Petiso\Documents\ideas millonarias\AD-BSA\data\cec2020_30runs_6algos_results.json"
SOURCE_PATH_2 = r"C:\Users\Petiso\Documents\ideas millonarias\AD-BSA\data\cec2020_elite_triad_results.json"

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


def generate_consolidated_data():
    with open(SOURCE_PATH_1, "r", encoding="utf-8") as f1, open(SOURCE_PATH_2, "r", encoding="utf-8") as f2:
        d1 = json.load(f1)
        d2 = json.load(f2)

    jso_fixed_path = os.path.join(os.path.dirname(__file__), "jso_fixed_30runs.json")
    jso_fixed_data = None
    if os.path.exists(jso_fixed_path):
        with open(jso_fixed_path, "r", encoding="utf-8") as jf:
            jso_fixed_data = json.load(jf)
        print("[+] Cargados datos de jSO corregido desde jso_fixed_30runs.json")

    biases = {f[0]: f[3] for f in CEC2020_META}
    summary = {}
    wilcoxon_tests = {}
    consolidated_raw_runs = {fid: {} for fid, _, _, _ in CEC2020_META}

    # Extraer errores directos por algoritmo y función
    for fid, name, cat, bias in CEC2020_META:
        summary[fid] = {
            "name": name,
            "category": cat,
            "bias": bias,
            "algorithms": {}
        }
        wilcoxon_tests[fid] = {}

        # Mapeo de nombres desde los archivos fuentes
        algo_data_map = {
            "AD-BSA": d1["summary_statistics"][fid]["AD-BSA-Csc"],
            "L-SHADE": d1["summary_statistics"][fid]["L-SHADE"],
            "Standard-DE": d1["summary_statistics"][fid]["Standard-DE"],
            "Standard-PSO": d1["summary_statistics"][fid]["Canonical-PSO"],
            "Cuckoo-Search": d1["summary_statistics"][fid]["Cuckoo-Search"],
            "jSO": d2["summary_statistics"][fid]["jSO"],
            "CMA-ES": d2["summary_statistics"][fid]["CMA-ES"],
        }

        # Sobrescribir con jSO corregido si existe
        if jso_fixed_data and fid in jso_fixed_data and len(jso_fixed_data[fid]) >= 30:
            jso_runs = jso_fixed_data[fid]
            jso_fits = np.array([r["fitness"] for r in jso_runs], dtype=np.float64)
            jso_errs = np.array([r["error"] for r in jso_runs], dtype=np.float64)
            algo_data_map["jSO"] = {
                "mean": float(np.mean(jso_fits)),
                "std": float(np.std(jso_errs)),
                "median": float(np.median(jso_fits)),
                "min": float(np.min(jso_fits)),
            }

        # Extraer raw runs para calcular Wilcoxon
        raw_ad = [max(0.0, float(fit) - bias) for fit in d1["raw_runs"][fid]["AD-BSA-Csc"]]
        consolidated_raw_runs[fid]["AD-BSA"] = raw_ad

        for a in ALGOS:
            st = algo_data_map[a]
            raw_mean = st["mean"]
            err_mean = max(0.0, raw_mean - bias)
            err_median = max(0.0, st["median"] - bias)
            err_best = max(0.0, st.get("min", raw_mean) - bias)

            summary[fid]["algorithms"][a] = {
                "mean_error": err_mean,
                "std_error": st["std"],
                "median_error": err_median,
                "best_error": err_best,
            }

            if a != "AD-BSA":
                if a == "jSO" and jso_fixed_data and fid in jso_fixed_data and len(jso_fixed_data[fid]) >= 30:
                    raw_other = [float(r["error"]) for r in jso_fixed_data[fid]]
                elif a in ["L-SHADE", "Standard-DE", "Standard-PSO", "Cuckoo-Search"]:
                    raw_other_key = "Canonical-PSO" if a == "Standard-PSO" else a
                    raw_other = [max(0.0, float(fit) - bias) for fit in d1["raw_runs"][fid][raw_other_key]]
                else:
                    raw_other = [max(0.0, float(fit) - bias) for fit in d2["raw_runs"][fid][a]]

                consolidated_raw_runs[fid][a] = raw_other

                if len(raw_ad) > 0 and len(raw_other) > 0:
                    try:
                        stat_val, p_val = stats.wilcoxon(raw_ad, raw_other)
                    except Exception:
                        stat_val, p_val = stats.mannwhitneyu(raw_ad, raw_other)

                    med_ad = np.median(raw_ad)
                    med_ot = np.median(raw_other)
                    if p_val < 0.05:
                        sign = "+" if med_ad < med_ot else "-"
                    else:
                        sign = "="
                    wilcoxon_tests[fid][a] = {"p_value": float(p_val), "sign": sign}
                else:
                    wilcoxon_tests[fid][a] = {"p_value": 1.0, "sign": "="}


    # Calcular Ranks de Friedman
    ranks = {a: [] for a in ALGOS}
    for fid, _, _, _ in CEC2020_META:
        means = {a: summary[fid]["algorithms"][a]["mean_error"] for a in ALGOS}
        sorted_algos = sorted(ALGOS, key=lambda x: means[x])
        for r, a in enumerate(sorted_algos, 1):
            ranks[a].append(r)

    friedman_ranks = {a: float(np.mean(ranks[a])) for a in ALGOS}

    consolidated = {
        "metadata": {
            "benchmark_suite": "IEEE CEC 2020",
            "dimension": 50,
            "bounds": [-100.0, 100.0],
            "max_evaluations": 50000,
            "num_runs": 30,
            "algorithms": ALGOS,
            "functions": [f[0] for f in CEC2020_META],
        },
        "friedman_ranks": friedman_ranks,
        "summary_statistics": summary,
        "wilcoxon_tests_vs_ad_bsa": wilcoxon_tests,
        "raw_runs": consolidated_raw_runs,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)

    return consolidated


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
        "L-SHADE": "Campeón IEEE CEC 2014 (DE Adaptativa con LPSR)",
        "AD-BSA": r"**Propuesto: Repulsión Cosecante Acotada $|\csc(x)|$ + Anti-Atractores**",
        "CMA-ES": "Hansen Covariance Matrix Adaptation con IPOP Restarts",
        "jSO": "Campeón IEEE CEC 2017 (Extensión de iL-SHADE)",
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
    print("[+] Generando datos consolidados y reporte CEC 2020 (50D)...")
    data = generate_consolidated_data()
    md = generate_markdown_report(data)
    generate_comparison_plot(data)
    print(f"[OK] Reporte generado en: {OUTPUT_MD}")
    print(f"[OK] Grafico generado en: {OUTPUT_PNG}")
    print(f"[OK] Datos JSON en: {OUTPUT_JSON}")

