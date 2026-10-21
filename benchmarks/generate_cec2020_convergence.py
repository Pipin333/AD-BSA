"""
================================================================================
  AD-BSA Benchmark Suite: IEEE CEC 2020 (50D) Convergence Curves Generator
  Generates publication-quality convergence curves across all 10 functions (F1-F10)
  for:
    1. AD-BSA (Proposed - Bounded Cosecant Repulsion |csc|)
    2. jSO (Canonical CEC 2017 Winner - Brest et al.)
    3. CMA-ES (Hansen IPOP Covariance Matrix Adaptation)
    4. L-SHADE (Canonical CEC 2014 Winner - Tanabe & Fukunaga)
    5. Standard-DE (DE/rand/1/bin)
    6. Standard-PSO (Shi & Eberhart Inertia Decay)
================================================================================
"""

import os
import sys
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from ad_bsa import (
    AD_BSA,
    jSO,
    CMA_ES,
    L_SHADE,
    StandardDE,
    StandardPSO,
    EvaluatorWrapper,
)
from opfunu.cec_based import cec2020

# Official IEEE CEC 2020 Metadata (50 Dimensions)
CEC2020_FUNCS = [
    ("F1", "F12020", "Shifted & Rotated Bent Cigar", "Unimodal", 100.0),
    ("F2", "F22020", "Shifted & Rotated Schwefel", "Multimodal", 1100.0),
    ("F3", "F32020", "Shifted & Rotated Lunacek bi-Rastrigin", "Multimodal", 700.0),
    ("F4", "F42020", "Expanded Rosenbrock + Griewank", "Multimodal", 1900.0),
    ("F5", "F52020", "Hybrid Function 1 (N=3)", "Hybrid", 1700.0),
    ("F6", "F62020", "Hybrid Function 2 (N=4)", "Hybrid", 1600.0),
    ("F7", "F72020", "Hybrid Function 3 (N=5)", "Hybrid", 2100.0),
    ("F8", "F82020", "Composition Function 1 (N=3)", "Composition", 2200.0),
    ("F9", "F92020", "Composition Function 2 (N=4)", "Composition", 2400.0),
    ("F10", "F102020", "Composition Function 3 (N=5)", "Composition", 2500.0),
]

ALGORITHMS = {
    "AD-BSA": {
        "cls": AD_BSA,
        "color": "#1D4ED8",  # Vibrant Deep Blue
        "label": "AD-BSA (Proposed)",
        "linestyle": "-",
        "linewidth": 2.5,
        "zorder": 10,
    },
    "jSO": {
        "cls": jSO,
        "color": "#059669",  # Emerald Green
        "label": "jSO (CEC 2017)",
        "linestyle": "-.",
        "linewidth": 1.8,
        "zorder": 8,
    },
    "CMA-ES": {
        "cls": CMA_ES,
        "color": "#7C3AED",  # Purple
        "label": "CMA-ES (Hansen)",
        "linestyle": ":",
        "linewidth": 1.8,
        "zorder": 7,
    },
    "L-SHADE": {
        "cls": L_SHADE,
        "color": "#D97706",  # Amber
        "label": "L-SHADE (CEC 2014)",
        "linestyle": "--",
        "linewidth": 1.8,
        "zorder": 6,
    },
    "Standard-DE": {
        "cls": StandardDE,
        "color": "#DC2626",  # Red
        "label": "Standard DE",
        "linestyle": ":",
        "linewidth": 1.4,
        "zorder": 4,
    },
    "Standard-PSO": {
        "cls": StandardPSO,
        "color": "#6B7280",  # Slate Gray
        "label": "Standard PSO",
        "linestyle": "-.",
        "linewidth": 1.4,
        "zorder": 3,
    },
}


def run_worker_task(args):
    """Executes a single algorithm on a specific benchmark function."""
    func_id, func_class_name, f_bias, algo_name, max_evals, seed = args
    f_class = getattr(cec2020, func_class_name)
    inst = f_class(ndim=50)
    evaluator = EvaluatorWrapper(inst.evaluate, f_bias=f_bias)
    bounds = np.column_stack([inst.lb, inst.ub])

    algo_info = ALGORITHMS[algo_name]
    algo_cls = algo_info["cls"]

    t0 = time.time()
    opt = algo_cls(evaluator, bounds=bounds, max_evaluations=max_evals, seed=seed)
    res = opt.optimize()
    elapsed = time.time() - t0

    # Extract history arrays
    raw_evals = np.array(res.history_evaluations, dtype=np.float64)
    raw_fits = np.array(res.history_best_fitness, dtype=np.float64)
    raw_errors = np.maximum(1e-16, raw_fits - f_bias)

    # Uniformly interpolate to 200 grid points for clean, smooth plotting
    target_evals = np.linspace(min(raw_evals[0], 100), max_evals, 200)
    interp_errors = np.zeros_like(target_evals)

    # Step-wise best error tracking
    for idx, e in enumerate(target_evals):
        mask = raw_evals <= e
        if np.any(mask):
            interp_errors[idx] = np.min(raw_errors[mask])
        else:
            interp_errors[idx] = raw_errors[0]

    return {
        "func_id": func_id,
        "algo_name": algo_name,
        "evals": target_evals.tolist(),
        "errors": interp_errors.tolist(),
        "final_error": float(interp_errors[-1]),
        "elapsed": float(elapsed),
    }


def main():
    max_evals = 50000
    seed = 42
    workers = min(8, os.cpu_count() or 4)

    tasks = []
    for fid, fcls, fname, fcat, fbias in CEC2020_FUNCS:
        for aname in ALGORITHMS.keys():
            tasks.append((fid, fcls, fbias, aname, max_evals, seed))

    print(f"=== Starting CEC 2020 50D Convergence Runner ({len(tasks)} tasks across {workers} workers) ===")
    t_start = time.time()

    results_by_func = {f[0]: {} for f in CEC2020_FUNCS}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_worker_task, t): t for t in tasks}
        completed = 0
        for fut in as_completed(futures):
            res = fut.result()
            fid = res["func_id"]
            aname = res["algo_name"]
            results_by_func[fid][aname] = res
            completed += 1
            print(f"[{completed:02d}/{len(tasks):02d}] {fid:<4} | {aname:<14} | Err: {res['final_error']:.2e} | {res['elapsed']:.1f}s")

    total_time = time.time() - t_start
    print(f"\nAll benchmark simulations completed in {total_time:.1f}s.")

    # Save JSON data
    json_path = os.path.join(os.path.dirname(__file__), "cec2020_convergence_data.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(results_by_func, jf, indent=2)
    print(f"Saved convergence history data to: {json_path}")

    # Plotting 2 rows x 5 columns
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 5, figsize=(24, 9.5), dpi=300)
    axes = axes.flatten()

    for idx, (fid, fcls, fname, fcat, fbias) in enumerate(CEC2020_FUNCS):
        ax = axes[idx]
        func_data = results_by_func[fid]

        for aname, astyle in ALGORITHMS.items():
            if aname in func_data:
                evs = np.array(func_data[aname]["evals"])
                errs = np.array(func_data[aname]["errors"])
                ax.plot(
                    evs,
                    errs,
                    color=astyle["color"],
                    linestyle=astyle["linestyle"],
                    linewidth=astyle["linewidth"],
                    label=astyle["label"],
                    zorder=astyle["zorder"],
                )

        ax.set_yscale("log")
        ax.set_xlim(0, max_evals)
        ax.set_title(f"{fid}: {fname}\n({fcat})", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Evaluations (NFE)", fontsize=10)
        ax.set_ylabel(r"Error $\log_{10}(f - f^*)$", fontsize=10)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=9)

    # Unified Legend on the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=6,
        fontsize=12,
        frameon=True,
        framealpha=0.95,
        edgecolor="#D1D5DB",
    )

    plt.suptitle(
        "IEEE CEC 2020 Convergence Profiles in 50 Dimensions (MaxNFE = 50,000)\nAD-BSA vs. Canonical Competitors",
        fontsize=15,
        fontweight="bold",
        y=1.06,
    )
    plt.tight_layout()

    out_png1 = os.path.join(os.path.dirname(__file__), "cec2020_convergence_curves.png")
    out_png2 = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "cec2020_convergence_curves.png")
    os.makedirs(os.path.dirname(out_png2), exist_ok=True)

    fig.savefig(out_png1, dpi=300, bbox_inches="tight")
    fig.savefig(out_png2, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated convergence curves figure at:\n  1. {out_png1}\n  2. {out_png2}")


if __name__ == "__main__":
    main()
