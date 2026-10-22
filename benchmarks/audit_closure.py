import json
import os
import sys
import numpy as np
from scipy import stats

def audit():
    print("=" * 80)
    print("AUDITORIA DE CIERRE: AD-BSA v2 PoC")
    print("=" * 80)

    # 1. Archivo de resultados y checkpoint
    results_path = "benchmarks/phase3_poc_results.json"
    ckpt_path = "benchmarks/phase3_poc_results_checkpoint.json"

    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)

    print(f"[1] Verificacion de tareas: {len(ckpt)} registros en checkpoint.")
    keys = [(r["variant"], r["func"], r["run"]) for r in ckpt]
    unique_keys = set(keys)
    print(f"    Claves unicas: {len(unique_keys)} (Esperadas: 360)")
    assert len(keys) == 360, f"Error: {len(keys)} != 360"
    assert len(unique_keys) == 360, "Error: Existen duplicados en checkpoint"

    paired = res["paired_errors"]
    funcs = ["F3", "F4", "F5", "F6", "F7", "F10"]
    variants = ["AD-BSA (Full)", "AD-BSA (w/o Repulsion)", "AD-BSA (Random Direction)", "L-SHADE (Reference)"]

    # 2. Analisis de F10 sin redondear
    print("\n[2] Detalle F10 sin redondear (15 corridas pareadas):")
    f10_full = np.array(paired["F10"]["AD-BSA (Full)"])
    f10_lshade = np.array(paired["F10"]["L-SHADE (Reference)"])
    f10_diff = f10_full - f10_lshade

    for r in range(15):
        print(f"    Run {r:2d}: Full={f10_full[r]:.8f} | L-SHADE={f10_lshade[r]:.8f} | Diff={f10_diff[r]:+.8f}")

    # Wilcoxon exacto en F10
    w_res = stats.wilcoxon(f10_full, f10_lshade, alternative="two-sided")
    print(f"    Wilcoxon F10 (Full vs L-SHADE): stat={w_res.statistic}, p-value={w_res.pvalue:.6e}")
    print(f"    Signos de las diferencias: Positivas (Full > L-SHADE)={np.sum(f10_diff > 0)}, Negativas={np.sum(f10_diff < 0)}, Ceros={np.sum(f10_diff == 0)}")

    # 3. Dispersión en F5 y F7
    print("\n[3] Analisis de Dispersion en F5 y F7:")
    for f_id in ["F5", "F7"]:
        print(f"    --- {f_id} ---")
        for v in variants:
            arr = np.array(paired[f_id][v])
            q25, q50, q75 = np.percentile(arr, [25, 50, 75])
            print(f"      {v:26s}: Median={q50:.4e}, Mean={np.mean(arr):.4e}, Std={np.std(arr):.4e}, IQR={q75-q25:.4e}, Min={np.min(arr):.4e}, Max={np.max(arr):.4e}")

    # 4. Estimando bootstrap y corrección Holm
    print("\n[4] Analisis Estadistico Exhaustivo (Full vs L-SHADE):")
    p_vals_lshade = []
    print(f"    {'Func':5s} | {'Median Diff':12s} | {'Med of Diffs':12s} | {'Bootstrap 95% CI (Med Diff)':30s} | {'Wilcoxon p':12s}")
    print("    " + "-" * 80)
    for f_id in funcs:
        a = np.array(paired[f_id]["AD-BSA (Full)"])
        b = np.array(paired[f_id]["L-SHADE (Reference)"])
        diffs = a - b

        med_diff = float(np.median(a) - np.median(b))
        med_of_diffs = float(np.median(diffs))

        # Bootstrap de med_diff
        rng = np.random.default_rng(42)
        n = len(a)
        boot_diffs = [float(np.median(a[rng.integers(0, n, size=n)]) - np.median(b[rng.integers(0, n, size=n)])) for _ in range(10000)]
        ci_low = float(np.percentile(boot_diffs, 2.5))
        ci_high = float(np.percentile(boot_diffs, 97.5))

        w_p = stats.wilcoxon(a, b, alternative="two-sided").pvalue
        p_vals_lshade.append((f_id, w_p, med_diff))
        print(f"    {f_id:5s} | {med_diff:12.4e} | {med_of_diffs:12.4e} | [{ci_low:12.4e}, {ci_high:12.4e}] | {w_p:12.4e}")

    # Aplicar correccion Holm a la familia de comparaciones contra L-SHADE
    print("\n[5] Correccion Holm-Bonferroni (Familia contra L-SHADE, m=6):")
    p_vals_lshade.sort(key=lambda x: x[1])
    m = len(p_vals_lshade)
    running_max_p = 0.0
    for k, (f_id, p_val, med_diff) in enumerate(p_vals_lshade, 1):
        threshold = 0.05 / (m - k + 1)
        adj_p = min(1.0, p_val * (m - k + 1))
        running_max_p = max(running_max_p, adj_p)
        adj_p = min(1.0, running_max_p)
        sig = "RECHAZA H0 (L-SHADE mejor)" if (adj_p < 0.05 and med_diff > 0) else "NO RECHAZA H0"
        print(f"    {f_id:5s}: Raw p={p_val:10.4e} | Holm Thresh={threshold:10.4e} | Adj p={adj_p:10.4e} -> {sig}")

    # Comparaciones Full vs w/o Repulsion con Holm
    print("\n[6] Correccion Holm-Bonferroni (Full vs w/o Repulsion, m=6):")
    p_vals_wout = []
    for f_id in funcs:
        a = np.array(paired[f_id]["AD-BSA (Full)"])
        b = np.array(paired[f_id]["AD-BSA (w/o Repulsion)"])
        w_p = stats.wilcoxon(a, b, alternative="two-sided").pvalue
        p_vals_wout.append((f_id, w_p))
    p_vals_wout.sort(key=lambda x: x[1])
    for k, (f_id, p_val) in enumerate(p_vals_wout, 1):
        adj_p = min(1.0, p_val * (m - k + 1))
        print(f"    {f_id:5s}: Raw p={p_val:10.4e} | Adj p={adj_p:10.4e}")

if __name__ == "__main__":
    audit()
