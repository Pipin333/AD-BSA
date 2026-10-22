"""
Audit Script for Phase 0:
Recalculate Friedman ranking using scipy.stats.rankdata(method='average')
Evaluate Mann-Whitney U per function between independent runs
Compute Wilcoxon signed-rank test on function means across the 10 benchmark functions
Apply Holm-Bonferroni correction and compare with reported values.
"""

import json
import os
import numpy as np
from scipy import stats

def audit_dataset(json_path: str, dataset_name: str, control_algo: str = "AD-BSA"):
    if not os.path.exists(json_path):
        print(f"[-] File not found: {json_path}")
        return

    print("=" * 80)
    print(f"AUDITING PHASE 0: {dataset_name} ({json_path})")
    print("=" * 80)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    funcs = meta.get("functions", [f"F{i}" for i in range(1, 11)])
    algos = meta.get("algorithms", list(data.get("friedman_ranks", {}).keys()))

    print(f"Algorithms ({len(algos)}): {algos}")
    print(f"Functions ({len(funcs)}): {funcs}")
    print(f"Reported Friedman ranks in JSON:")
    for a, r in sorted(data.get("friedman_ranks", {}).items(), key=lambda x: x[1]):
        print(f"  {a:15s}: {r:.4f}")

    # 1. Compute means per function per algorithm
    summary = data.get("summary_statistics", {})
    raw_runs = data.get("raw_runs", {})

    means_matrix = np.zeros((len(funcs), len(algos)))  # shape: (n_funcs, n_algos)

    for i, f in enumerate(funcs):
        for j, a in enumerate(algos):
            if f in summary and a in summary[f].get("algorithms", {}):
                means_matrix[i, j] = summary[f]["algorithms"][a]["mean_error"]
            elif f in raw_runs and a in raw_runs[f]:
                means_matrix[i, j] = np.mean(raw_runs[f][a])
            else:
                print(f"Warning: missing data for {f}, {a}")

    # 2. Recalculate Friedman with rankdata(method='average')
    # For minimization, lowest error gets rank 1
    recalculated_ranks = np.zeros_like(means_matrix)
    for i in range(len(funcs)):
        recalculated_ranks[i, :] = stats.rankdata(means_matrix[i, :], method="average")

    mean_friedman_ranks = np.mean(recalculated_ranks, axis=0)

    print("\n--- RECALCULATED FRIEDMAN RANKS (method='average' on means) ---")
    recalc_dict = {algos[j]: mean_friedman_ranks[j] for j in range(len(algos))}
    for a, r in sorted(recalc_dict.items(), key=lambda x: x[1]):
        orig_r = data.get("friedman_ranks", {}).get(a, None)
        diff_str = f"(Original: {orig_r:.4f}, Diff: {r - orig_r:+.4f})" if orig_r is not None else ""
        print(f"  {a:15s}: {r:.4f} {diff_str}")

    # Check ties per function
    print("\n--- TIES ANALYSIS PER FUNCTION ---")
    for i, f in enumerate(funcs):
        row = means_matrix[i, :]
        unique_vals, counts = np.unique(row, return_counts=True)
        ties = unique_vals[counts > 1]
        if len(ties) > 0:
            ranks_str = ", ".join(f"{algos[j]}={recalculated_ranks[i, j]:.1f}" for j in range(len(algos)))
            print(f"  {f}: {len(ties)} tie groups. Recalculated ranks: {ranks_str}")

    # Friedman test overall statistic
    try:
        f_stat, p_val = stats.friedmanchisquare(*[means_matrix[:, j] for j in range(len(algos))])
        print(f"\nFriedman Chi-Square test over {len(funcs)} functions:")
        print(f"  Statistic: {f_stat:.4f}, p-value: {p_val:.4e}")
    except Exception as e:
        print(f"Could not compute Friedman test statistic: {e}")

    # 3. Mann-Whitney U test per function between independent runs
    print(f"\n--- MANN-WHITNEY U PER FUNCTION ({control_algo} vs Competitors) ---")
    if control_algo not in algos:
        print(f"Control algo {control_algo} not found in algos.")
        return

    ctrl_idx = algos.index(control_algo)
    mw_signs = {a: {"+": 0, "=": 0, "-": 0} for a in algos if a != control_algo}
    mw_pvals_per_func = {f: {} for f in funcs}

    for f in funcs:
        if f not in raw_runs:
            continue
        ctrl_runs = np.asarray(raw_runs[f][control_algo], dtype=np.float64)
        for a in algos:
            if a == control_algo:
                continue
            comp_runs = np.asarray(raw_runs[f][a], dtype=np.float64)
            if np.allclose(ctrl_runs, comp_runs, atol=1e-12):
                p_u = 1.0
                sign = "="
            else:
                try:
                    res = stats.mannwhitneyu(ctrl_runs, comp_runs, alternative="two-sided")
                    p_u = res.pvalue
                    med_ctrl = np.median(ctrl_runs)
                    med_comp = np.median(comp_runs)
                    if p_u < 0.05:
                        sign = "+" if med_ctrl < med_comp else "-"
                    else:
                        sign = "="
                except Exception as e:
                    p_u = 1.0
                    sign = "="
            mw_signs[a][sign] += 1
            mw_pvals_per_func[f][a] = (p_u, sign)

    print("\nSummary of Mann-Whitney U (+/=/ - where + means control wins):")
    for a in algos:
        if a == control_algo:
            continue
        s = mw_signs[a]
        print(f"  {control_algo} vs {a:15s}: + {s['+']} / = {s['=']} / - {s['-']}")

    # 4. Wilcoxon signed-rank test across the 10 function means
    print(f"\n--- WILCOXON SIGNED-RANK TEST ON FUNCTION MEANS ({len(funcs)} functions) ---")
    wilcoxon_means = {}
    for j, a in enumerate(algos):
        if a == control_algo:
            continue
        ctrl_means = means_matrix[:, ctrl_idx]
        comp_means = means_matrix[:, j]
        diffs = ctrl_means - comp_means

        if np.allclose(diffs, 0.0, atol=1e-12):
            wilcoxon_means[a] = {"p_value": 1.0, "stat": 0.0, "sign": "="}
            continue

        try:
            res = stats.wilcoxon(ctrl_means, comp_means, alternative="two-sided")
            p_val = res.pvalue
            stat = res.statistic
            med_ctrl = np.median(ctrl_means)
            med_comp = np.median(comp_means)
            mean_ctrl = np.mean(ctrl_means)
            mean_comp = np.mean(comp_means)
            # In signed rank, check sum of ranks of positive vs negative differences
            pos_diffs = np.sum(diffs < 0) # control is better (error is smaller)
            neg_diffs = np.sum(diffs > 0) # competitor is better
            sign = "="
            if p_val < 0.05:
                sign = "+" if np.sum(diffs < 0) > np.sum(diffs > 0) else "-"
            wilcoxon_means[a] = {"p_value": p_val, "stat": stat, "sign": sign, "diffs": diffs}
        except Exception as e:
            wilcoxon_means[a] = {"p_value": 1.0, "stat": 0.0, "sign": "=", "error": str(e)}

    # Holm-Bonferroni correction on Wilcoxon p-values across competitors
    comp_names = [a for a in algos if a != control_algo]
    p_values = [wilcoxon_means[a]["p_value"] for a in comp_names]
    sorted_indices = np.argsort(p_values)
    m = len(comp_names)

    print(f"{'Competitor':15s} | {'Raw p-val':10s} | {'Holm Threshold':14s} | {'Holm Adj p-val':14s} | {'Significant?'}")
    print("-" * 75)
    running_max_adj = 0.0
    for rank_idx, idx in enumerate(sorted_indices):
        algo = comp_names[idx]
        raw_p = p_values[idx]
        k = rank_idx + 1
        threshold = 0.05 / (m - k + 1)
        adj_p = min(1.0, raw_p * (m - k + 1))
        # Holm adjusted p-values must be non-decreasing
        running_max_adj = max(running_max_adj, adj_p)
        adj_p = min(1.0, running_max_adj)
        sig = "YES" if adj_p < 0.05 else "NO"
        sign = wilcoxon_means[algo]["sign"]
        print(f"{algo:15s} | {raw_p:10.4e} | {threshold:14.4e} | {adj_p:14.4e} | {sig} ({sign})")

    # 5. Post-hoc Nemenyi or Holm test on Friedman ranks
    print(f"\n--- POST-HOC TEST ON FRIEDMAN RANKS (Control vs All) ---")
    # For Friedman post-hoc with control algorithm (Bonferroni-Dunn or Holm):
    # z = (R_control - R_i) / sqrt(k * (k + 1) / (6 * N))
    # where k = len(algos), N = len(funcs)
    k = len(algos)
    N = len(funcs)
    se = np.sqrt(k * (k + 1) / (6.0 * N))
    print(f"Standard Error SE = sqrt({k} * {k+1} / (6 * {N})) = {se:.4f}")

    r_ctrl = recalc_dict[control_algo]
    z_scores = {}
    p_vals_posthoc = {}
    for a in comp_names:
        r_i = recalc_dict[a]
        z = (r_i - r_ctrl) / se  # positive z means control has smaller (better) rank
        # Two-sided p-value
        p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
        z_scores[a] = z
        p_vals_posthoc[a] = p

    # Sort by p-value for Holm procedure
    sorted_posthoc_indices = np.argsort([p_vals_posthoc[a] for a in comp_names])
    print(f"{'Competitor':15s} | {'Rank Diff (R_i - R_ctrl)':24s} | {'z-score':10s} | {'Raw p-val':10s} | {'Holm Adj p-val':14s} | {'Significant?'}")
    print("-" * 90)
    running_max_adj_ph = 0.0
    for rank_idx, idx in enumerate(sorted_posthoc_indices):
        algo = comp_names[idx]
        raw_p = p_vals_posthoc[algo]
        r_diff = recalc_dict[algo] - r_ctrl
        z = z_scores[algo]
        k_step = rank_idx + 1
        adj_p = min(1.0, raw_p * (m - k_step + 1))
        running_max_adj_ph = max(running_max_adj_ph, adj_p)
        adj_p = min(1.0, running_max_adj_ph)
        sig = "YES" if adj_p < 0.05 else "NO"
        print(f"{algo:15s} | {r_diff:24.4f} | {z:10.4f} | {raw_p:10.4e} | {adj_p:14.4e} | {sig}")

if __name__ == "__main__":
    audit_dataset("benchmarks/cec2020_20d_results.json", "CEC 2020 20D")
    audit_dataset("benchmarks/cec2020_50d_results.json", "CEC 2020 50D")
