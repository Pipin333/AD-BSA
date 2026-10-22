import json
import numpy as np
from scipy import stats

def run_part3():
    with open("benchmarks/phase3_poc_results.json", "r", encoding="utf-8") as f:
        res = json.load(f)

    paired = res["paired_errors"]
    funcs = ["F3", "F4", "F5", "F6", "F7", "F10"]

    results = []
    for f in funcs:
        full = np.array(paired[f]["AD-BSA (Full)"])
        rand = np.array(paired[f]["AD-BSA (Random Direction)"])
        diffs = full - rand  # Si < 0, Full es mejor (menor error)

        med_full = float(np.median(full))
        med_rand = float(np.median(rand))
        med_diff_marginal = med_full - med_rand
        med_of_diffs_paired = float(np.median(diffs))

        # Wilcoxon signed-rank
        w_res = stats.wilcoxon(full, rand, alternative="two-sided")
        w_stat = float(w_res.statistic)
        raw_p = float(w_res.pvalue)

        # Effect size: Rank-Biserial Correlation r_rb = (W+ - W-) / (W+ + W-)
        nz_diffs = diffs[diffs != 0]
        if len(nz_diffs) > 0:
            ranks = stats.rankdata(np.abs(nz_diffs))
            w_plus = np.sum(ranks[nz_diffs > 0])   # Rand es mejor
            w_minus = np.sum(ranks[nz_diffs < 0])  # Full es mejor
            total_w = w_plus + w_minus
            r_rb = float((w_plus - w_minus) / total_w)
        else:
            r_rb = 0.0

        # Bootstrap 95% CI para Mediana de Diferencias Pareadas (10,000 resamples)
        rng = np.random.default_rng(42)
        n = len(diffs)
        boot_diffs = [float(np.median(diffs[rng.integers(0, n, size=n)])) for _ in range(10000)]
        ci_low = float(np.percentile(boot_diffs, 2.5))
        ci_high = float(np.percentile(boot_diffs, 97.5))

        results.append({
            "func": f,
            "med_full": med_full,
            "med_rand": med_rand,
            "med_diff_marginal": med_diff_marginal,
            "med_of_diffs_paired": med_of_diffs_paired,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "w_stat": w_stat,
            "r_rb": r_rb,
            "raw_p": raw_p,
        })

    # Correccion Holm-Bonferroni (m=6)
    results.sort(key=lambda x: x["raw_p"])
    m = len(results)
    running_max_p = 0.0

    print("=" * 115)
    print("COMPARACION EMPIRICA: AD-BSA (Full) vs AD-BSA (Random Direction) [n=15 corridas pareadas]")
    print("=" * 115)
    print(f"{'Func':5s} | {'Med Full':10s} | {'Med Rand':10s} | {'Med of Diffs':13s} | {'95% CI (Med of Diffs)':24s} | {'r_rb':7s} | {'Raw p':10s} | {'Holm p':10s} | {'Significativo?'}")
    print("-" * 115)

    for k, r in enumerate(results, 1):
        raw_p = r["raw_p"]
        adj_p = min(1.0, raw_p * (m - k + 1))
        running_max_p = max(running_max_p, adj_p)
        adj_p = min(1.0, running_max_p)
        sig = "RECHAZA H0" if adj_p < 0.05 else "No rechaza H0"
        print(f"{r['func']:5s} | {r['med_full']:10.2e} | {r['med_rand']:10.2e} | {r['med_of_diffs_paired']:13.2e} | [{r['ci_low']:10.2e}, {r['ci_high']:10.2e}] | {r['r_rb']:+6.3f} | {raw_p:10.4e} | {adj_p:10.4e} | {sig}")

if __name__ == "__main__":
    run_part3()
