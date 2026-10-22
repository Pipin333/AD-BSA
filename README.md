<div align="center">

# AD-BSA: Adaptive Differential Boogeyman Search Algorithm

### *An Exploratory Investigation of Explicit Negative Learning and Anti-Attractor Dynamics in Differential Evolution*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-exploratory%20research%20%2F%20diagnostic-orange)](#academic-status--limitations)
[![Tests](https://github.com/Pipin333/AD-BSA/actions/workflows/ci.yml/badge.svg)](https://github.com/Pipin333/AD-BSA/actions)
[![Code Style](https://img.shields.io/badge/code%20style-PEP%208-black)](https://www.python.org/dev/peps/pep-0008/)

</div>

---

## 📌 Academic Status & Research Context

> [!IMPORTANT]
> **Document Purpose**: This repository contains an exploratory research implementation of **AD-BSA**, designed to investigate whether **explicit negative learning** (repulsion from sub-optimal solution clusters) can enhance basin evacuation in continuous global optimization.
>
> The documentation below has been audited to separate **implemented algorithms**, **preliminary observations**, **diagnostic findings**, and **open questions requiring further validation**. Results are presented without promotional claims of superiority over state-of-the-art baselines.

---

## 1. Algorithmic Overview & Mathematical Formulation

AD-BSA builds upon the success-history parameter adaptation of **SHADE** / **L-SHADE** (Tanabe & Fukunaga, 2014) and **jSO** (Brest et al., 2017), introducing an exploratory tripartite mutation operator:

$$\mathbf{v}_i = \mathbf{x}_i + \underbrace{F_{\text{safe}, i} \cdot (\mathbf{x}_{p\text{-best}} - \mathbf{x}_i)}_{\text{Positive Elite Attraction}} + \underbrace{\mathbf{v}_{\text{escape}, i}}_{\text{Repulsive Escape Term}} + \underbrace{F_{\text{diff}, i} \cdot (\mathbf{x}_{r1} - \tilde{\mathbf{x}}_{r2})}_{\text{Differential Perturbation}}$$

### 1.1. Anti-Attractor Barycenters
At each generation, the worst $k = 15\%$ fraction of the population is sorted and partitioned into $M = 2$ contiguous rank-based sub-tiers $\mathcal{C}_1, \mathcal{C}_2$. The barycenter of each tier acts as an anti-attractor epicenter:

$$\mathbf{S}_m = \frac{1}{|\mathcal{C}_m|} \sum_{\mathbf{x} \in \mathcal{C}_m} \mathbf{x}, \quad m \in \{1, \dots, M\}$$

Each individual $\mathbf{x}_i$ is assigned to its nearest anti-attractor $\mathbf{S}_{\text{closest}, i} = \arg\min_{\mathbf{S}_m} \|\mathbf{x}_i - \mathbf{S}_m\|_2$.

### 1.2. The Implemented Repulsion Profile
The capture radius $R_c$ is calibrated to the population's mean coordinate standard deviation:

$$R_c = \max\left(0.5 \cdot \bar{\sigma}_{\text{pop}}, \, 10^{-12}\right), \quad \bar{\sigma}_{\text{pop}} = \frac{1}{D} \sum_{j=1}^D \sigma_j$$

The Euclidean distance $r_i = \|\mathbf{x}_i - \mathbf{S}_{\text{closest}, i}\|_2$ is normalized as:

$$r_{\text{norm}, i} = \frac{r_i}{2 R_c + \epsilon}$$

In the current code (`src/ad_bsa/algorithm.py`), the force profile $\phi(r_{\text{norm}})$ is implemented as:

$$\arg = \text{clip}\left(r_{\text{norm}} \cdot \frac{\pi}{2}, \; 10^{-3}, \; 0.999\pi\right)$$

$$\phi(r_{\text{norm}}) = \begin{cases}
\text{clip}\left(\left|\csc(\arg)\right|, \, 1.0, \, F_{\max}\right) - 1.0 & \text{if } r_{\text{norm}} < 2.0 \\
0.0 & \text{if } r_{\text{norm}} \ge 2.0
\end{cases}$$

$$\mathbf{v}_{\text{escape}, i} = F_{\text{escape}, i}(t) \cdot \phi(r_{\text{norm}, i}) \cdot \frac{\mathbf{x}_i - \mathbf{S}_{\text{closest}, i}}{r_i} \cdot R_c$$

where $F_{\text{escape}, i}(t) = F_{\text{raw}, i} \cdot \max\left(0, \left(1 - \frac{t}{\text{MaxNFE}}\right)^{1.5}\right)$.

### 1.3. Mathematical Properties of the Implemented Profile
An analytical trace of the implemented profile reveals key structural properties:

| $r_{\text{norm}}$ | $\arg$ (rad) | $\sin(\arg)$ | $\phi(r_{\text{norm}})$ ($F_{\max}=4$) | Mathematical Characterization |
| :---: | :---: | :---: | :---: | :--- |
| **$0.00$** | $0.0010$ | $0.0010$ | **$3.0000$** | Upper saturation limit ($F_{\max} - 1.0$). |
| **$0.50$** | $0.7854$ ($\pi/4$) | $0.7071$ | **$0.4142$** | Monotonic decrease from peak. |
| **$1.00$** | $1.5708$ ($\pi/2$) | $1.0000$ | **$0.0000$** | **Null force point**: $\phi \equiv 0$ at distance $r = 2 R_c$. |
| **$1.50$** | $2.3562$ ($3\pi/4$) | $0.7071$ | **$0.4142$** | **Rebound anomaly**: force increases as distance increases. |
| **$1.80$** | $2.8274$ | $0.3090$ | **$2.2361$** | Force increases as solution moves away from target. |
| **$1.90$** | $2.9845$ | $0.1564$ | **$3.0000$** | Force saturates at upper bound far from the anti-attractor. |
| **$1.99$** | $3.1259$ | $0.0157$ | **$3.0000$** | Maximum force on outer boundary. |
| **$2.00$** | $3.1385$ | $0.0031$ | **$0.0000$** | **Finite jump discontinuity**: drops from $3.0 \to 0.0$ at threshold. |
| **$> 2.00$** | — | — | **$0.0000$** | Inactive by boolean filter. |

> [!WARNING]
> **Discrepancy Between Conceptual Intent and Algebraic Implementation**:
> 1. **Non-monotonicity**: The function does not decrease continuously. It reaches zero at $r_{\text{norm}} = 1.0$, increases back to $3.0$ between $1.0$ and $1.9$, and then drops discontinuously to $0.0$ at $r_{\text{norm}} = 2.0$.
> 2. **Conceptual Distinction**: While the physical motivation aimed for an aversive potential field repelling trapped solutions, the current algebraic mapping produces secondary outer repulsion and a sharp boundary cutoff. Addressing this formulation is an open task for future versions.

---

## 2. Experimental Diagnostics & Observations

### 2.1. Mechanistic Diagnostic on F3 ($D=20$)
To verify whether the repulsion operator was actively contributing in 20-dimensional search spaces, a single instrumented run was executed on **CEC 2020 F3** ($D=20$, $\text{MaxNFE} = 200,000$, $N_{\text{init}}=360$, seed `101000`) comparing `Full AD-BSA` and `w/o Repulsion` from an identical initial population $P_0$.

Detailed generation-by-generation telemetry (archived in [`benchmarks/diagnostic_pre_meeting_f3_20d.json`](benchmarks/diagnostic_pre_meeting_f3_20d.json)) showed:

* **Total generations evaluated**: $2,528$.
* **Generaciones con repulsión activa ($\|v_{\text{esc}}\| > 0$)**: **12 de 2.528 ($0.47\%$)**.
* **Generaciones con repulsión nula ($\|v_{\text{esc}}\| \equiv 0$)**: **2.516 de 2.528 ($99.53\%$)**.
* **Identidad inicial y divergencia**: Hasta la generación 25, $v_{\text{esc}} \equiv 0$ para todos los individuos; ambas variantes ejecutaron pasos idénticos bit a bit. La primera divergencia ocurrió en la generación 26 por una única activación local ($\|v_{\text{esc}}\|_{\max} = 2.59$).
* **Magnitud media en la corrida**: $\|v_{\text{esc}}\|_{\text{media}} = 3.93 \times 10^{-6}$ frente a $\|v_{\text{safe}}\|_{\text{media}} = 1.57$ (razón efectiva $\sim 2.5 \times 10^{-6}$).
* **Causa geométrica**: En $D=20$, la distancia euclidiana inter-individuos e individuo-centroide escala con $\mathcal{O}(\sqrt{D} \sigma) \approx \sqrt{20}\sigma \approx 4.47\sigma$. Como $r_{\text{norm}} = \frac{d}{2 R_c} \approx \frac{d}{\sigma}$, la mediana observada de $r_{\text{norm}}$ se mantuvo en **$4.64$** durante toda la búsqueda. Debido a la condición de corte $r_{\text{norm}} < 2.0$, la casi totalidad de los individuos quedó fuera del radio de activación.

> [!NOTE]
> These figures reflect a **single diagnostic execution** on F3 ($D=20$) and serve as an empirical explanation of why the operator remained largely inactive under the current dimensional parameterization, rather than a definitive characterization across all possible settings.

---

## 3. Benchmark Protocol Evaluations

### 3.1. Paired Proof-of-Concept Protocol ($D=20$, $\text{MaxNFE} = 200,000$, $n=15$ Runs)
To evaluate the contribution of individual components without initialization noise, a strictly paired benchmark was conducted across 15 independent runs per function. Each run shared the identical initial population $P_0$ and random seed across four variants:
1. `AD-BSA (Full)`: Complete architecture with cosecant repulsion.
2. `AD-BSA (w/o Repulsion)`: Repulsion disabled ($v_{\text{esc}} \equiv 0$).
3. `AD-BSA (Random Direction)`: Identical $\|v_{\text{esc}}\|$ module, isotropic random direction.
4. `L-SHADE (Reference)`: Canonical CEC 2014 winner initialized from the same $P_0$.

Results are archived in [`benchmarks/phase3_poc_results.json`](benchmarks/phase3_poc_results.json).

#### Validated Functions (F3, F4, F10)
As documented in the audit section below, F3, F4, and F10 do not depend on the third-party Schwefel operator defect and represent clean continuous landscapes:

| Function | Landscape Type | `AD-BSA (Full)` | `w/o Repulsion` | `Random Dir` | `L-SHADE (Ref)` | Wilcoxon: Full vs w/o Rep | Wilcoxon: Full vs L-SHADE | Holm Adj. $p$ (vs L-SHADE) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F3** | Lunacek bi-Rastrigin | $22.75$ (mean $22.4$) | $22.47$ (mean $22.4$) | $22.17$ (mean $22.5$) | $22.65$ (mean $22.6$) | $p = 0.937$ ($=$) | $p = 0.561$ ($=$) | $p_{\text{adj}} = 1.000$ |
| **F4** | Rosenbrock + Griewank| $1.62$ (mean $1.70$) | $1.60$ (mean $1.70$) | $1.86$ (mean $2.06$) | **$1.41$ (mean $1.33$)** | $p = 0.678$ ($=$) | **$p = 4.27 \times 10^{-4}$ ($-$)** | **$p_{\text{adj}} = 0.0026$ ($-$)** |
| **F10** | Composition 3 ($N=5$)| $622.07$ (mean $622.1$) | $622.10$ (mean $623.4$) | $622.11$ (mean $625.7$) | $622.05$ (mean $622.1$) | $p = 0.638$ ($=$) | $p = 0.048$ ($-$) | $p_{\text{adj}} = 0.144$ ($=$) |

*Notation*:
* `(+)` : AD-BSA outperforms competitor with statistical significance ($p < 0.05$).
* `(=)` : Difference is not statistically significant ($p \ge 0.05$; **does not demonstrate equivalence**).
* `(-)` : Competitor outperforms AD-BSA with statistical significance ($p < 0.05$).
* All reported differences and $p$-values are two-sided paired Wilcoxon signed-rank tests ($n=15$). Multiplicity correction applied via Holm-Bonferroni ($m=6$).

**Key Takeaways from the Paired PoC**:
1. **Full vs. w/o Repulsion**: In all three clean functions, $p \gg 0.05$. In $D=20$, deactivating the repulsion operator does not produce a statistically detectable change in error, consistent with the mechanistic observation that $v_{\text{esc}} \equiv 0$ during $99.5\%$ of the generations.
2. **AD-BSA vs. L-SHADE**: Following Holm-Bonferroni correction, L-SHADE exhibits a statistically significant advantage on **F4** ($p_{\text{adj}} = 0.0026$). On F3 and F10, differences do not reach statistical significance at the $5\%$ level.

---

### 3.2. Exploratory Fixed-Budget Protocol ($D=50$, $\text{MaxNFE} = 50,000$, $n=10$ Runs)
An earlier exploratory benchmark evaluated 10 independent runs across the 10 problem formulations of the CEC 2020 suite under an independent, compact budget protocol ($1,000 \cdot D = 50,000$ evaluations).

> [!WARNING]
> **Methodological Caveat on 50D**:
> 1. **Non-Official Dimension**: The official IEEE CEC 2020 competition established standard dimensions $D \in \{5, 10, 15, 20\}$ with budgets of $10,000 \cdot D$. Dimension $D=50$ is not an official CEC 2020 competition category; rotation matrices were inherited by third-party packages from earlier suites.
> 2. **Population Constraint on Baselines**: Canonical L-SHADE ($N_{\text{init}} = 18D = 900$) and jSO ($N_{\text{init}} \approx 691$) exhaust a large fraction of the 50,000 evaluation budget during their initial iterations, constraining their terminal exploitation phase.
> 3. **Exclusion of Contaminated Problems**: Problems F2, F5, F6, and F7 contain dependencies on the third-party Schwefel defect (detailed below) and are excluded from validated comparative claims.

---

## 4. Benchmark Audit & Known Third-Party Defect in `opfunu`

An inspection of the third-party dependency `opfunu==1.0.1` (`opfunu/utils/operator.py`, lines 95–105) identified a sign inversion defect in `modified_schwefel_func`:
1. **Amplitude Sign Inversion**: For coordinates exceeding the domain boundary ($z > 500$), the official IEEE CEC C-code specification defines the folded coordinate amplitude as $(500.0 - \text{fmod}(z, 500))$. In `opfunu/utils/operator.py`, this term is incorrectly implemented with an addition: `(500.0 + np.fmod(np.abs(z), 500))`.
2. **Spurious Negative Valleys**: Because the amplitude grows up to $\approx 1000$ instead of decaying within $[0, 500]$, subtracting this term (`fx -= (...) - penalty`) causes the amplified sine wave to overpower the quadratic boundary penalty $( (z - 500) / 100 )^2 / n_x$ around $z \approx 997.55$. This creates an artificial global minimum of **$-553.791$** (verified empirically), strictly below the theoretical function minimum of $0.0$.
3. **Artificial Zero Truncation**: When individuals escape into this spurious basin, $f(\mathbf{x}) < f_{\text{bias}}$. Benchmark runners computing $\text{error} = \max(0.0, f_{\text{best}} - f_{\text{bias}})$ truncate these negative deltas to exact zeros ($0.00\text{e}+00$).

### Dependency Map and Status Across CEC 2020 Problems:
* **F1, F3, F4, F9, F10**: **Clean**. No dependency on `modified_schwefel_func`.
* **F2**: **Direct dependency**. Historical results reporting exact zeros ($0.00\text{e}+00$) across multiple algorithms are an artifact of this defect and are **invalid**.
* **F5, F6, F7, F8**: **Sub-component dependency**. These hybrid and composition functions route a partition of coordinates into `modified_schwefel_func`. In the 20D PoC, negative deltas were observed in 43% of F5 runs and 35% of F6 runs, truncating to zero. Results on these problems are marked as **contaminated / non-validated**.

---

## 5. Component Ablation Analysis

A preliminary exploratory ablation study on 4 functions in 50D ($n=5$ runs) evaluated:
* `Full AD-BSA`
* `w/o Cosecant Repulsion` ($v_{\text{esc}} \equiv 0$)
* `w/o Power-Law Cooling` ($\gamma = 0$)
* `w/o Stratification` ($M = 1$)

### Status of Ablation Conclusions:
1. **F2 Exclusion**: F2 can no longer serve as evidence of component contribution due to the Schwefel penalty defect.
2. **Sample Size Limitations**: With $n=5$ runs, small mean differences cannot establish causal contribution or statistical significance without larger sample sizes.
3. **Reported Figures**: Prior statements claiming a "38.5% performance improvement on F1" reflected an unverified mean difference over 5 runs and have been reclassified as **exploratory / unverified**.

---

## 6. Open Questions & Roadmap for AD-BSA

Based on the diagnostic findings, future research will address:
1. **Dimensional Scaling of Capture Radius ($R_c$)**: Reformulating $R_c \propto \mathcal{O}(\sqrt{D} \cdot \bar{\sigma})$ so that the active threshold $r_{\text{norm}} < \tau$ scales with the natural concentration of measure in $\mathbb{R}^D$.
2. **Continuous Monotonic Repulsion Kernels**: Replacing the bounded cosecant profile with smooth, monotonic radial kernels (e.g., Gaussian $\exp(-r_{\text{norm}}^2)$ or inverse-quadratic $\frac{1}{1 + r_{\text{norm}}^2}$) to eliminate jump discontinuities and rebound anomalies.
3. **Low-Dimensional Evaluation ($D \le 5$)**: Assessing whether explicit negative learning provides measurable escape utility in low-dimensional multimodal landscapes where swarm density remains coupled to local basins.

---

## 📂 Repository Structure

```
AD-BSA/
├── LICENSE                         # Apache License Version 2.0
├── NOTICE                          # Apache attribution notice
├── CITATION.cff                    # Citation metadata
├── pyproject.toml                  # Package configuration
├── requirements.txt                # Core dependencies
├── README.md                       # Audited scientific documentation
├── src/
│   └── ad_bsa/                     # Core Library
│       ├── __init__.py             # Public API exports
│       ├── algorithm.py            # AD-BSA implementation
│       ├── competitors.py          # Baseline implementations (jSO, L-SHADE, CMA-ES, etc.)
│       └── utils.py                # Boundary handlers and evaluation wrappers
├── benchmarks/
│   ├── phase3_poc_results.json            # 360-run paired PoC results (D=20, n=15, 4 variants)
│   ├── diagnostic_pre_meeting_f3_20d.json # Generation-by-generation diagnostic telemetry
│   ├── run_phase3_paired_experiment.py    # Paired PoC benchmark runner
│   └── run_diagnostic_pre_meeting.py      # Diagnostic telemetry runner
└── tests/
    ├── test_algorithms.py          # Convergence tests across all algorithms
    ├── test_bounds.py              # Boundary constraint tests
    └── test_cec2020_integration.py # Benchmark suite integration tests
```

---

## ⚖️ License & Attribution

Licensed under the **Apache License Version 2.0**. See the [LICENSE](LICENSE) file for details.
