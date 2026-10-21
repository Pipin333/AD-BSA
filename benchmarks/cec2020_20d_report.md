# IEEE CEC 2020 Benchmark Results (20 Dimensions, 5 Runs)

Evaluación comparativa rigurosa sobre las 10 funciones canónicas de la **IEEE CEC 2020** en **$D = 20$**, bajo el estándar oficial de $\text{MaxNFE} = 10,000 \cdot D = 200,000$ evaluaciones por corrida y $5$ ejecuciones independientes controladas con semillas pseudoaleatorias.

### Ranking Promedio de Friedman (Menor es Mejor):

| Posición | Algoritmo | Rango Friedman | Tipo / Linaje |
| :---: | :--- | :---: | :--- |
| **#1** | `L-SHADE` | **2.80** | Campeón IEEE CEC 2014 (DE Adaptativa con LPSR, Tanabe & Fukunaga) |
| **#2** | `AD-BSA` | **3.20** | **Propuesto: Repulsión Cosecante Acotada $|\csc(x)|$ + Anti-Atractores** |
| **#3** | `jSO` | **4.30** | Campeón IEEE CEC 2017 (Extensión de iL-SHADE, Brest et al.) |
| **#4** | `CMA-ES` | **4.60** | Hansen Covariance Matrix Adaptation con IPOP Restarts |
| **#5** | `NL-SHADE-RSP` | **5.10** | Campeón IEEE CEC 2021 (NLPSR + Rank Selective Pressure + Hybrid Crossover, Stanovov et al.) |
| **#6** | `NL-SHADE-LBC` | **5.30** | Top IEEE CEC 2022 (NLPSR + Generalized Lehmer Linear Bias Change, Stanovov et al.) |
| **#7** | `MadDE` | **6.40** | Top IEEE CEC 2021 (Multi-Adaptation DE + QBX + Sobol QMC, Biswas et al.) |
| **#8** | `Standard-DE` | **6.40** | Differential Evolution clásica DE/rand/1/bin |
| **#9** | `Standard-PSO` | **8.40** | Particle Swarm Optimization canónico con decaimiento de inercia |
| **#10** | `Cuckoo-Search` | **8.50** | Cuckoo Search canónico con Vuelos de Lévy |

---

### Tabla Detallada de Rendimiento: Media ± Desv. Estándar (Error $\Delta f = f(x) - f^*$) y Test de Wilcoxon:

Signos de Wilcoxon vs `AD-BSA`: `+` (AD-BSA supera significativamente al competidor, $p < 0.05$), `=` (estadísticamente equivalente, $p \ge 0.05$), `-` (competidor supera significativamente a AD-BSA, $p < 0.05$).

| Función | Categoría | `AD-BSA` (Propuesto) | `NL-SHADE-LBC` | `NL-SHADE-RSP` | `MadDE` | `jSO` | `CMA-ES` | `L-SHADE` | `Standard-DE` | `Standard-PSO` | `Cuckoo-Search` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1** (Shifted and Rota...) | Unimodal | **0.00e+00** | 0.00e+00 (=) | 0.00e+00 (=) | 0.00 ± 0.0 (=) | 0.00e+00 (=) | 0.00e+00 (=) | 0.00e+00 (=) | 0.00e+00 (=) | 1145.11 ± 945.9 (=) | 8200.59 ± 3944.7 (=) |
| **F2** (Shifted and Rota...) | Multimodal | **0.00e+00** | 0.00e+00 (=) | 0.00e+00 (=) | 0.00e+00 (=) | 0.00e+00 (=) | 0.00e+00 (=) | 0.00e+00 (=) | 0.00e+00 (=) | 0.00e+00 (=) | 0.00e+00 (=) |
| **F3** (Shifted and Rota...) | Multimodal | **21.94 ± 0.7** | 22.14 ± 0.8 (=) | 23.98 ± 1.6 (=) | 25.85 ± 1.6 (=) | 21.66 ± 0.4 (=) | 22.70 ± 0.8 (=) | 22.96 ± 0.3 (=) | 86.13 ± 15.4 (=) | 46.32 ± 5.9 (=) | 98.03 ± 23.6 (=) |
| **F4** (Expanded Rosenbr...) | Multimodal | **1.66 ± 0.4** | 1.45 ± 0.4 (=) | 3.33 ± 0.2 (=) | 2.19 ± 0.4 (=) | 1.28 ± 0.1 (=) | 1.38 ± 0.4 (=) | 1.16 ± 0.1 (=) | 8.07 ± 2.7 (=) | 4.66 ± 1.9 (=) | 5.41 ± 1.1 (=) |
| **F5** (Hybrid Function ...) | Hybrid | **89.66 ± 147.8** | 0.08 ± 0.2 (=) | 3568.01 ± 1348.5 (=) | 2.03e+04 ± 4.8e+03 (=) | 0.00e+00 (=) | 21.30 ± 19.8 (=) | 0.00e+00 (=) | 164.02 ± 281.6 (=) | 1.37e+05 ± 1.4e+05 (=) | 7.61e+04 ± 2.3e+04 (=) |
| **F6** (Hybrid Function ...) | Hybrid | **0.54 ± 0.7** | 0.00e+00 (=) | 0.00e+00 (=) | 0.00e+00 (=) | 0.57 ± 0.7 (=) | 68.52 ± 135.9 (=) | 0.56 ± 0.7 (=) | 0.00e+00 (=) | 3659.21 ± 6267.2 (=) | 31.13 ± 45.0 (=) |
| **F7** (Hybrid Function ...) | Hybrid | **66.03 ± 79.8** | 141.83 ± 95.4 (=) | 66.72 ± 64.7 (=) | 712.78 ± 265.6 (=) | 64.78 ± 80.0 (=) | 72.97 ± 143.3 (=) | 54.39 ± 84.1 (=) | 10.50 ± 21.0 (=) | 1.54e+04 ± 8.5e+03 (=) | 2.63e+04 ± 1.5e+04 (=) |
| **F8** (Composition Func...) | Composition | **0.00e+00** | 80.23 ± 40.3 (=) | 0.00e+00 (=) | 0.00e+00 (=) | 0.00e+00 (=) | 82.34 ± 41.2 (=) | 0.00e+00 (=) | 0.00e+00 (=) | 0.00e+00 (=) | 0.00e+00 (=) |
| **F9** (Composition Func...) | Composition | **200.00 ± 0.0** | 200.00 ± 0.0 (=) | 200.00 ± 0.0 (=) | 200.00 ± 0.0 (=) | 200.00 ± 0.0 (=) | 180.00 ± 40.0 (=) | 200.00 ± 0.0 (=) | 200.00 ± 0.0 (=) | 234.77 ± 69.5 (=) | 200.00 ± 0.0 (=) |
| **F10** (Composition Func...) | Composition | **628.97 ± 13.7** | 654.80 ± 15.3 (=) | 655.41 ± 15.4 (=) | 642.64 ± 25.2 (=) | 622.02 ± 0.0 (=) | 622.04 ± 0.0 (=) | 622.07 ± 0.0 (=) | 628.88 ± 13.7 (=) | 646.02 ± 19.5 (=) | 624.09 ± 3.0 (=) |

---

### Resumen Estadístico Global de Wilcoxon (`AD-BSA` vs Competidores en $D=20$):

| Competidor | Año / Referencia | `AD-BSA` Mejor (`+`) | Empate Estadístico (`=`) | Competidor Mejor (`-`) |
| :--- | :--- | :---: | :---: | :---: |
| `NL-SHADE-LBC` | Top IEEE CEC 2022 (NLPSR + Generalized Lehmer Linear Bias Change, Stanovov et al.) | **0** | 10 | 0 |
| `NL-SHADE-RSP` | Campeón IEEE CEC 2021 (NLPSR + Rank Selective Pressure + Hybrid Crossover, Stanovov et al.) | **0** | 10 | 0 |
| `MadDE` | Top IEEE CEC 2021 (Multi-Adaptation DE + QBX + Sobol QMC, Biswas et al.) | **0** | 10 | 0 |
| `jSO` | Campeón IEEE CEC 2017 (Extensión de iL-SHADE, Brest et al.) | **0** | 10 | 0 |
| `CMA-ES` | Hansen Covariance Matrix Adaptation con IPOP Restarts | **0** | 10 | 0 |
| `L-SHADE` | Campeón IEEE CEC 2014 (DE Adaptativa con LPSR, Tanabe & Fukunaga) | **0** | 10 | 0 |
| `Standard-DE` | Differential Evolution clásica DE/rand/1/bin | **0** | 10 | 0 |
| `Standard-PSO` | Particle Swarm Optimization canónico con decaimiento de inercia | **0** | 10 | 0 |
| `Cuckoo-Search` | Cuckoo Search canónico con Vuelos de Lévy | **0** | 10 | 0 |
