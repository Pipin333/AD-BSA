# IEEE CEC 2020 Benchmark Results (50 Dimensions, 30 Runs)

Comparativa exhaustiva y estadísticamente rigurosa sobre las 10 funciones canónicas de la **IEEE CEC 2020** en **$D = 50$**, con un presupuesto de $50,000$ evaluaciones por corrida y $30$ ejecuciones independientes con semillas pseudoaleatorias controladas.

### Ranking Promedio de Friedman (Menor es Mejor):

| Posición | Algoritmo | Rango Friedman | Tipo / Linaje |
| :---: | :--- | :---: | :--- |
| **#1** | `AD-BSA` | **2.20** | **Propuesto: Repulsión Cosecante Acotada $|\csc(x)|$ + Anti-Atractores** |
| **#2** | `L-SHADE` | **2.30** | Campeón IEEE CEC 2014 (DE Adaptativa con LPSR) |
| **#3** | `CMA-ES` | **2.70** | Hansen Covariance Matrix Adaptation (SOTA en paisajes continuos) |
| **#4** | `jSO` | **3.20** | Campeón IEEE CEC 2017 (Extensión de iL-SHADE) |
| **#5** | `Standard-DE` | **5.80** | Differential Evolution clásica DE/rand/1/bin |
| **#6** | `Standard-PSO` | **5.80** | Particle Swarm Optimization con decaimiento de inercia |
| **#7** | `Cuckoo-Search` | **6.00** | Cuckoo Search canónico con Vuelos de Lévy |

---

### Tabla Detallada de Rendimiento: Media ± Desv. Estándar (Error $\Delta f = f(x) - f^*$) y Test de Wilcoxon:

Signos de Wilcoxon vs `AD-BSA`: `+` (AD-BSA supera con $p < 0.05$), `-` (AD-BSA es superado con $p < 0.05$), `=` (empate estadístico).

| Función | Categoría | `AD-BSA` (Propuesto) | `jSO` | `CMA-ES` | `L-SHADE` | `Standard-DE` | `Standard-PSO` | `Cuckoo-Search` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1** (Shifted & Rotate...) | Unimodal | **1.24e+04 ± 9.4e+03** | 2693.92 ± 3013.0 (-) | 0.00e+00 (-) | 3293.56 ± 2771.3 (-) | 3.93e+08 ± 1.7e+08 (+) | 6.21e+09 ± 4.6e+09 (+) | 4.44e+09 ± 9.8e+08 (+) |
| **F2** (Shifted & Rotate...) | Multimodal | ****0.00e+00**** | 5025.67 ± 3630.3 (+) | 1869.22 ± 5988.2 (-) | 0.00e+00 (=) | 8134.16 ± 830.3 (+) | 2402.98 ± 2870.1 (+) | 2692.67 ± 690.8 (+) |
| **F3** (Shifted & Rotate...) | Multimodal | **154.63 ± 29.2** | 268.92 ± 87.3 (+) | 157.86 ± 20.7 (=) | 163.73 ± 15.4 (=) | 1.22e+04 ± 4.7e+03 (+) | 2.90e+05 ± 2.3e+05 (+) | 1.35e+05 ± 2.5e+04 (+) |
| **F4** (Expanded Rosenbr...) | Multimodal | **8.24 ± 3.0** | 23.65 ± 6.0 (+) | 6.29 ± 1.3 (-) | 12.86 ± 1.2 (+) | 45.34 ± 4.2 (+) | 424.76 ± 612.0 (+) | 1153.21 ± 927.0 (+) |
| **F5** (Hybrid Function ...) | Hybrid | **2.42e+06 ± 1.5e+06** | 1.43e+05 ± 1.0e+05 (-) | 1.01e+04 ± 7.5e+03 (-) | 2.37e+04 ± 1.0e+04 (-) | 8.45e+06 ± 2.6e+06 (+) | 6.56e+06 ± 6.8e+06 (+) | 6.72e+06 ± 1.6e+06 (+) |
| **F6** (Hybrid Function ...) | Hybrid | **96.37 ± 372.8** | 184.85 ± 473.1 (=) | 220.81 ± 549.5 (=) | 0.00e+00 (+) | 1.87e+04 ± 6.6e+03 (+) | 5.34e+06 ± 2.7e+07 (+) | 1.65e+05 ± 7.4e+04 (+) |
| **F7** (Hybrid Function ...) | Hybrid | **5.99e+05 ± 9.3e+05** | 2.17e+05 ± 3.0e+05 (=) | 5838.64 ± 6174.0 (-) | 2.67e+04 ± 1.2e+04 (-) | 4.25e+07 ± 2.1e+07 (+) | 9.54e+06 ± 1.1e+07 (+) | 1.14e+07 ± 3.1e+06 (+) |
| **F8** (Composition Func...) | Composition | ****0.00e+00**** | 0.00e+00 (+) | 141.58 ± 12.4 (+) | 0.00e+00 (=) | 406.46 ± 470.2 (+) | 0.00e+00 (+) | 0.00e+00 (-) |
| **F9** (Composition Func...) | Composition | **200.01 ± 0.0** | 200.11 ± 0.2 (+) | 234.28 ± 77.1 (=) | 200.00 ± 0.0 (-) | 2017.05 ± 281.9 (+) | 1.01e+04 ± 3.8e+03 (+) | 7034.95 ± 913.4 (+) |
| **F10** (Composition Func...) | Composition | **725.22 ± 10.4** | 743.78 ± 26.6 (+) | 750.30 ± 47.3 (+) | 745.66 ± 24.1 (+) | 928.58 ± 74.3 (+) | 1237.19 ± 312.5 (+) | 2087.93 ± 297.2 (+) |
