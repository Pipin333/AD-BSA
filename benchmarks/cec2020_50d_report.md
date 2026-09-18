# IEEE CEC 2020 Benchmark Results (50 Dimensions, 10 Runs)

Comparativa exhaustiva y estadísticamente rigurosa sobre las 10 funciones canónicas de la **IEEE CEC 2020** en **$D = 50$**, con un presupuesto de $50,000$ evaluaciones por corrida y $10$ ejecuciones independientes con semillas pseudoaleatorias controladas.

### Ranking Promedio de Friedman (Menor es Mejor):

| Posición | Algoritmo | Rango Friedman | Tipo / Linaje |
| :---: | :--- | :---: | :--- |
| **#1** | `AD-BSA` | **2.30** | **Propuesto: Repulsión Cosecante Acotada $|\csc(x)|$ + Anti-Atractores** |
| **#2** | `CMA-ES` | **2.30** | Hansen Covariance Matrix Adaptation con IPOP Restarts |
| **#3** | `jSO` | **2.40** | Campeón IEEE CEC 2017 (Extensión de iL-SHADE) |
| **#4** | `L-SHADE` | **3.40** | Campeón IEEE CEC 2014 (DE Adaptativa con LPSR) |
| **#5** | `Standard-PSO` | **5.00** | Particle Swarm Optimization canónico con decaimiento de inercia |
| **#6** | `Standard-DE` | **6.10** | Differential Evolution clásica DE/rand/1/bin |
| **#7** | `Cuckoo-Search` | **6.50** | Cuckoo Search canónico con Vuelos de Lévy |

---

### Tabla Detallada de Rendimiento: Media ± Desv. Estándar (Error $\Delta f = f(x) - f^*$) y Test de Wilcoxon:

Signos de Wilcoxon vs `AD-BSA`: `+` (AD-BSA supera significativamente al competidor, $p < 0.05$), `=` (estadísticamente equivalente, $p \ge 0.05$), `-` (competidor supera significativamente a AD-BSA, $p < 0.05$).

| Función | Categoría | `AD-BSA` (Propuesto) | `jSO` | `CMA-ES` | `L-SHADE` | `Standard-DE` | `Standard-PSO` | `Cuckoo-Search` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1** (Shifted & Rotate...) | Unimodal | **834.28 ± 785.1** | 2112.29 ± 1756.8 (=) | 0.00e+00 (-) | 4296.57 ± 1823.2 (+) | 3.34e+08 ± 1.2e+08 (+) | 3.85e+08 ± 1.1e+09 (+) | 4.46e+09 ± 1.1e+09 (+) |
| **F2** (Shifted & Rotate...) | Multimodal | ****0.00e+00**** | 696.40 ± 1786.6 (=) | 1869.31 ± 3608.1 (=) | 157.99 ± 250.1 (=) | 8075.19 ± 719.9 (+) | 2761.38 ± 2686.2 (+) | 2004.58 ± 636.7 (+) |
| **F3** (Shifted & Rotate...) | Multimodal | **138.19 ± 19.8** | 246.56 ± 19.1 (+) | 101.12 ± 10.0 (-) | 317.47 ± 25.9 (+) | 1.24e+04 ± 4.1e+03 (+) | 9212.57 ± 11962.2 (+) | 1.43e+05 ± 2.6e+04 (+) |
| **F4** (Expanded Rosenbr...) | Multimodal | **6.61 ± 3.0** | 17.39 ± 0.9 (+) | 6.20 ± 0.9 (=) | 19.70 ± 1.3 (+) | 44.47 ± 2.8 (+) | 37.17 ± 9.6 (+) | 710.42 ± 327.1 (+) |
| **F5** (Hybrid Function ...) | Hybrid | **6.88e+04 ± 3.1e+04** | 2.29e+04 ± 8.2e+03 (-) | 8855.96 ± 6267.5 (-) | 5.11e+04 ± 1.4e+04 (=) | 7.49e+06 ± 1.9e+06 (+) | 2.89e+06 ± 2.6e+06 (+) | 6.33e+06 ± 1.9e+06 (+) |
| **F6** (Hybrid Function ...) | Hybrid | **124.94 ± 167.0** | 126.65 ± 151.3 (=) | 271.38 ± 365.2 (=) | 341.26 ± 331.8 (=) | 1.97e+04 ± 9.5e+03 (+) | 2.20e+04 ± 2.2e+04 (+) | 1.90e+05 ± 7.4e+04 (+) |
| **F7** (Hybrid Function ...) | Hybrid | **5.70e+04 ± 2.2e+04** | 3.59e+04 ± 1.6e+04 (=) | 3417.47 ± 1650.1 (-) | 4.76e+04 ± 1.2e+04 (=) | 4.26e+07 ± 1.8e+07 (+) | 2.41e+06 ± 1.5e+06 (+) | 1.40e+07 ± 4.3e+06 (+) |
| **F8** (Composition Func...) | Composition | **74.69 ± 38.2** | 106.77 ± 5.6 (+) | 136.10 ± 17.4 (+) | 121.29 ± 8.3 (+) | 564.89 ± 698.1 (+) | 69.12 ± 73.8 (=) | 191.58 ± 236.1 (=) |
| **F9** (Composition Func...) | Composition | **200.00 ± 0.0** | 200.43 ± 0.2 (+) | 214.15 ± 42.4 (=) | 203.53 ± 0.8 (+) | 1982.99 ± 300.5 (+) | 3177.06 ± 2441.9 (+) | 7014.56 ± 961.3 (+) |
| **F10** (Composition Func...) | Composition | **789.48 ± 46.2** | 724.82 ± 9.6 (-) | 726.99 ± 17.2 (-) | 733.65 ± 7.8 (-) | 953.66 ± 52.9 (+) | 849.16 ± 84.8 (=) | 2075.40 ± 206.2 (+) |
