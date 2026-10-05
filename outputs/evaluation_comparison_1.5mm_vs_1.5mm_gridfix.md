# Model comparison: 1.5mm vs 1.5mm_gridfix

Made by `scripts/08_compare_two_evaluations.py` from the step-6 results. Dice on the original CT grid.

| | TotalSeg test 1.5mm | TotalSeg test 1.5mm_gridfix | VerSe19 1.5mm | VerSe19 1.5mm_gridfix |
|---|---|---|---|---|
| Dice L1: all, mean / median | 0.876 / 0.971 | 0.876 / 0.971 | 0.902 / 0.937 | 0.897 / 0.938 |
| Dice L1: not cut off, mean / median | 0.929 / 0.975 | 0.929 / 0.975 | 0.900 / 0.937 | 0.895 / 0.938 |
| Dice L2: all, mean / median | 0.901 / 0.970 | 0.901 / 0.970 | 0.910 / 0.942 | 0.910 / 0.942 |
| Dice L2: not cut off, mean / median | 0.931 / 0.975 | 0.931 / 0.975 | 0.924 / 0.943 | 0.924 / 0.943 |
| Dice L3: all, mean / median | 0.920 / 0.967 | 0.920 / 0.967 | 0.916 / 0.943 | 0.914 / 0.943 |
| Dice L3: not cut off, mean / median | 0.945 / 0.973 | 0.945 / 0.973 | 0.929 / 0.943 | 0.929 / 0.943 |
| Dice L4: all, mean / median | 0.907 / 0.965 | 0.907 / 0.965 | 0.914 / 0.939 | 0.912 / 0.939 |
| Dice L4: not cut off, mean / median | 0.922 / 0.967 | 0.922 / 0.967 | 0.926 / 0.939 | 0.926 / 0.939 |
| Dice L5: all, mean / median | 0.869 / 0.957 | 0.869 / 0.957 | 0.891 / 0.931 | 0.891 / 0.931 |
| Dice L5: not cut off, mean / median | 0.879 / 0.958 | 0.879 / 0.958 | 0.889 / 0.932 | 0.889 / 0.932 |
| Dice all: all, mean / median | 0.894 / 0.965 | 0.894 / 0.965 | 0.906 / 0.939 | 0.905 / 0.939 |
| Dice all: not cut off, mean / median | 0.921 / 0.968 | 0.921 / 0.968 | 0.913 / 0.939 | 0.912 / 0.939 |
| Resampling ceiling, mean | 1.000 | 1.000 | 0.956 | 0.956 |
| Vertebrae scored (cases) | 328 (78) | 328 (78) | 305 (67) | 305 (67) |
| Cases counted one level off | 2: s0829, s1248 | 2: s0829, s1248 | 1: sub-verse252 | 1: sub-verse252 |
| Body centre: predicted vs ground-truth body, median / 90th pct (mm) | 0.3 / 15.7 | 0.3 / 15.7 | 0.6 / 3.9 | 0.6 / 3.9 |
| Body centre: predicted vs VerSe annotated point, median / 90th pct (mm) | - | - | 1.0 / 5.6 | 1.0 / 5.6 |
| False-positive levels: cases with any / levels / median size | 52 / 194 / 3.1 ml | 52 / 194 / 3.1 ml | 23 / 91 / 7.0 ml | 23 / 92 / 5.8 ml |
| False-positive levels: vertebra-sized (>= 20 ml) | 11 | 11 | 26 | 27 |

CPU-only inference time per scan (same scans, same machine, seconds):

| scan | 2 mm model (s) | 1.5 mm model (s) |
|---|---|---|
| s0013 | 26.0 | 64.0 |
| s0291 | 38.0 | 96.0 |
| s1446 | 9.0 | 22.0 |
| sub-verse020 | 18.0 | 45.0 |
| sub-verse053 | 5.0 | 12.0 |
| sub-verse089 | 5.0 | 7.0 |
| median | 13.5 | 33.5 |
