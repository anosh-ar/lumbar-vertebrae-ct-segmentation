# Model comparison: deliverable_1.5mm vs nas_fast

Made by `scripts/08_compare_two_evaluations.py` from the step-6 results. Dice on the original CT grid.

| | TotalSeg test deliverable_1.5mm | TotalSeg test nas_fast | VerSe19 deliverable_1.5mm | VerSe19 nas_fast |
|---|---|---|---|---|
| Dice L1: all, mean / median | 0.876 / 0.971 | 0.729 / 0.883 | 0.902 / 0.937 | 0.834 / 0.893 |
| Dice L1: not cut off, mean / median | 0.929 / 0.975 | 0.823 / 0.885 | 0.900 / 0.937 | 0.832 / 0.893 |
| Dice L2: all, mean / median | 0.901 / 0.970 | 0.768 / 0.882 | 0.910 / 0.942 | 0.841 / 0.897 |
| Dice L2: not cut off, mean / median | 0.931 / 0.975 | 0.819 / 0.885 | 0.924 / 0.943 | 0.853 / 0.896 |
| Dice L3: all, mean / median | 0.920 / 0.967 | 0.802 / 0.884 | 0.916 / 0.943 | 0.845 / 0.894 |
| Dice L3: not cut off, mean / median | 0.945 / 0.973 | 0.831 / 0.885 | 0.929 / 0.943 | 0.858 / 0.894 |
| Dice L4: all, mean / median | 0.907 / 0.965 | 0.817 / 0.881 | 0.914 / 0.939 | 0.831 / 0.893 |
| Dice L4: not cut off, mean / median | 0.922 / 0.967 | 0.845 / 0.883 | 0.926 / 0.939 | 0.843 / 0.892 |
| Dice L5: all, mean / median | 0.869 / 0.957 | 0.830 / 0.875 | 0.891 / 0.931 | 0.788 / 0.887 |
| Dice L5: not cut off, mean / median | 0.879 / 0.958 | 0.851 / 0.877 | 0.889 / 0.932 | 0.783 / 0.888 |
| Dice all: all, mean / median | 0.894 / 0.965 | 0.788 / 0.882 | 0.906 / 0.939 | 0.828 / 0.893 |
| Dice all: not cut off, mean / median | 0.921 / 0.968 | 0.833 / 0.884 | 0.913 / 0.939 | 0.834 / 0.893 |
| Resampling ceiling, mean | 1.000 | 0.891 | 0.956 | 0.913 |
| Vertebrae scored (cases) | 328 (78) | 328 (78) | 305 (67) | 305 (67) |
| Cases counted one level off | 2: s0829, s1248 | 3: s0753, s1119, s1436 | 1: sub-verse252 | 2: sub-verse083, sub-verse252 |
| Body centre: predicted vs ground-truth body, median / 90th pct (mm) | 0.3 / 15.7 | 0.7 / 26.9 | 0.6 / 3.9 | 0.8 / 8.3 |
| Body centre: predicted vs VerSe annotated point, median / 90th pct (mm) | - | - | 1.0 / 5.6 | 1.3 / 7.9 |
| False-positive levels: cases with any / levels / median size | 52 / 194 / 3.1 ml | 52 / 169 / 1.6 ml | 23 / 91 / 7.0 ml | 23 / 83 / 5.4 ml |
| False-positive levels: vertebra-sized (>= 20 ml) | 11 | 8 | 26 | 15 |

CPU-only inference time per scan (same scans, same machine, seconds):

| scan | 2 mm model (s) | 1.5 mm model (s) | 3 mm NAS model (s) |
|---|---|---|---|
| s0013 | 26.0 | 64.0 | 8.0 |
| s0291 | 38.0 | 96.0 | 8.0 |
| s1446 | 9.0 | 22.0 | 3.0 |
| sub-verse020 | 18.0 | 45.0 | 4.0 |
| sub-verse053 | 5.0 | 12.0 | 1.0 |
| sub-verse089 | 5.0 | 7.0 | 1.0 |
| median | 13.5 | 33.5 | 3.5 |
