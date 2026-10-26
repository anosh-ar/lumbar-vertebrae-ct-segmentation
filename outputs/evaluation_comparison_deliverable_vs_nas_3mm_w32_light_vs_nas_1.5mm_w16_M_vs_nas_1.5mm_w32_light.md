# Model comparison: deliverable vs nas_3mm_w32_light vs nas_1.5mm_w16_M vs nas_1.5mm_w32_light

Made by `scripts/08_compare_two_evaluations.py` from the step-6 results. Dice on the original CT grid.

| | TotalSeg test deliverable | TotalSeg test nas_3mm_w32_light | TotalSeg test nas_1.5mm_w16_M | TotalSeg test nas_1.5mm_w32_light | VerSe19 deliverable | VerSe19 nas_3mm_w32_light | VerSe19 nas_1.5mm_w16_M | VerSe19 nas_1.5mm_w32_light |
|---|---|---|---|---|---|---|---|---|
| Dice L1: all, mean / median | 0.876 / 0.971 | 0.729 / 0.883 | 0.877 / 0.969 | 0.859 / 0.970 | 0.902 / 0.937 | 0.834 / 0.893 | 0.892 / 0.937 | 0.880 / 0.939 |
| Dice L1: not cut off, mean / median | 0.929 / 0.975 | 0.823 / 0.885 | 0.915 / 0.973 | 0.937 / 0.975 | 0.900 / 0.937 | 0.832 / 0.893 | 0.890 / 0.937 | 0.890 / 0.939 |
| Dice L2: all, mean / median | 0.901 / 0.970 | 0.768 / 0.882 | 0.896 / 0.968 | 0.883 / 0.975 | 0.910 / 0.942 | 0.841 / 0.897 | 0.888 / 0.941 | 0.909 / 0.941 |
| Dice L2: not cut off, mean / median | 0.931 / 0.975 | 0.819 / 0.885 | 0.925 / 0.974 | 0.942 / 0.976 | 0.924 / 0.943 | 0.853 / 0.896 | 0.901 / 0.941 | 0.908 / 0.941 |
| Dice L3: all, mean / median | 0.920 / 0.967 | 0.802 / 0.884 | 0.908 / 0.968 | 0.890 / 0.973 | 0.916 / 0.943 | 0.845 / 0.894 | 0.897 / 0.941 | 0.900 / 0.942 |
| Dice L3: not cut off, mean / median | 0.945 / 0.973 | 0.831 / 0.885 | 0.936 / 0.969 | 0.932 / 0.976 | 0.929 / 0.943 | 0.858 / 0.894 | 0.911 / 0.941 | 0.913 / 0.942 |
| Dice L4: all, mean / median | 0.907 / 0.965 | 0.817 / 0.881 | 0.910 / 0.961 | 0.922 / 0.967 | 0.914 / 0.939 | 0.831 / 0.893 | 0.894 / 0.937 | 0.899 / 0.938 |
| Dice L4: not cut off, mean / median | 0.922 / 0.967 | 0.845 / 0.883 | 0.914 / 0.965 | 0.937 / 0.969 | 0.926 / 0.939 | 0.843 / 0.892 | 0.908 / 0.937 | 0.913 / 0.938 |
| Dice L5: all, mean / median | 0.869 / 0.957 | 0.830 / 0.875 | 0.911 / 0.950 | 0.932 / 0.959 | 0.891 / 0.931 | 0.788 / 0.887 | 0.903 / 0.927 | 0.866 / 0.932 |
| Dice L5: not cut off, mean / median | 0.879 / 0.958 | 0.851 / 0.877 | 0.928 / 0.950 | 0.936 / 0.959 | 0.889 / 0.932 | 0.783 / 0.888 | 0.902 / 0.927 | 0.862 / 0.932 |
| Dice all: all, mean / median | 0.894 / 0.965 | 0.788 / 0.882 | 0.900 / 0.962 | 0.896 / 0.968 | 0.906 / 0.939 | 0.828 / 0.893 | 0.895 / 0.937 | 0.891 / 0.939 |
| Dice all: not cut off, mean / median | 0.921 / 0.968 | 0.833 / 0.884 | 0.923 / 0.966 | 0.937 / 0.970 | 0.913 / 0.939 | 0.834 / 0.893 | 0.902 / 0.937 | 0.898 / 0.939 |
| Resampling ceiling, mean | 1.000 | 0.891 | 1.000 | 1.000 | 0.956 | 0.913 | 0.956 | 0.956 |
| Vertebrae scored (cases) | 328 (78) | 328 (78) | 328 (78) | 328 (78) | 305 (67) | 305 (67) | 305 (67) | 305 (67) |
| Cases counted one level off | 2: s0829, s1248 | 3: s0753, s1119, s1436 | 1: s1762 | 1: s2016 | 1: sub-verse252 | 2: sub-verse083, sub-verse252 | 1: sub-verse252 | 1: sub-verse083 |
| Body centre: predicted vs ground-truth body, median / 90th pct (mm) | 0.3 / 15.7 | 0.7 / 26.9 | 0.3 / 6.3 | 0.3 / 13.9 | 0.6 / 3.9 | 0.8 / 8.3 | 0.7 / 4.0 | 0.7 / 4.0 |
| Body centre: predicted vs VerSe annotated point, median / 90th pct (mm) | - | - | - | - | 1.0 / 5.6 | 1.3 / 7.9 | 1.0 / 5.6 | 1.0 / 5.6 |
| False-positive levels: cases with any / levels / median size | 52 / 194 / 3.1 ml | 52 / 169 / 1.6 ml | 51 / 182 / 4.4 ml | 51 / 173 / 1.1 ml | 23 / 91 / 7.0 ml | 23 / 83 / 5.4 ml | 23 / 85 / 9.5 ml | 23 / 81 / 2.5 ml |
| False-positive levels: vertebra-sized (>= 20 ml) | 11 | 8 | 26 | 8 | 26 | 15 | 21 | 12 |

CPU-only inference time per scan (same scans, same machine, seconds):

| scan | 2 mm model (s) | 1.5 mm model (s) | 3 mm NAS w32 light (s) | 1.5 mm NAS w16 M (s) | 1.5 mm NAS w32 light (s) |
|---|---|---|---|---|---|
| s0013 | 26.0 | 61.0 | 8.0 | 29.0 | 49.0 |
| s0291 | 38.0 | 89.0 | 8.0 | 41.0 | 70.0 |
| s1446 | 9.0 | 21.0 | 3.0 | 10.0 | 17.0 |
| sub-verse020 | 18.0 | 43.0 | 4.0 | 20.0 | 34.0 |
| sub-verse053 | 5.0 | 12.0 | 1.0 | 6.0 | 10.0 |
| sub-verse089 | 5.0 | 7.0 | 1.0 | 4.0 | 6.0 |
| median | 13.5 | 32.0 | 3.5 | 15.0 | 25.5 |
