# Part D: evaluation on the test sets

Made by `scripts/06_evaluate_on_test_sets.py`. Scoring rules: `lumbar/src/lumbar/evaluation.py`. Dice on the original CT grid; a level is scored only where the ground truth has it (label scraps < 2 ml excluded). Cells: mean / median (number of vertebrae).

### Dice per vertebra (strict)

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.876 / 0.971 (70) | 0.901 / 0.970 (67) | 0.920 / 0.967 (63) | 0.907 / 0.965 (63) | 0.869 / 0.957 (65) | 0.894 / 0.965 (328) |
| VerSe19 val + test (out-of-domain) | 0.902 / 0.937 (66) | 0.910 / 0.942 (62) | 0.916 / 0.943 (60) | 0.914 / 0.939 (59) | 0.891 / 0.931 (58) | 0.906 / 0.939 (305) |

### Dice per vertebra, complete vertebrae only (not cut off by the scan edge)

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.929 / 0.975 (60) | 0.931 / 0.975 (60) | 0.945 / 0.973 (58) | 0.922 / 0.967 (57) | 0.879 / 0.958 (57) | 0.921 / 0.968 (292) |
| VerSe19 val + test (out-of-domain) | 0.900 / 0.937 (63) | 0.924 / 0.943 (59) | 0.929 / 0.943 (57) | 0.926 / 0.939 (56) | 0.889 / 0.932 (55) | 0.913 / 0.939 (290) |

### Resampling ceiling (ground truth -> 1.5 mm -> back; best possible at 1.5 mm)

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.999 / 1.000 (70) | 1.000 / 1.000 (67) | 1.000 / 1.000 (63) | 1.000 / 1.000 (63) | 1.000 / 1.000 (65) | 1.000 / 1.000 (328) |
| VerSe19 val + test (out-of-domain) | 0.955 / 0.957 (66) | 0.956 / 0.958 (62) | 0.957 / 0.961 (60) | 0.957 / 0.959 (59) | 0.955 / 0.956 (58) | 0.956 / 0.957 (305) |

### Dice per vertebra, without the L6 patients

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.876 / 0.971 (70) | 0.901 / 0.970 (67) | 0.920 / 0.967 (63) | 0.907 / 0.965 (63) | 0.869 / 0.957 (65) | 0.894 / 0.965 (328) |
| VerSe19 val + test (out-of-domain) | 0.900 / 0.937 (63) | 0.908 / 0.943 (59) | 0.914 / 0.943 (57) | 0.913 / 0.939 (56) | 0.891 / 0.932 (55) | 0.905 / 0.939 (290) |

### Level counting: does renaming the predicted levels by one step fit better?

| Test set | cases with a scored level | best: as predicted | best: one level up | best: one level down |
|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 78 | 76 | 0 | 2 |
| VerSe19 val + test (out-of-domain) | 67 | 66 | 0 | 1 |

Cases where a one-level renaming raises the mean Dice by more than 0.1 (the model found the vertebrae but counted them wrong): s0829 (one level down: 0.00 -> 0.65), s1248 (one level down: 0.24 -> 0.60), sub-verse252 (one level down: 0.20 -> 0.80)

### Levels predicted where the ground truth has none

Size matters: a speck of a few voxels and a whole extra vertebra are different errors. A lumbar vertebra is about 60 ml; 'vertebra-sized' = 20 ml or more.

| Test set | cases | cases with a false-positive level | of these: no lumbar level in the ground truth | false-positive levels: median size | vertebra-sized ones |
|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 109 | 52 | 27 | 3.1 ml (194) | 11 |
| VerSe19 val + test (out-of-domain) | 80 | 23 | 13 | 7.0 ml (91) | 26 |

Vertebra-sized false-positive levels: s1096 L2 69 ml (touches scan edge), sub-verse081 L3 68 ml (touches scan edge), sub-verse116 L4 63 ml (touches scan edge), sub-verse012 L4 63 ml, sub-verse221 L5 55 ml, sub-verse252 L5 55 ml, sub-verse250 L5 47 ml, sub-verse018 L5 47 ml, sub-verse059 L2 46 ml, sub-verse404_split-verse209 L5 46 ml (touches scan edge), s1323 L2 40 ml, sub-verse417_split-verse277 L2 40 ml, sub-verse125 L2 38 ml (touches scan edge), s1323 L4 34 ml, sub-verse400_split-verse090 L5 31 ml (touches scan edge), sub-verse412_split-verse235 L5 29 ml (touches scan edge), sub-verse217 L5 28 ml (touches scan edge), sub-verse150 L4 28 ml (touches scan edge), sub-verse011 L3 28 ml, s0684 L2 28 ml, s0684 L3 27 ml (touches scan edge), sub-verse150 L3 27 ml (touches scan edge), sub-verse404_split-verse209 L1 27 ml, s0802 L3 26 ml, sub-verse400_split-verse090 L1 26 ml, s0684 L1 26 ml, sub-verse250 L4 25 ml, s1051 L5 25 ml (touches scan edge), sub-verse250 L2 24 ml, s0230 L4 23 ml, sub-verse250 L1 23 ml, sub-verse414_split-verse241 L1 23 ml, sub-verse404_split-verse209 L4 23 ml, s0295 L5 22 ml (touches scan edge), sub-verse250 L3 22 ml, s1075 L5 22 ml, sub-verse412_split-verse235 L2 22 ml

### Patients with a sixth lumbar vertebra (L6)

| Case | L1 | L2 | L3 | L4 | L5 | best renaming |
|---|---|---|---|---|---|---|
| sub-verse023 | 0.95 | 0.94 | 0.94 | 0.94 | 0.88 | as predicted |
| sub-verse083 | 0.94 | 0.94 | 0.95 | 0.92 | 0.87 | as predicted |
| sub-verse131 | 0.92 | 0.94 | 0.94 | 0.93 | 0.91 | as predicted |

### Lowest mean Dice per test set (for a visual check)

- TotalSegmentator v3 test (in-domain): s0230 0.00 [TRUNCATED]; s0829 0.00 [LABEL_AFFINE_EPS TRUNCATED]; s1248 0.24 [METAL MASK_OVERLAP]; s1119 0.29 [PAD_OUTSIDE_FOV TRUNCATED MULTI_CC]; s0907 0.32 [MASK_OVERLAP TRUNCATED MULTI_CC]; s1762 0.33 [VOLUME_OUTLIER]; s1551 0.46 [TRUNCATED]; s0753 0.47 [TRUNCATED]
- VerSe19 val + test (out-of-domain): sub-verse252 0.20 [NONAXIAL LABEL_AFFINE_EPS TRUNCATED]; sub-verse144 0.51 [METAL]; sub-verse085 0.78 []; sub-verse040 0.82 [NONAXIAL ANISO METAL LABEL_AFFINE_EPS]; sub-verse417_split-verse278 0.84 [NONAXIAL LABEL_AFFINE_EPS SPLIT_SCAN]; sub-verse108 0.84 [NONAXIAL]; sub-verse022 0.85 []; sub-verse070 0.87 []

## Part C: vertebral body centres

Distances in mm: median / 90th percentile (number of vertebrae). 'VerSe point' = the annotated centroid in VerSe, which marks the vertebral body centre (Part A).

| Test set | predicted body vs ground-truth body | predicted body vs VerSe point | ground-truth body vs VerSe point | ground-truth whole vertebra vs VerSe point |
|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.3 / 15.7 (340) | - | - | - |
| VerSe19 val + test (out-of-domain) | 0.6 / 3.9 (305) | 1.0 / 5.6 (305) | 0.9 / 4.4 (305) | 12.5 / 14.7 (305) |

Of 645 predicted vertebrae that also exist in the ground truth, 28 were too small to separate the body (mostly cut off by the scan edge); their whole-vertebra centre is used.

Inference + scoring time per case on this machine: median 7 s, max 62 s.
