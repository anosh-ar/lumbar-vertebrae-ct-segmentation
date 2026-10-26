# Part D: evaluation on the test sets

Made by `scripts/06_evaluate_on_test_sets.py`. Scoring rules: `lumbar/src/lumbar/evaluation.py`. Dice on the original CT grid; a level is scored only where the ground truth has it (label scraps < 2 ml excluded). Cells: mean / median (number of vertebrae).

### Dice per vertebra (strict)

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.859 / 0.970 (70) | 0.883 / 0.975 (67) | 0.890 / 0.973 (63) | 0.922 / 0.967 (63) | 0.932 / 0.959 (65) | 0.896 / 0.968 (328) |
| VerSe19 val + test (out-of-domain) | 0.880 / 0.939 (66) | 0.909 / 0.941 (62) | 0.900 / 0.942 (60) | 0.899 / 0.938 (59) | 0.866 / 0.932 (58) | 0.891 / 0.939 (305) |

### Dice per vertebra, complete vertebrae only (not cut off by the scan edge)

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.937 / 0.975 (60) | 0.942 / 0.976 (60) | 0.932 / 0.976 (58) | 0.937 / 0.969 (57) | 0.936 / 0.959 (57) | 0.937 / 0.970 (292) |
| VerSe19 val + test (out-of-domain) | 0.890 / 0.939 (63) | 0.908 / 0.941 (59) | 0.913 / 0.942 (57) | 0.913 / 0.938 (56) | 0.862 / 0.932 (55) | 0.898 / 0.939 (290) |

### Resampling ceiling (ground truth -> 1.5 mm -> back; best possible at 1.5 mm)

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.999 / 1.000 (70) | 1.000 / 1.000 (67) | 1.000 / 1.000 (63) | 1.000 / 1.000 (63) | 1.000 / 1.000 (65) | 1.000 / 1.000 (328) |
| VerSe19 val + test (out-of-domain) | 0.955 / 0.957 (66) | 0.956 / 0.958 (62) | 0.957 / 0.961 (60) | 0.957 / 0.959 (59) | 0.955 / 0.956 (58) | 0.956 / 0.957 (305) |

### Dice per vertebra, without the L6 patients

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.859 / 0.970 (70) | 0.883 / 0.975 (67) | 0.890 / 0.973 (63) | 0.922 / 0.967 (63) | 0.932 / 0.959 (65) | 0.896 / 0.968 (328) |
| VerSe19 val + test (out-of-domain) | 0.892 / 0.939 (63) | 0.923 / 0.942 (59) | 0.914 / 0.942 (57) | 0.913 / 0.939 (56) | 0.881 / 0.933 (55) | 0.905 / 0.940 (290) |

### Level counting: does renaming the predicted levels by one step fit better?

| Test set | cases with a scored level | best: as predicted | best: one level up | best: one level down |
|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 78 | 77 | 0 | 1 |
| VerSe19 val + test (out-of-domain) | 67 | 65 | 1 | 1 |

Cases where a one-level renaming raises the mean Dice by more than 0.1 (the model found the vertebrae but counted them wrong): s2016 (one level down: 0.04 -> 0.31), sub-verse083 (one level up: 0.01 -> 0.94)

### Levels predicted where the ground truth has none

Size matters: a speck of a few voxels and a whole extra vertebra are different errors. A lumbar vertebra is about 60 ml; 'vertebra-sized' = 20 ml or more.

| Test set | cases | cases with a false-positive level | of these: no lumbar level in the ground truth | false-positive levels: median size | vertebra-sized ones |
|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 109 | 51 | 26 | 1.1 ml (173) | 8 |
| VerSe19 val + test (out-of-domain) | 80 | 23 | 13 | 2.5 ml (81) | 12 |

Vertebra-sized false-positive levels: sub-verse081 L3 68 ml (touches scan edge), sub-verse252 L5 64 ml, sub-verse012 L4 61 ml, s1096 L2 55 ml, sub-verse059 L2 50 ml, sub-verse417_split-verse277 L2 48 ml, sub-verse018 L5 46 ml, s1323 L5 46 ml (touches scan edge), sub-verse150 L3 45 ml (touches scan edge), sub-verse116 L4 44 ml (touches scan edge), sub-verse011 L3 41 ml, s0802 L3 39 ml, sub-verse125 L2 38 ml (touches scan edge), s1323 L4 36 ml, s1096 L3 33 ml (touches scan edge), s0802 L4 30 ml, s0829 L2 25 ml, s0230 L5 21 ml, sub-verse011 L2 20 ml, sub-verse414_split-verse241 L5 20 ml

### Patients with a sixth lumbar vertebra (L6)

| Case | L1 | L2 | L3 | L4 | L5 | best renaming |
|---|---|---|---|---|---|---|
| sub-verse023 | 0.95 | 0.94 | 0.95 | 0.94 | 0.87 | as predicted |
| sub-verse083 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | one level up |
| sub-verse131 | 0.91 | 0.94 | 0.94 | 0.94 | 0.89 | as predicted |

### Lowest mean Dice per test set (for a visual check)

- TotalSegmentator v3 test (in-domain): s0230 0.00 [TRUNCATED]; s0753 0.00 [TRUNCATED]; s0829 0.00 [LABEL_AFFINE_EPS TRUNCATED]; s2016 0.04 [MASK_OVERLAP TRUNCATED]; s0482 0.12 [TRUNCATED]; s1762 0.27 [VOLUME_OUTLIER]; s1119 0.46 [PAD_OUTSIDE_FOV TRUNCATED MULTI_CC]; s1248 0.48 [METAL MASK_OVERLAP]
- VerSe19 val + test (out-of-domain): sub-verse083 0.01 [NONAXIAL LABEL_AFFINE_EPS TRANSITIONAL_L6]; sub-verse059 0.11 [NONAXIAL LABEL_AFFINE_EPS TRUNCATED]; sub-verse252 0.47 [NONAXIAL LABEL_AFFINE_EPS TRUNCATED]; sub-verse144 0.50 [METAL]; sub-verse417_split-verse278 0.77 [NONAXIAL LABEL_AFFINE_EPS SPLIT_SCAN]; sub-verse070 0.78 []; sub-verse093 0.82 [METAL]; sub-verse067 0.83 [NONAXIAL ANISO LABEL_AFFINE_EPS]

## Part C: vertebral body centres

Distances in mm: median / 90th percentile (number of vertebrae). 'VerSe point' = the annotated centroid in VerSe, which marks the vertebral body centre (Part A).

| Test set | predicted body vs ground-truth body | predicted body vs VerSe point | ground-truth body vs VerSe point | ground-truth whole vertebra vs VerSe point |
|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.3 / 13.9 (340) | - | - | - |
| VerSe19 val + test (out-of-domain) | 0.7 / 4.0 (305) | 1.0 / 5.6 (305) | 0.9 / 4.4 (305) | 12.5 / 14.7 (305) |

Of 645 predicted vertebrae that also exist in the ground truth, 28 were too small to separate the body (mostly cut off by the scan edge); their whole-vertebra centre is used.

Inference + scoring time per case on this machine: median 7 s, max 58 s.
