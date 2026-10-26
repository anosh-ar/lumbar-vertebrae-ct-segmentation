# Part D: evaluation on the test sets

Made by `scripts/06_evaluate_on_test_sets.py`. Scoring rules: `lumbar/src/lumbar/evaluation.py`. Dice on the original CT grid; a level is scored only where the ground truth has it (label scraps < 2 ml excluded). Cells: mean / median (number of vertebrae).

### Dice per vertebra (strict)

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.877 / 0.969 (70) | 0.896 / 0.968 (67) | 0.908 / 0.968 (63) | 0.910 / 0.961 (63) | 0.911 / 0.950 (65) | 0.900 / 0.962 (328) |
| VerSe19 val + test (out-of-domain) | 0.892 / 0.937 (66) | 0.888 / 0.941 (62) | 0.897 / 0.941 (60) | 0.894 / 0.937 (59) | 0.903 / 0.927 (58) | 0.895 / 0.937 (305) |

### Dice per vertebra, complete vertebrae only (not cut off by the scan edge)

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.915 / 0.973 (60) | 0.925 / 0.974 (60) | 0.936 / 0.969 (58) | 0.914 / 0.965 (57) | 0.928 / 0.950 (57) | 0.923 / 0.966 (292) |
| VerSe19 val + test (out-of-domain) | 0.890 / 0.937 (63) | 0.901 / 0.941 (59) | 0.911 / 0.941 (57) | 0.908 / 0.937 (56) | 0.902 / 0.927 (55) | 0.902 / 0.937 (290) |

### Resampling ceiling (ground truth -> 1.5 mm -> back; best possible at 1.5 mm)

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.999 / 1.000 (70) | 1.000 / 1.000 (67) | 1.000 / 1.000 (63) | 1.000 / 1.000 (63) | 1.000 / 1.000 (65) | 1.000 / 1.000 (328) |
| VerSe19 val + test (out-of-domain) | 0.955 / 0.957 (66) | 0.956 / 0.958 (62) | 0.957 / 0.961 (60) | 0.957 / 0.959 (59) | 0.955 / 0.956 (58) | 0.956 / 0.957 (305) |

### Dice per vertebra, without the L6 patients

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.877 / 0.969 (70) | 0.896 / 0.968 (67) | 0.908 / 0.968 (63) | 0.910 / 0.961 (63) | 0.911 / 0.950 (65) | 0.900 / 0.962 (328) |
| VerSe19 val + test (out-of-domain) | 0.890 / 0.937 (63) | 0.885 / 0.941 (59) | 0.895 / 0.941 (57) | 0.892 / 0.937 (56) | 0.907 / 0.930 (55) | 0.894 / 0.937 (290) |

### Level counting: does renaming the predicted levels by one step fit better?

| Test set | cases with a scored level | best: as predicted | best: one level up | best: one level down |
|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 78 | 77 | 0 | 1 |
| VerSe19 val + test (out-of-domain) | 67 | 66 | 0 | 1 |

Cases where a one-level renaming raises the mean Dice by more than 0.1 (the model found the vertebrae but counted them wrong): s1762 (one level down: 0.11 -> 0.26), sub-verse252 (one level down: 0.01 -> 0.94)

### Levels predicted where the ground truth has none

Size matters: a speck of a few voxels and a whole extra vertebra are different errors. A lumbar vertebra is about 60 ml; 'vertebra-sized' = 20 ml or more.

| Test set | cases | cases with a false-positive level | of these: no lumbar level in the ground truth | false-positive levels: median size | vertebra-sized ones |
|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 109 | 51 | 27 | 4.4 ml (182) | 26 |
| VerSe19 val + test (out-of-domain) | 80 | 23 | 13 | 9.5 ml (85) | 21 |

Vertebra-sized false-positive levels: sub-verse081 L3 68 ml (touches scan edge), sub-verse012 L4 66 ml, sub-verse252 L5 64 ml, s1096 L2 62 ml, sub-verse116 L4 61 ml, s1096 L3 60 ml (touches scan edge), s1323 L3 53 ml, sub-verse059 L2 52 ml (touches scan edge), s1053 L5 52 ml (touches scan edge), sub-verse150 L3 51 ml (touches scan edge), s0802 L1 51 ml (touches scan edge), s1323 L2 48 ml, sub-verse417_split-verse277 L2 48 ml, sub-verse221 L5 46 ml (touches scan edge), sub-verse018 L5 46 ml, s1323 L5 45 ml, sub-verse250 L5 42 ml (touches scan edge), sub-verse400_split-verse090 L4 37 ml, sub-verse125 L2 37 ml (touches scan edge), s1075 L4 37 ml, s1357 L1 36 ml, s0684 L3 34 ml (touches scan edge), s2137 L5 34 ml, s1119 L5 34 ml, s1075 L5 32 ml, sub-verse412_split-verse235 L5 31 ml (touches scan edge), sub-verse217 L5 31 ml (touches scan edge), s1997 L5 30 ml, sub-verse250 L3 29 ml, sub-verse011 L1 28 ml (touches scan edge), s0829 L2 28 ml, s0684 L2 26 ml, sub-verse404_split-verse209 L1 26 ml, s1112 L4 25 ml, s1192 L4 24 ml, sub-verse400_split-verse090 L3 22 ml, s1195 L2 22 ml, sub-verse404_split-verse209 L4 22 ml, s1075 L2 22 ml, s0684 L5 22 ml, s1195 L4 22 ml, s0802 L2 21 ml (touches scan edge), s1827 L5 21 ml (touches scan edge), s1593 L5 21 ml, sub-verse230 L5 21 ml (touches scan edge), sub-verse250 L1 21 ml, s0230 L4 20 ml

### Patients with a sixth lumbar vertebra (L6)

| Case | L1 | L2 | L3 | L4 | L5 | best renaming |
|---|---|---|---|---|---|---|
| sub-verse023 | 0.94 | 0.94 | 0.94 | 0.92 | 0.72 | as predicted |
| sub-verse083 | 0.94 | 0.94 | 0.94 | 0.94 | 0.88 | as predicted |
| sub-verse131 | 0.91 | 0.94 | 0.94 | 0.93 | 0.91 | as predicted |

### Lowest mean Dice per test set (for a visual check)

- TotalSegmentator v3 test (in-domain): s0230 0.00 [TRUNCATED]; s0482 0.00 [TRUNCATED]; s0829 0.00 [LABEL_AFFINE_EPS TRUNCATED]; s1762 0.11 [VOLUME_OUTLIER]; s1357 0.47 [MASK_OVERLAP TRUNCATED MULTI_CC]; s1119 0.47 [PAD_OUTSIDE_FOV TRUNCATED MULTI_CC]; s2060 0.53 [MASK_OVERLAP MULTI_CC]; s1248 0.55 [METAL MASK_OVERLAP]
- VerSe19 val + test (out-of-domain): sub-verse252 0.01 [NONAXIAL LABEL_AFFINE_EPS TRUNCATED]; sub-verse144 0.46 [METAL]; sub-verse093 0.50 [METAL]; sub-verse085 0.57 []; sub-verse022 0.77 []; sub-verse150 0.85 []; sub-verse020 0.88 [NONAXIAL LABEL_AFFINE_EPS]; sub-verse080 0.89 []

## Part C: vertebral body centres

Distances in mm: median / 90th percentile (number of vertebrae). 'VerSe point' = the annotated centroid in VerSe, which marks the vertebral body centre (Part A).

| Test set | predicted body vs ground-truth body | predicted body vs VerSe point | ground-truth body vs VerSe point | ground-truth whole vertebra vs VerSe point |
|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.3 / 6.3 (339) | - | - | - |
| VerSe19 val + test (out-of-domain) | 0.7 / 4.0 (305) | 1.0 / 5.6 (305) | 0.9 / 4.4 (305) | 12.5 / 14.7 (305) |

Of 644 predicted vertebrae that also exist in the ground truth, 22 were too small to separate the body (mostly cut off by the scan edge); their whole-vertebra centre is used.

Inference + scoring time per case on this machine: median 6 s, max 56 s.
