# Part D: evaluation on the test sets

Made by `scripts/06_evaluate_on_test_sets.py`. Scoring rules: `lumbar/src/lumbar/evaluation.py`. Dice on the original CT grid; a level is scored only where the ground truth has it (label scraps < 2 ml excluded). Cells: mean / median (number of vertebrae).

### Dice per vertebra (strict)

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.729 / 0.883 (70) | 0.768 / 0.882 (67) | 0.802 / 0.884 (63) | 0.817 / 0.881 (63) | 0.830 / 0.875 (65) | 0.788 / 0.882 (328) |
| VerSe19 val + test (out-of-domain) | 0.834 / 0.893 (66) | 0.841 / 0.897 (62) | 0.845 / 0.894 (60) | 0.831 / 0.893 (59) | 0.788 / 0.887 (58) | 0.828 / 0.893 (305) |

### Dice per vertebra, complete vertebrae only (not cut off by the scan edge)

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.823 / 0.885 (60) | 0.819 / 0.885 (60) | 0.831 / 0.885 (58) | 0.845 / 0.883 (57) | 0.851 / 0.877 (57) | 0.833 / 0.884 (292) |
| VerSe19 val + test (out-of-domain) | 0.832 / 0.893 (63) | 0.853 / 0.896 (59) | 0.858 / 0.894 (57) | 0.843 / 0.892 (56) | 0.783 / 0.888 (55) | 0.834 / 0.893 (290) |

### Resampling ceiling (ground truth -> 3 mm -> back; best possible at 3 mm)

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.892 / 0.896 (70) | 0.886 / 0.896 (67) | 0.892 / 0.898 (63) | 0.896 / 0.901 (63) | 0.893 / 0.899 (65) | 0.891 / 0.897 (328) |
| VerSe19 val + test (out-of-domain) | 0.910 / 0.908 (66) | 0.912 / 0.911 (62) | 0.915 / 0.914 (60) | 0.915 / 0.913 (59) | 0.915 / 0.914 (58) | 0.913 / 0.912 (305) |

### Dice per vertebra, without the L6 patients

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.729 / 0.883 (70) | 0.768 / 0.882 (67) | 0.802 / 0.884 (63) | 0.817 / 0.881 (63) | 0.830 / 0.875 (65) | 0.788 / 0.882 (328) |
| VerSe19 val + test (out-of-domain) | 0.847 / 0.893 (63) | 0.857 / 0.897 (59) | 0.867 / 0.895 (57) | 0.855 / 0.894 (56) | 0.814 / 0.889 (55) | 0.848 / 0.894 (290) |

### Level counting: does renaming the predicted levels by one step fit better?

| Test set | cases with a scored level | best: as predicted | best: one level up | best: one level down |
|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 78 | 73 | 0 | 5 |
| VerSe19 val + test (out-of-domain) | 67 | 65 | 1 | 1 |

Cases where a one-level renaming raises the mean Dice by more than 0.1 (the model found the vertebrae but counted them wrong): s0753 (one level down: 0.00 -> 0.36), s1119 (one level down: 0.00 -> 0.22), s1436 (one level down: 0.06 -> 0.38), sub-verse083 (one level up: 0.01 -> 0.86), sub-verse252 (one level down: 0.02 -> 0.82)

### Levels predicted where the ground truth has none

Size matters: a speck of a few voxels and a whole extra vertebra are different errors. A lumbar vertebra is about 60 ml; 'vertebra-sized' = 20 ml or more.

| Test set | cases | cases with a false-positive level | of these: no lumbar level in the ground truth | false-positive levels: median size | vertebra-sized ones |
|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 109 | 52 | 27 | 1.6 ml (169) | 8 |
| VerSe19 val + test (out-of-domain) | 80 | 23 | 13 | 5.4 ml (83) | 15 |

Vertebra-sized false-positive levels: sub-verse081 L3 68 ml (touches scan edge), sub-verse116 L4 62 ml (touches scan edge), sub-verse012 L4 58 ml, sub-verse150 L3 54 ml (touches scan edge), sub-verse059 L2 51 ml (touches scan edge), sub-verse252 L5 51 ml, sub-verse417_split-verse277 L2 47 ml, s0684 L4 44 ml, sub-verse018 L5 40 ml, s1096 L4 37 ml (touches scan edge), s0470 L3 32 ml, s1997 L5 30 ml, sub-verse125 L2 28 ml (touches scan edge), sub-verse221 L2 27 ml, sub-verse011 L1 27 ml, sub-verse404_split-verse209 L5 26 ml, s1551 L2 25 ml, sub-verse404_split-verse209 L4 24 ml, s1053 L5 23 ml (touches scan edge), s1051 L3 23 ml, sub-verse414_split-verse241 L3 23 ml, sub-verse414_split-verse241 L2 21 ml, s1096 L2 21 ml

### Patients with a sixth lumbar vertebra (L6)

| Case | L1 | L2 | L3 | L4 | L5 | best renaming |
|---|---|---|---|---|---|---|
| sub-verse023 | 0.90 | 0.70 | 0.41 | 0.21 | 0.05 | as predicted |
| sub-verse083 | 0.02 | 0.01 | 0.02 | 0.01 | 0.01 | one level up |
| sub-verse131 | 0.72 | 0.90 | 0.89 | 0.90 | 0.86 | as predicted |

### Lowest mean Dice per test set (for a visual check)

- TotalSegmentator v3 test (in-domain): s0230 0.00 [TRUNCATED]; s0470 0.00 [TRUNCATED]; s0482 0.00 [TRUNCATED]; s0753 0.00 [TRUNCATED]; s0829 0.00 [LABEL_AFFINE_EPS TRUNCATED]; s1119 0.00 [PAD_OUTSIDE_FOV TRUNCATED MULTI_CC]; s1436 0.06 [TRUNCATED]; s1762 0.28 [VOLUME_OUTLIER]
- VerSe19 val + test (out-of-domain): sub-verse083 0.01 [NONAXIAL LABEL_AFFINE_EPS TRANSITIONAL_L6]; sub-verse252 0.02 [NONAXIAL LABEL_AFFINE_EPS TRUNCATED]; sub-verse067 0.44 [NONAXIAL ANISO LABEL_AFFINE_EPS]; sub-verse023 0.45 [NONAXIAL LABEL_AFFINE_EPS TRANSITIONAL_L6]; sub-verse085 0.63 []; sub-verse138 0.69 []; sub-verse070 0.70 []; sub-verse144 0.72 [METAL]

## Part C: vertebral body centres

Distances in mm: median / 90th percentile (number of vertebrae). 'VerSe point' = the annotated centroid in VerSe, which marks the vertebral body centre (Part A).

| Test set | predicted body vs ground-truth body | predicted body vs VerSe point | ground-truth body vs VerSe point | ground-truth whole vertebra vs VerSe point |
|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.7 / 26.9 (338) | - | - | - |
| VerSe19 val + test (out-of-domain) | 0.8 / 8.3 (305) | 1.3 / 7.9 (305) | 1.0 / 4.4 (305) | 12.5 / 14.6 (305) |

Of 643 predicted vertebrae that also exist in the ground truth, 21 were too small to separate the body (mostly cut off by the scan edge); their whole-vertebra centre is used.

Inference + scoring time per case on this machine: median 2 s, max 24 s.
