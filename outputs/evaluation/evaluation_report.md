# Part D: evaluation on the test sets

Made by `scripts/06_evaluate_on_test_sets.py`. Scoring rules: `lumbar/src/lumbar/evaluation.py`. Dice on the original CT grid; a level is scored only where the ground truth has it (label scraps < 2 ml excluded). Cells: mean / median (number of vertebrae).

### Dice per vertebra (strict)

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.863 / 0.927 (70) | 0.875 / 0.931 (67) | 0.886 / 0.930 (63) | 0.904 / 0.929 (63) | 0.879 / 0.922 (65) | 0.881 / 0.928 (328) |
| VerSe19 val + test (out-of-domain) | 0.847 / 0.919 (66) | 0.848 / 0.923 (62) | 0.861 / 0.922 (60) | 0.855 / 0.920 (59) | 0.843 / 0.909 (58) | 0.851 / 0.919 (305) |

### Dice per vertebra, complete vertebrae only (not cut off by the scan edge)

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.902 / 0.931 (60) | 0.907 / 0.932 (60) | 0.916 / 0.932 (58) | 0.918 / 0.931 (57) | 0.900 / 0.923 (57) | 0.908 / 0.930 (292) |
| VerSe19 val + test (out-of-domain) | 0.844 / 0.919 (63) | 0.844 / 0.921 (59) | 0.858 / 0.922 (57) | 0.873 / 0.920 (56) | 0.855 / 0.910 (55) | 0.854 / 0.919 (290) |

### Resampling ceiling (ground truth -> 2 mm -> back; best possible at 2 mm)

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.936 / 0.940 (70) | 0.936 / 0.942 (67) | 0.937 / 0.941 (63) | 0.940 / 0.944 (63) | 0.937 / 0.941 (65) | 0.937 / 0.942 (328) |
| VerSe19 val + test (out-of-domain) | 0.936 / 0.938 (66) | 0.938 / 0.940 (62) | 0.939 / 0.944 (60) | 0.939 / 0.943 (59) | 0.937 / 0.937 (58) | 0.938 / 0.940 (305) |

### Dice per vertebra, without the L6 patients

| Test set | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.863 / 0.927 (70) | 0.875 / 0.931 (67) | 0.886 / 0.930 (63) | 0.904 / 0.929 (63) | 0.879 / 0.922 (65) | 0.881 / 0.928 (328) |
| VerSe19 val + test (out-of-domain) | 0.844 / 0.919 (63) | 0.844 / 0.923 (59) | 0.858 / 0.922 (57) | 0.852 / 0.920 (56) | 0.840 / 0.910 (55) | 0.847 / 0.919 (290) |

### Level counting: does renaming the predicted levels by one step fit better?

| Test set | cases with a scored level | best: as predicted | best: one level up | best: one level down |
|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 78 | 77 | 0 | 1 |
| VerSe19 val + test (out-of-domain) | 67 | 63 | 3 | 1 |

Cases where a one-level renaming raises the mean Dice by more than 0.1 (the model found the vertebrae but counted them wrong): s0829 (one level down: 0.02 -> 0.58), sub-verse078 (one level up: 0.14 -> 0.79), sub-verse085 (one level up: 0.15 -> 0.59), sub-verse093 (one level down: 0.00 -> 0.67), sub-verse260 (one level up: 0.20 -> 0.55)

### Levels predicted where the ground truth has none

Size matters: a speck of a few voxels and a whole extra vertebra are different errors. A lumbar vertebra is about 60 ml; 'vertebra-sized' = 20 ml or more.

| Test set | cases | cases with a false-positive level | of these: no lumbar level in the ground truth | false-positive levels: median size | vertebra-sized ones |
|---|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 109 | 51 | 27 | 0.8 ml (158) | 4 |
| VerSe19 val + test (out-of-domain) | 80 | 23 | 13 | 1.1 ml (76) | 12 |

Vertebra-sized false-positive levels: s1096 L3 69 ml (touches scan edge), sub-verse081 L3 67 ml (touches scan edge), sub-verse116 L4 64 ml (touches scan edge), sub-verse012 L4 62 ml (touches scan edge), sub-verse150 L3 51 ml (touches scan edge), sub-verse018 L5 48 ml, sub-verse417_split-verse277 L2 46 ml, sub-verse059 L2 43 ml (touches scan edge), sub-verse011 L1 38 ml (touches scan edge), s1323 L5 35 ml, sub-verse221 L5 35 ml, sub-verse125 L2 32 ml (touches scan edge), s2137 L5 25 ml, sub-verse252 L5 25 ml, s1096 L2 25 ml, sub-verse011 L2 22 ml (touches scan edge)

### Patients with a sixth lumbar vertebra (L6)

| Case | L1 | L2 | L3 | L4 | L5 | best renaming |
|---|---|---|---|---|---|---|
| sub-verse023 | 0.93 | 0.92 | 0.92 | 0.93 | 0.87 | as predicted |
| sub-verse083 | 0.92 | 0.93 | 0.93 | 0.94 | 0.92 | as predicted |
| sub-verse131 | 0.90 | 0.92 | 0.93 | 0.92 | 0.90 | as predicted |

### Lowest mean Dice per test set (for a visual check)

- TotalSegmentator v3 test (in-domain): s0230 0.00 [TRUNCATED]; s1573 0.00 [TRUNCATED FRAGMENT]; s0829 0.02 [LABEL_AFFINE_EPS TRUNCATED]; s1119 0.42 [PAD_OUTSIDE_FOV TRUNCATED MULTI_CC]; s1357 0.43 [MASK_OVERLAP TRUNCATED MULTI_CC]; s1248 0.46 [METAL MASK_OVERLAP]; s0482 0.51 [TRUNCATED]; s1551 0.57 [TRUNCATED]
- VerSe19 val + test (out-of-domain): sub-verse093 0.00 [METAL]; sub-verse078 0.14 []; sub-verse085 0.15 []; sub-verse260 0.20 [NONAXIAL LABEL_AFFINE_EPS TRUNCATED]; sub-verse144 0.47 [METAL]; sub-verse149 0.63 []; sub-verse116 0.85 []; sub-verse252 0.86 [NONAXIAL LABEL_AFFINE_EPS TRUNCATED]

## Part C: vertebral body centres

Distances in mm: median / 90th percentile (number of vertebrae). 'VerSe point' = the annotated centroid in VerSe, which marks the vertebral body centre (Part A).

| Test set | predicted body vs ground-truth body | predicted body vs VerSe point | ground-truth body vs VerSe point | ground-truth whole vertebra vs VerSe point |
|---|---|---|---|---|
| TotalSegmentator v3 test (in-domain) | 0.6 / 8.9 (337) | - | - | - |
| VerSe19 val + test (out-of-domain) | 0.5 / 7.5 (305) | 1.0 / 7.5 (305) | 0.9 / 3.9 (305) | 12.5 / 14.6 (305) |

Of 642 predicted vertebrae that also exist in the ground truth, 32 were too small to separate the body (mostly cut off by the scan edge); their whole-vertebra centre is used.

Inference + scoring time per case on this machine: median 4 s, max 40 s.
