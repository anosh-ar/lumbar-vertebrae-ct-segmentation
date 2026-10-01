# Part A: header and label anomalies

Made by `scripts/01_find_header_and_label_anomalies.py`. Numbers count **cases**; one case can have several anomalies. Per-case details are in `anomalies_per_case_<dataset>.csv`.

## Datasets checked

| Dataset | Cases | Stored orientation (cases) | Largest voxel side in mm: min / median / max | Header orientation fields set, qform/sform code (cases) | Stored CT data type (cases) |
|---|---|---|---|---|---|
| totalseg_v3_test | 109 | RAS 109 | 1.50 / 1.50 / 1.50 | 0/2 109 | int16 109 |
| totalseg_v3_train | 1830 | RAS 1830 | 1.50 / 1.50 / 1.50 | 0/2 1830 | int16 1830 |
| verse19_test | 40 | PIR 17, LAS 15, PSR 8 | 1.00 / 2.00 / 3.00 | 0/2 3, 1/0 37 | float32 39, int16 1 |
| verse19_validation | 40 | PIR 17, LAS 17, PSR 6 | 1.00 / 1.00 / 3.00 | 0/2 2, 1/0 38 | float32 38, int16 2 |

Orientation letters say where each array axis points, e.g. RAS = Right, Anterior, Superior. A code of 0 means the field is not set.

## Anomalies found

| Problem | Code | totalseg_v3_test | totalseg_v3_train | verse19_test | verse19_validation | How we find it | What we do about it |
|---|---|---|---|---|---|---|---|
| Slices are not stored as axial slices (e.g. stored as sagittal slices) | `NONAXIAL` | 0 | 0 | 25 | 23 | The third array axis does not run head-to-foot | Rotated into the standard RAS orientation when we resample |
| Voxels are much longer in one direction than the others (thick slices) | `ANISO` | 0 | 0 | 2 | 2 | Longest voxel side is more than 3x the shortest | Smoothed before downsampling so no aliasing appears; flagged because training data has thin slices |
| The area outside the scanned circle is filled with a very low value | `PAD_OUTSIDE_FOV` | 12 | 92 | 0 | 0 | More than 1% of voxels are below -2000 HU (air is about -1000) | Clipped to -1000 HU by the intensity window, so it looks like air |
| Metal near the lumbar spine (e.g. screws or implants) | `METAL` | 6 | 118 | 2 | 2 | More than 0.5 ml above 3000 HU within the labelled vertebrae + 30 mm | Clipped to 1500 HU by the intensity window; flagged to explain errors later |
| The label's position differs from the CT's only by rounding noise | `LABEL_AFFINE_EPS` | 14 | 214 | 18 | 18 | Position/orientation differs by more than 0 but at most 0.001 mm | Treated as the same grid |
| The label mask is stored as decimal numbers instead of integers | `LABEL_FLOAT` | 0 | 0 | 1 | 0 | Stored data type is float | Rounded to whole numbers |
| A voxel belongs to two TotalSegmentator masks at the same time | `MASK_OVERLAP` | 57 | 760 | 0 | 0 | The voxel is non-zero in two of the per-structure mask files | The lumbar vertebra wins over T12/sacrum (lumbar masks are merged last); overlap size saved in the CSV |
| No lumbar vertebra is labelled (the scan does not cover the lumbar spine) | `NO_LUMBAR` | 27 | 693 | 4 | 9 | None of L1-L5 is in the mask | Not usable for L1-L5. Still preprocessed if T12 or sacrum is present, so the training-case selection must filter it out |
| A lumbar level is missing between two present ones (e.g. L2 and L4 but no L3) | `LEVEL_GAP` | 0 | 1 | 0 | 0 | The present lumbar levels are not consecutive | Flagged. Plan (Part D): the missing level is not scored |
| A lumbar vertebra is cut off at the edge of the scan | `TRUNCATED` | 29 | 402 | 5 | 3 | Its mask touches the image border | Kept for training; flagged because Dice on a partial vertebra is not comparable |
| A vertebra label is split into separate pieces | `MULTI_CC` | 10 | 156 | 0 | 0 | Pieces other than the largest add up to more than 0.5 ml | Left as it is; reported |
| A lumbar level is only a tiny scrap of label | `FRAGMENT` | 11 | 135 | 0 | 0 | The level's volume is below 2 ml (a real lumbar vertebra is about 60 ml) | Ignored in the order check. Plan (Part D): not scored as a vertebra |
| A vertebra has an implausible volume | `VOLUME_OUTLIER` | 1 | 31 | 0 | 0 | A lumbar vertebra that is not cut off measures 2-10 ml or more than 150 ml | Flagged for a visual check |
| The patient has a sixth lumbar vertebra (VerSe label 25) | `TRANSITIONAL_L6` | 0 | 0 | 2 | 1 | Label 25 is in the mask | Kept as its own class (id 8), never merged into L5 or background. Plan (Part D): reported separately |
| One patient's spine is split over several scans | `SPLIT_SCAN` | 0 | 0 | 6 | 6 | The VerSe file name contains '_split-' | Each scan is processed on its own; the patient id is kept so results can be grouped per patient |

"Plan (Part D)" marks handling that belongs to the evaluation step and is not built yet.

Checked but not found in any case:

- The file cannot be opened (`LOAD_ERROR`)
- The header has no position or orientation (qform and sform both unset) (`NO_XFORM`)
- The header's two position/orientation fields (qform and sform) disagree (`QS_MISMATCH`)
- Voxel size is zero, negative or missing (`BAD_PIXDIM`)
- Voxel axes are not at right angles (e.g. tilted CT gantry) (`SHEAR`)
- Image is stored with an extra, empty 4th dimension (`IMG_4D`)
- Stored values must be rescaled to get HU (scl_slope / scl_inter) (`IMG_SCALED`)
- Intensities are not in Hounsfield units: there are no negative values (`HU_OFFSET`)
- Some voxels are NaN or infinite (`NONFINITE`)
- The label mask does not line up with the CT (other size or position) (`LABEL_GRID_MISMATCH`)
- The label mask contains values that are not whole numbers (`LABEL_NONINT`)
- Label values must be rescaled (scl_slope / scl_inter) (`LABEL_SCALED`)
- Vertebrae are in the wrong top-to-bottom order (e.g. L3 above L2) (`ORDER_VIOLATION`)
- The patient has a thirteenth thoracic vertebra (VerSe label 28) (`TRANSITIONAL_T13`)
- The VerSe centroid file and the mask list different lumbar levels (`CTD_LABEL_MISMATCH`)

## Example cases

- `NONAXIAL`: verse19_test: sub-verse012, sub-verse020, sub-verse029, sub-verse032, sub-verse038, sub-verse040 ...; verse19_validation: sub-verse010, sub-verse011, sub-verse013, sub-verse016, sub-verse018, sub-verse023 ...
- `ANISO`: verse19_test: sub-verse040, sub-verse414_split-verse241; verse19_validation: sub-verse041, sub-verse067
- `PAD_OUTSIDE_FOV`: totalseg_v3_test: s0441, s0802, s0994, s1057, s1119, s1240 ...; totalseg_v3_train: s0339, s0486, s0487, s0494, s0495, s0586 ...
- `METAL`: totalseg_v3_test: s0029, s0236, s0673, s1152, s1248, s2073; totalseg_v3_train: s0054, s0058, s0062, s0068, s0209, s0229 ...; verse19_test: sub-verse040, sub-verse144; verse19_validation: sub-verse093, sub-verse264
- `LABEL_AFFINE_EPS`: totalseg_v3_test: s0013, s0235, s0236, s0295, s0308, s0311 ...; totalseg_v3_train: s0000, s0001, s0002, s0004, s0012, s0025 ...; verse19_test: sub-verse012, sub-verse020, sub-verse032, sub-verse038, sub-verse040, sub-verse053 ...; verse19_validation: sub-verse011, sub-verse013, sub-verse016, sub-verse018, sub-verse023, sub-verse024 ...
- `LABEL_FLOAT`: verse19_test: sub-verse012
- `MASK_OVERLAP`: totalseg_v3_test: s0013, s0029, s0038, s0040, s0119, s0235 ...; totalseg_v3_train: s0001, s0004, s0006, s0009, s0010, s0011 ...
- `NO_LUMBAR`: totalseg_v3_test: s0295, s0684, s0802, s1051, s1053, s1075 ...; totalseg_v3_train: s0000, s0002, s0003, s0021, s0025, s0035 ...; verse19_test: sub-verse217, sub-verse250, sub-verse414_split-verse241, sub-verse416_split-verse247; verse19_validation: sub-verse011, sub-verse205, sub-verse221, sub-verse225, sub-verse230, sub-verse242 ...
- `LEVEL_GAP`: totalseg_v3_train: s1048
- `TRUNCATED`: totalseg_v3_test: s0230, s0459, s0470, s0482, s0673, s0753 ...; totalseg_v3_train: s0006, s0032, s0037, s0039, s0046, s0048 ...; verse19_test: sub-verse012, sub-verse059, sub-verse119, sub-verse131, sub-verse260; verse19_validation: sub-verse030, sub-verse071, sub-verse252
- `MULTI_CC`: totalseg_v3_test: s0311, s0673, s0907, s1119, s1240, s1357 ...; totalseg_v3_train: s0037, s0062, s0068, s0071, s0074, s0115 ...
- `FRAGMENT`: totalseg_v3_test: s0951, s1057, s1096, s1176, s1323, s1573 ...; totalseg_v3_train: s0048, s0069, s0074, s0102, s0106, s0162 ...
- `VOLUME_OUTLIER`: totalseg_v3_test: s1762; totalseg_v3_train: s0437, s1329, s1411, s1414, s1447, s1487 ...
- `TRANSITIONAL_L6`: verse19_test: sub-verse083, sub-verse131; verse19_validation: sub-verse023
- `SPLIT_SCAN`: verse19_test: sub-verse414_split-verse241, sub-verse414_split-verse273, sub-verse416_split-verse247, sub-verse416_split-verse279, sub-verse417_split-verse277, sub-verse417_split-verse278; verse19_validation: sub-verse400_split-verse090, sub-verse400_split-verse155, sub-verse404_split-verse209, sub-verse404_split-verse256, sub-verse412_split-verse235, sub-verse412_split-verse290

## How much of the lumbar spine each scan shows

| Dataset | All of L1-L5 | Some of L1-L5 | None |
|---|---|---|---|
| totalseg_v3_test | 57 | 25 | 27 |
| totalseg_v3_train | 750 | 387 | 693 |
| verse19_test | 31 | 5 | 4 |
| verse19_validation | 26 | 5 | 9 |

## What the VerSe centroid annotations mean (verse19_test)

VerSe gives one annotated point per vertebra. The file does not say which unit the numbers are in, so we tried both readings and measured how far each lands from the centre of the same vertebra's mask (median per case, then median over cases). The two readings only differ on scans with non-cubic voxels, so those are shown separately.

| Reading of the numbers | Distance, all cases (n) | Distance, non-cubic voxels (n) |
|---|---|---|
| Voxel indices, after turning the image to the file's `direction` | 12.1 mm (36) | 12.8 mm (22) |
| Millimetres along those axes | 26.3 mm (36) | 40.7 mm (22) |

The annotated point lies a median +11.7 mm in front of the whole-vertebra centre (+ = towards the belly), i.e. it marks the vertebral body, not the whole vertebra.

## What the VerSe centroid annotations mean (verse19_validation)

VerSe gives one annotated point per vertebra. The file does not say which unit the numbers are in, so we tried both readings and measured how far each lands from the centre of the same vertebra's mask (median per case, then median over cases). The two readings only differ on scans with non-cubic voxels, so those are shown separately.

| Reading of the numbers | Distance, all cases (n) | Distance, non-cubic voxels (n) |
|---|---|---|
| Voxel indices, after turning the image to the file's `direction` | 12.2 mm (32) | 12.1 mm (16) |
| Millimetres along those axes | 17.9 mm (32) | 39.0 mm (16) |

The annotated point lies a median +11.7 mm in front of the whole-vertebra centre (+ = towards the belly), i.e. it marks the vertebral body, not the whole vertebra.
