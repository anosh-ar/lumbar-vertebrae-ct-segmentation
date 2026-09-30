"""Canonical label convention and the mapping of each dataset into it.

Every label volume in the pipeline uses these ids, whatever its source.
The model is trained on ids 0-7; ids 8 and 9 only occur in VerSe and are
kept so that transitional anatomy stays visible to QC and evaluation
instead of silently becoming background.
"""

BACKGROUND = 0
CANONICAL = {
    1: "L1",
    2: "L2",
    3: "L3",
    4: "L4",
    5: "L5",
    6: "T12",     # auxiliary anchor class (upper end of the lumbar sequence)
    7: "sacrum",  # auxiliary anchor class (lower end of the lumbar sequence)
    8: "L6",      # transitional, VerSe only
    9: "T13",     # transitional, VerSe only
}
NAME_TO_ID = {v: k for k, v in CANONICAL.items()}

TARGET_LEVELS = ("L1", "L2", "L3", "L4", "L5")  # what we report Dice on
TARGET_IDS = tuple(NAME_TO_ID[n] for n in TARGET_LEVELS)
NUM_TRAIN_CLASSES = 8                            # background + L1-L5 + T12 + sacrum
ROI_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9)            # structures that define the lumbar crop

# TotalSegmentator ships one binary mask per structure. S1 and sacrum are both
# merged into the sacrum anchor. Masks are merged in this order and a later mask
# wins where two overlap, so the target levels come last: where L5 touches
# S1/sacrum, the voxel stays L5 (the audit counts these overlaps).
TOTALSEG_MASKS = {
    "vertebrae_T12": 6,
    "sacrum": 7,
    "vertebrae_S1": 7,
    "vertebrae_L1": 1,
    "vertebrae_L2": 2,
    "vertebrae_L3": 3,
    "vertebrae_L4": 4,
    "vertebrae_L5": 5,
}

# VerSe ships one multi-label mask: 1-7 C1-C7, 8-19 T1-T12, 20-25 L1-L6,
# 26 sacrum, 27 coccyx, 28 T13. Everything not listed maps to background.
VERSE_LABELS = {
    20: 1, 21: 2, 22: 3, 23: 4, 24: 5,
    19: 6,   # T12
    26: 7,   # sacrum
    25: 8,   # L6
    28: 9,   # T13
}
VERSE_NAMES = (
    {i: f"C{i}" for i in range(1, 8)}
    | {i: f"T{i - 7}" for i in range(8, 20)}
    | {i: f"L{i - 19}" for i in range(20, 26)}
    | {26: "sacrum", 27: "coccyx", 28: "T13"}
)
