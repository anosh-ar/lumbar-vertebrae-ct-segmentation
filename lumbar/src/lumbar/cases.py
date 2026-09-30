"""Case discovery and loading for TotalSegmentator and VerSe.

A ``Case`` is a small picklable record (paths, not data) so it can be sent to
worker processes. Files are read through a ``FolderOrZipReader``, so the full
TotalSegmentator v3 archive is read in place and never extracted.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath

import nibabel as nib

from .labels import TOTALSEG_MASKS, VERSE_LABELS


# --------------------------------------------------------------------------- storage

class FolderOrZipReader:
    """Read-only view of a dataset root: a directory or a .zip archive."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.is_zip = self.root.suffix.lower() == ".zip"
        self._zip = zipfile.ZipFile(self.root) if self.is_zip else None

    def list_files(self) -> list[str]:
        if self.is_zip:
            return [n for n in self._zip.namelist()
                    if not n.endswith("/") and not n.startswith("__MACOSX/")]
        return [p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()]

    def read_file_bytes(self, rel: str) -> bytes:
        if self.is_zip:
            return self._zip.read(rel)
        return (self.root / rel).read_bytes()

    def load_nifti(self, rel: str) -> nib.Nifti1Image:
        if not self.is_zip:
            return nib.load(self.root / rel)
        data = self.read_file_bytes(rel)
        if rel.endswith(".gz"):
            data = gzip.decompress(data)
        return nib.Nifti1Image.from_bytes(data)


@lru_cache(maxsize=8)
def open_folder_or_zip(root: str) -> FolderOrZipReader:
    """One reader per root per process (zip handles are not picklable)."""
    return FolderOrZipReader(root)


# --------------------------------------------------------------------------- case

@dataclass(frozen=True)
class LabelFile:
    name: str                                # e.g. "vertebrae_L3" or "seg-vert_msk"
    path: str                                # relative to the dataset root
    value_to_canonical_id: dict[int, int]    # raw label value -> canonical id


@dataclass(frozen=True)
class Case:
    case_id: str
    dataset: str               # "totalseg" | "verse"
    split: str                 # official split name
    root: str                  # dataset folder or zip
    image_path: str
    label_files: tuple[LabelFile, ...]
    centroid_path: str | None = None
    subject_id: str | None = None
    info: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def reader(self) -> FolderOrZipReader:
        return open_folder_or_zip(self.root)

    def load_ct(self) -> nib.Nifti1Image:
        return self.reader.load_nifti(self.image_path)

    def load_label_file(self, label_file: LabelFile) -> nib.Nifti1Image:
        return self.reader.load_nifti(label_file.path)

    def load_verse_centroids(self) -> dict | None:
        """VerSe centroid file: {"direction": ("P","I","R"), "points": {label: (x,y,z)}}."""
        if not self.centroid_path:
            return None
        raw = json.loads(self.reader.read_file_bytes(self.centroid_path))
        direction, points = None, {}
        for entry in raw:
            if "direction" in entry:
                direction = tuple(entry["direction"])
            elif "label" in entry:
                points[int(entry["label"])] = (entry["X"], entry["Y"], entry["Z"])
        return {"direction": direction, "points": points}


# --------------------------------------------------------------------------- TotalSegmentator

def find_totalsegmentator_cases(root: str | Path, splits: set[str] | None = None) -> list[Case]:
    """Cases listed in meta.csv whose CT exists. Works on an extracted
    folder or directly on the downloaded zip (with or without a top folder)."""
    reader = open_folder_or_zip(str(root))
    names = reader.list_files()
    meta_rel = next(n for n in names if PurePosixPath(n).name == "meta.csv")
    prefix = meta_rel[: -len("meta.csv")]            # "" or "Totalsegmentator_dataset_v300/"
    present = set(names)

    text = reader.read_file_bytes(meta_rel).decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text), delimiter=";"))

    cases = []
    for row in rows:
        sid, split = row["image_id"], row["split"]
        if splits and split not in splits:
            continue
        image = f"{prefix}{sid}/ct.nii.gz"
        if image not in present:
            continue
        label_files = tuple(
            LabelFile(stem, f"{prefix}{sid}/segmentations/{stem}.nii.gz", {1: cid})
            for stem, cid in TOTALSEG_MASKS.items()
            if f"{prefix}{sid}/segmentations/{stem}.nii.gz" in present
        )
        cases.append(Case(sid, "totalseg", split, str(root), image, label_files,
                          subject_id=sid, info=dict(row)))
    return cases


# --------------------------------------------------------------------------- VerSe

def find_verse_cases(root: str | Path, split: str) -> list[Case]:
    """VerSe BIDS-like layout: rawdata/<sub>/<case>_ct.nii.gz and
    derivatives/<sub>/<case>_seg-vert_msk.nii.gz (+ _seg-subreg_ctd.json).
    <case> may carry a split-id, e.g. sub-verse400_split-verse090, when one
    subject was scanned in several pieces."""
    reader = open_folder_or_zip(str(root))
    names = reader.list_files()
    present = set(names)
    cases = []
    for rel in sorted(names):
        p = PurePosixPath(rel)
        if not p.name.endswith("_ct.nii.gz") or "rawdata" not in p.parts:
            continue
        case_id = p.name[: -len("_ct.nii.gz")]
        subject = p.parent.name
        base = PurePosixPath(*p.parts[: p.parts.index("rawdata")])
        deriv = base / "derivatives" / subject
        msk = (deriv / f"{case_id}_seg-vert_msk.nii.gz").as_posix()
        ctd = (deriv / f"{case_id}_seg-subreg_ctd.json").as_posix()
        if msk not in present:
            continue
        sidecar = rel.replace(".nii.gz", ".json")
        info = json.loads(reader.read_file_bytes(sidecar)) if sidecar in present else {}
        cases.append(Case(case_id, "verse", split, str(root), rel,
                          (LabelFile("seg-vert_msk", msk, dict(VERSE_LABELS)),),
                          centroid_path=ctd if ctd in present else None,
                          subject_id=subject, info=info))
    return cases
