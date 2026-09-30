"""Step 0: unzip the downloaded VerSe archives.

What it does
  data/archives/dataset-verse19<split>.zip  ->  data/raw/verse19/<split>/
  (CT, mask, centroid file, preview PNG; the __MACOSX junk in the zips is skipped)

TotalSegmentator v3 is NOT extracted: its 35 GB zip is read in place by the
later steps. The old TotalSegmentator small subset (v2.0.1 labels) is ignored,
because the full v3 dataset replaces it.

Safe to re-run: files that already exist with the right size are skipped.

    python scripts/00_extract_verse_archives.py
"""

import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

from lumbar.sources import ARCHIVES_DIR, EXTRACTED_DIR


def extract_zip_members(zip_path: Path, dest: Path) -> tuple[int, int]:
    """Extract every file below the zip's top folder into ``dest``.
    Returns (files written, files already present)."""
    written = skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/"):
                continue
            parts = PurePosixPath(info.filename).parts
            if len(parts) < 2:
                continue
            out = dest / PurePosixPath(*parts[1:])      # drop the zip's top folder
            if out.exists() and out.stat().st_size == info.file_size:
                skipped += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(out, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            written += 1
    return written, skipped


def main():
    jobs = []
    for zp in sorted(ARCHIVES_DIR.glob("*.zip")):
        if m := re.fullmatch(r"dataset-verse(19|20)(training|validation|test)\.zip", zp.name):
            jobs.append((zp, EXTRACTED_DIR / f"verse{m[1]}" / m[2]))
        else:
            print(f"[skip] {zp.name}: not a VerSe archive (TotalSegmentator v3 is read from its zip)")
    if not jobs:
        sys.exit(f"no VerSe archives in {ARCHIVES_DIR}")
    for zp, dest in jobs:
        print(f"[extract] {zp.name} -> {dest}")
        w, s = extract_zip_members(zp, dest)
        print(f"          {w} written, {s} already present")


if __name__ == "__main__":
    main()
