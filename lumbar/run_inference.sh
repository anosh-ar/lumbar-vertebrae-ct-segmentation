#!/usr/bin/env bash
# Segment L1-L5 and find the vertebral body centres in every CT of a folder.
#
#   ./run_inference.sh <input_dir> <output_dir>
#
# Needs: Python 3.10-3.12 and internet access on the first run (to install the
# packages into a private virtual environment next to this script). No GPU and no
# Docker needed: it runs on the CPU. Tested on Windows 11 (Git Bash); the same
# commands work on Linux and macOS.
#
# Input:  CT files (*.nii.gz / *.nii, any orientation), searched recursively.
# Output: per CT <case>_lumbar_labels.nii.gz (0 background, 1-5 = L1-L5, same grid
#         as the CT) and <case>_centroids.json (vertebral body centres, RAS, mm).
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <input_dir> <output_dir>" >&2
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv_inference"

# The first Python 3.10-3.12 that really runs. Trying each one matters on Windows, where
# "python3" is often only a Microsoft Store placeholder. Override with PYTHON=/path/to/python.
find_python() {
    local candidate
    for candidate in "${PYTHON:-}" python3 python "py -3"; do
        [ -n "$candidate" ] || continue
        if $candidate -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 12))' >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    echo "error: no Python 3.10-3.12 found; install one or set PYTHON=/path/to/python" >&2
    return 1
}

if [ -x "$VENV/bin/python" ]; then
    PY="$VENV/bin/python"
elif [ -x "$VENV/Scripts/python.exe" ]; then
    PY="$VENV/Scripts/python.exe"
else
    PYTHON="$(find_python)"
    echo "[setup] creating $VENV with $PYTHON (first run only)"
    $PYTHON -m venv "$VENV"
    if [ -x "$VENV/bin/python" ]; then PY="$VENV/bin/python"; else PY="$VENV/Scripts/python.exe"; fi
    "$PY" -m pip install --quiet --upgrade pip
    # CPU build of PyTorch first, then everything else at the pinned versions
    "$PY" -m pip install --quiet torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
    "$PY" -m pip install --quiet -r "$HERE/requirements_inference.txt"
    "$PY" -m pip install --quiet --no-deps -e "$HERE"
fi

"$PY" "$HERE/scripts/05_predict_lumbar_vertebrae.py" "$1" "$2" \
    --weights "$HERE/weights/lumbar_vertebrae_resenc_unet_1.5mm.pt" --device cpu
