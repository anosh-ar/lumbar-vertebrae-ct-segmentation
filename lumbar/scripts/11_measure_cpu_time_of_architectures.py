"""Step 11 (NAS extension): measure the CPU inference time of every architecture.

What it does
  For every architecture of the search space (lumbar/architecture_search_space.py)
  1. builds the network with untrained (random) weights: the time does not depend
     on the weight values, so no training is needed for this table
  2. runs the full CPU inference of the deliverable (inference.predict_ct: resampling
     to the model grid, sliding window, clean-up, resampling back) on a few fixed CTs
  3. writes the median seconds per CT, the parameter count, the peak RAM of the
     process during inference and the GFLOPs of one forward pass of one patch

The hardware cost is measured on the real device, not estimated from FLOPs
(MnasNet, Tan et al., CVPR 2019): FLOPs miss memory traffic, resampling and the
number of sliding windows. Fixed thread count (--threads, default 8), so the
numbers are comparable between runs; nothing else should run on the machine.

Output
  outputs/nas/cpu_time_per_architecture.csv   one row per architecture (step 12 reads it)
  Safe to re-run: architectures already in the file are skipped (unless --overwrite).

    python scripts/11_measure_cpu_time_of_architectures.py
    python scripts/11_measure_cpu_time_of_architectures.py --spacings 3 --repeats 3
"""

import argparse
import ctypes
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from lumbar.architecture_search_space import (SEARCHED_SPACINGS_MM, all_architectures, architecture_name,
                                              network_settings_for, patch_size_for)
from lumbar.inference import predict_ct
from lumbar.network import build_lumbar_segmentation_network
from lumbar.sources import NAS_OUTPUTS_DIR, downloaded_sources, load_cases

CPU_TIME_TABLE = NAS_OUTPUTS_DIR / "cpu_time_per_architecture.csv"
# the CTs of the earlier CPU timing (outputs/cpu_inference_seconds_per_scan.csv):
# 1.5 mm model 22 s, 45 s and 12 s per scan, i.e. a small, a large and a short scan
TIMING_CASES = ["s1446", "sub-verse020", "sub-verse053"]


def process_ram_bytes() -> int:
    """Current RAM (resident set / working set) of this process."""
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + \
                       [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        c = Counters()
        c.cb = ctypes.sizeof(c)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p        # 64-bit handle
        kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.WorkingSetSize
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class PeakRamWatcher:
    """Highest process RAM while the with-block runs (sampled every 20 ms)."""
    def __enter__(self):
        self.peak, self._stop = process_ram_bytes(), threading.Event()

        def watch():
            while not self._stop.wait(0.02):
                self.peak = max(self.peak, process_ram_bytes())
        self._thread = threading.Thread(target=watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def gflops_per_patch(settings: dict, patch_size) -> float:
    """Floating-point operations of one forward pass of one patch, counted on the
    'meta' device (shapes only, nothing is computed). A multiply-add counts as 2."""
    try:
        with torch.device("meta"):
            net = build_lumbar_segmentation_network(settings).eval()
            x = torch.zeros((1, 1) + tuple(patch_size))
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            net(x)
        return counter.get_total_flops() / 1e9
    except Exception:                       # FLOPs are only extra information
        return float("nan")


def find_timing_cases(case_ids) -> dict:
    """{case_id: Case} for the requested ids, searched in every downloaded source."""
    found = {}
    for source in downloaded_sources():
        for case in load_cases(source):
            if case.case_id in case_ids and case.case_id not in found:
                found[case.case_id] = case
    missing = [c for c in case_ids if c not in found]
    if missing:
        raise SystemExit(f"timing CTs not found in the downloaded sources: {missing}")
    return found


def measure_architecture(arch: dict, images: dict, repeats: int) -> dict:
    settings, patch = network_settings_for(arch), patch_size_for(arch)
    torch.manual_seed(0)
    net = build_lumbar_segmentation_network(settings).eval()
    with torch.no_grad():                   # warm-up: first call allocates buffers
        net(torch.zeros((1, 1) + patch))
    row = {"architecture": architecture_name(arch), **arch,
           "features_per_stage": "-".join(map(str, settings["features_per_stage"])),
           "patch_size": "x".join(map(str, patch)),
           "parameters_millions": sum(p.numel() for p in net.parameters()) / 1e6,
           "gflops_per_patch": gflops_per_patch(settings, patch)}
    peak = 0
    for cid, img in images.items():
        times = []
        for _ in range(repeats):
            with PeakRamWatcher() as ram:
                t0 = time.perf_counter()
                predict_ct(net, img, patch, "cpu", arch["spacing_mm"])
                times.append(time.perf_counter() - t0)
            peak = max(peak, ram.peak)
        row[f"seconds_{cid}"] = float(np.median(times))
    row["cpu_seconds_per_case"] = float(np.median([row[f"seconds_{c}"] for c in images]))
    row["peak_ram_gb"] = peak / 1e9
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spacings", type=float, nargs="+", default=list(SEARCHED_SPACINGS_MM))
    ap.add_argument("--cases", nargs="+", default=TIMING_CASES, help="CTs to time (case ids)")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads (fixed, written to the table)")
    ap.add_argument("--repeats", type=int, default=1, help="runs per CT; the median is kept")
    ap.add_argument("--overwrite", action="store_true", help="measure again architectures already in the table")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    NAS_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(CPU_TIME_TABLE) if CPU_TIME_TABLE.exists() and not args.overwrite else pd.DataFrame()
    todo = [a for a in all_architectures(args.spacings)
            if done.empty or architecture_name(a) not in set(done["architecture"])]
    print(f"[cpu time] {len(todo)} architectures to measure, {args.threads} threads, "
          f"CTs {args.cases}, {args.repeats} run(s) each")
    cases = find_timing_cases(args.cases)
    images = {cid: cases[cid].load_ct() for cid in args.cases}     # read and unzipped once, outside the timing

    rows = done.to_dict("records")
    for i, arch in enumerate(todo, 1):
        row = measure_architecture(arch, images, args.repeats)
        row["threads"] = args.threads
        rows.append(row)
        pd.DataFrame(rows).to_csv(CPU_TIME_TABLE, index=False)
        print(f"   {i}/{len(todo)} {row['architecture']:<24} {row['cpu_seconds_per_case']:6.1f} s/CT  "
              f"{row['parameters_millions']:6.1f} M params  {row['gflops_per_patch']:7.0f} GFLOPs/patch  "
              f"peak RAM {row['peak_ram_gb']:.1f} GB", flush=True)
    print(f"[cpu time] table -> {CPU_TIME_TABLE}")


if __name__ == "__main__":
    main()
