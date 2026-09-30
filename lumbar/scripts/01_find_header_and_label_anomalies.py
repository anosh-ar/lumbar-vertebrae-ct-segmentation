"""Step 1 (Part A): check every case for header, intensity and label anomalies.

What it does
  For each case: read the CT header and voxels, merge the label files, and test
  for every anomaly listed in lumbar/audit.py (ANOMALIES). Nothing is changed
  on disk; this step only measures and reports.

Output (outputs/qc/)
  anomalies_per_case_<source>.csv   one row per case: measurements + anomaly codes
  anomaly_report.md                 the readable summary for the report, rebuilt
                                    from every anomalies_per_case_*.csv present

Sources: all downloaded ones by default (TotalSegmentator v3 train + test,
read from the zip, and the extracted VerSe splits). Step 3 needs these CSVs.
Takes about 1.5 s per case with 3 workers (~1 h for the full v3 dataset).

    python scripts/01_find_header_and_label_anomalies.py
    python scripts/01_find_header_and_label_anomalies.py --source verse19_test --limit 5
"""

import argparse
import ast
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from lumbar.audit import ANOMALIES, audit_case_for_anomalies
from lumbar.sources import QC_REPORTS_DIR, downloaded_sources, load_cases

CSV_PREFIX = "anomalies_per_case_"
REPORT_NAME = "anomaly_report.md"


def audit_case_without_crashing(case):
    try:
        return audit_case_for_anomalies(case)
    except Exception as e:  # noqa: BLE001 - report, never abort the whole run
        return {"case_id": case.case_id, "dataset": case.dataset, "split": case.split,
                "subject_id": case.subject_id, "flags": "LOAD_ERROR", "error": repr(e)}


def audit_all_cases_of_source(source: str, workers: int, limit: int | None) -> pd.DataFrame:
    cases = load_cases(source)[:limit]
    print(f"[anomalies] {source}: {len(cases)} cases, {workers} workers")
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(workers) as ex:
        futures = [ex.submit(audit_case_without_crashing, c) for c in cases]
        for i, f in enumerate(as_completed(futures), 1):
            rows.append(f.result())
            if i % 10 == 0 or i == len(cases):
                print(f"   {i}/{len(cases)}  {time.time() - t0:.0f}s")
    df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    df.to_csv(QC_REPORTS_DIR / f"{CSV_PREFIX}{source}.csv", index=False)
    return df


def _flags_per_case(df):
    return df["flags"].fillna("").str.split()


def build_anomaly_report_markdown(dfs: dict[str, pd.DataFrame]) -> str:
    srcs = list(dfs)
    L = ["# Part A: header and label anomalies", "",
         "Made by `scripts/01_find_header_and_label_anomalies.py`. Numbers count **cases**; "
         "one case can have several anomalies. Per-case details are in `anomalies_per_case_<dataset>.csv`.", ""]

    L += ["## Datasets checked", "",
          "| Dataset | Cases | Stored orientation (cases) | Largest voxel side in mm: min / median / max "
          "| Header orientation fields set, qform/sform code (cases) | Stored CT data type (cases) |",
          "|---|---|---|---|---|---|"]
    for src, df in dfs.items():
        ok = df[df["flags"].fillna("") != "LOAD_ERROR"]
        orient = ", ".join(f"{k} {v}" for k, v in ok["img_axcodes"].value_counts().items())
        big = ok["img_zooms"].map(lambda z: max(ast.literal_eval(z)) if isinstance(z, str) else max(z))
        codes = ", ".join(f"{q}/{s} {n}" for (q, s), n in
                          ok.groupby(["img_qform_code", "img_sform_code"]).size().items())
        dtypes = ", ".join(f"{k} {v}" for k, v in ok["img_dtype"].value_counts().items())
        L.append(f"| {src} | {len(df)} | {orient} | {big.min():.2f} / {big.median():.2f} / {big.max():.2f} "
                 f"| {codes} | {dtypes} |")
    L += ["", "Orientation letters say where each array axis points, e.g. RAS = Right, Anterior, Superior. "
          "A code of 0 means the field is not set.", ""]

    counts = {src: pd.Series([f for fs in _flags_per_case(df) for f in fs]).value_counts()
              for src, df in dfs.items()}
    found = [c for c in ANOMALIES if any(c in cnt.index for cnt in counts.values())]
    L += ["## Anomalies found", "",
          "| Problem | Code | " + " | ".join(srcs) + " | How we find it | What we do about it |",
          "|---|---|" + "---|" * len(srcs) + "---|---|"]
    for code in found:
        a = ANOMALIES[code]
        n = " | ".join(str(int(counts[s].get(code, 0))) for s in srcs)
        L.append(f"| {a.problem} | `{code}` | {n} | {a.how_found} | {a.what_we_do} |")
    L += ["", "\"Plan (Part D)\" marks handling that belongs to the evaluation step and is not built yet.", ""]

    not_found = [c for c in ANOMALIES if c not in found]
    if not_found:
        L += ["Checked but not found in any case:", ""]
        L += [f"- {ANOMALIES[c].problem} (`{c}`)" for c in not_found]
        L.append("")

    L += ["## Example cases", ""]
    for code in found:
        ex = []
        for src, df in dfs.items():
            ids = df.loc[_flags_per_case(df).map(lambda fs: code in fs), "case_id"].tolist()
            if ids:
                ex.append(f"{src}: {', '.join(ids[:6])}{' ...' if len(ids) > 6 else ''}")
        L.append(f"- `{code}`: " + "; ".join(ex))
    L.append("")

    L += ["## How much of the lumbar spine each scan shows", "",
          "| Dataset | All of L1-L5 | Some of L1-L5 | None |", "|---|---|---|---|"]
    for src, df in dfs.items():
        n = df["n_lumbar"].fillna(0)
        L.append(f"| {src} | {(n == 5).sum()} | {((n > 0) & (n < 5)).sum()} | {(n == 0).sum()} |")
    L.append("")

    for src, df in dfs.items():
        if "ctd_dist_med_vox_hyp" not in df:
            continue
        d = df.dropna(subset=["ctd_dist_med_vox_hyp"])
        aniso = d[d["img_aniso"] > 1.05]
        L += [f"## What the VerSe centroid annotations mean ({src})", "",
              "VerSe gives one annotated point per vertebra. The file does not say which unit the numbers "
              "are in, so we tried both readings and measured how far each lands from the centre of the "
              "same vertebra's mask (median per case, then median over cases). The two readings only "
              "differ on scans with non-cubic voxels, so those are shown separately.", "",
              "| Reading of the numbers | Distance, all cases (n) | Distance, non-cubic voxels (n) |",
              "|---|---|---|",
              f"| Voxel indices, after turning the image to the file's `direction` "
              f"| {d['ctd_dist_med_vox_hyp'].median():.1f} mm ({len(d)}) "
              f"| {aniso['ctd_dist_med_vox_hyp'].median():.1f} mm ({len(aniso)}) |",
              f"| Millimetres along those axes | {d['ctd_dist_med_mm_hyp'].median():.1f} mm ({len(d)}) "
              f"| {aniso['ctd_dist_med_mm_hyp'].median():.1f} mm ({len(aniso)}) |", "",
              f"The annotated point lies a median {d['ctd_anterior_offset_med'].median():+.1f} mm in front of "
              "the whole-vertebra centre (+ = towards the belly), i.e. it marks the vertebral body, "
              "not the whole vertebra.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="+", help="default: every downloaded source")
    ap.add_argument("--workers", type=int, default=3, help="parallel processes; each holds one full CT in memory")
    ap.add_argument("--limit", type=int, help="only the first N cases per source, for quick tests")
    args = ap.parse_args()
    QC_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for src in args.source or downloaded_sources():
        audit_all_cases_of_source(src, args.workers, args.limit)

    dfs = {p.stem[len(CSV_PREFIX):]: pd.read_csv(p) for p in sorted(QC_REPORTS_DIR.glob(f"{CSV_PREFIX}*.csv"))}
    (QC_REPORTS_DIR / REPORT_NAME).write_text(build_anomaly_report_markdown(dfs), encoding="utf-8")
    print(f"[anomalies] report -> {QC_REPORTS_DIR / REPORT_NAME}")


if __name__ == "__main__":
    main()
