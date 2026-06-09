#!/usr/bin/env python3
"""
summarize_param_table.py
========================
Step 2 of the aim2-prep parameter-table build.

Read the per-substrate CSV produced by ``extract_substrate_params.py`` and
aggregate the achieved parameters into one row per design group
``(designed_diam, designed_K)``. For each group the mean +/- SD across the
substrate replicates is computed for:

  - achieved Vin (avf)
  - achieved axon diameter
  - achieved global ODI

SD conventions (per the build plan):
  - achieved diameter : mean of the per-substrate means +/- mean of the
    per-substrate stds (the diameter spread is itself a per-substrate quantity,
    so its group value is the average of the substrate stds).
  - avf / global ODI : these are single numbers per substrate,
    so the group value is mean +/- SD across substrates.

Outputs (written next to the input CSV by default):
  - param_table_summary.csv : machine-readable, separate mean & sd columns.
  - param_table.md          : Markdown table mirroring the target figure.

Usage
-----
    python summarize_param_table.py
    python summarize_param_table.py --input substrate_params.csv \
        --summary-csv param_table_summary.csv --markdown param_table.md
"""

import argparse
import csv
import math
from pathlib import Path

_HERE = Path(__file__).resolve()
_BUILD_DIR = _HERE.parent
_DEFAULT_INPUT = _BUILD_DIR / "substrate_params.csv"
_DEFAULT_SUMMARY_CSV = _BUILD_DIR / "param_table_summary.csv"
_DEFAULT_MARKDOWN = _BUILD_DIR / "param_table.md"

SUMMARY_FIELDS = [
    "designed_diam",
    "designed_diam_sd",
    "designed_K",
    "designed_ODI",
    "n_substrates",
    "achieved_avf_mean",
    "achieved_avf_sd",
    "achieved_diam_mean",
    "achieved_diam_sd",
    "achieved_global_ODI_mean",
    "achieved_global_ODI_sd",
]


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def _sample_sd(values):
    """Sample standard deviation (ddof=1). NaN for <2 values."""
    n = len(values)
    if n < 2:
        return float("nan")
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def _read_rows(input_path: Path):
    with input_path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _to_float(row, key):
    val = row.get(key, "")
    if val is None or val == "":
        return None
    return float(val)


def summarize(rows):
    """Group rows by (designed_diam, designed_K) and compute mean/sd stats."""
    groups = {}
    for row in rows:
        diam = _to_float(row, "designed_diam")
        k = row.get("designed_K", "")
        key = (diam, int(float(k)) if k not in ("", None) else None)
        groups.setdefault(key, []).append(row)

    summary = []
    for (diam, k), grp_rows in sorted(groups.items(),
                                      key=lambda kv: (kv[0][0] if kv[0][0] is not None else 0,
                                                      -(kv[0][1] if kv[0][1] is not None else 0))):
        diam_means = [_to_float(r, "achieved_diam_mean") for r in grp_rows]
        diam_means = [v for v in diam_means if v is not None]
        diam_stds = [_to_float(r, "achieved_diam_std") for r in grp_rows]
        diam_stds = [v for v in diam_stds if v is not None]
        avf = [v for v in (_to_float(r, "achieved_avf") for r in grp_rows) if v is not None]
        g_odi = [v for v in (_to_float(r, "achieved_global_ODI") for r in grp_rows) if v is not None]

        # Designed metadata is constant within a group; take it from the first row.
        first = grp_rows[0]

        summary.append({
            "designed_diam": diam,
            "designed_diam_sd": _to_float(first, "designed_diam_sd"),
            "designed_K": k,
            "designed_ODI": _to_float(first, "designed_ODI"),
            "n_substrates": len(grp_rows),
            "achieved_avf_mean": _mean(avf),
            "achieved_avf_sd": _sample_sd(avf),
            "achieved_diam_mean": _mean(diam_means),
            "achieved_diam_sd": _mean(diam_stds),
            "achieved_global_ODI_mean": _mean(g_odi),
            "achieved_global_ODI_sd": _sample_sd(g_odi),
        })
    return summary


def write_summary_csv(summary, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def _fmt(mean, sd, places=3):
    if mean != mean:  # NaN
        return "-"
    if sd != sd:  # NaN (single substrate)
        return f"{mean:.{places}f}"
    return f"{mean:.{places}f} \u00b1 {sd:.{places}f}"


def write_markdown(summary, output_path: Path):
    headers = [
        "Designed Axon diameter (um)",
        "Designed Orientation dispersion (K)",
        "Designed ODI",
        "N substrates",
        "Achieved Vin (avf)",
        "Achieved Axon diameter (um)",
        "Achieved Global ODI",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in summary:
        designed_diam = (f"{r['designed_diam']:.2f} \u00b1 {r['designed_diam_sd']:.2f}"
                         if r["designed_diam"] is not None and r["designed_diam_sd"] is not None
                         else "-")
        designed_k = str(r["designed_K"]) if r["designed_K"] is not None else "-"
        designed_odi = (f"{r['designed_ODI']:.4f}" if r["designed_ODI"] is not None else "-")
        cells = [
            designed_diam,
            designed_k,
            designed_odi,
            str(r["n_substrates"]),
            _fmt(r["achieved_avf_mean"], r["achieved_avf_sd"], places=3),
            _fmt(r["achieved_diam_mean"], r["achieved_diam_sd"], places=3),
            _fmt(r["achieved_global_ODI_mean"], r["achieved_global_ODI_sd"], places=4),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT,
                        help=f"Per-substrate CSV from step 1 (default: {_DEFAULT_INPUT})")
    parser.add_argument("--summary-csv", type=Path, default=_DEFAULT_SUMMARY_CSV,
                        help=f"Output summary CSV (default: {_DEFAULT_SUMMARY_CSV})")
    parser.add_argument("--markdown", type=Path, default=_DEFAULT_MARKDOWN,
                        help=f"Output Markdown table (default: {_DEFAULT_MARKDOWN})")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input CSV does not exist: {args.input}  (run extract_substrate_params.py first)")

    rows = _read_rows(args.input)
    summary = summarize(rows)
    write_summary_csv(summary, args.summary_csv)
    write_markdown(summary, args.markdown)

    print(f"Input        : {args.input}  ({len(rows)} substrate rows)")
    print(f"Wrote        : {args.summary_csv}  ({len(summary)} groups)")
    print(f"Wrote        : {args.markdown}")


if __name__ == "__main__":
    main()
