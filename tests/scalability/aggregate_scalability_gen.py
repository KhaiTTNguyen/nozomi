"""
Aggregate the substrate-generation scalability sweep.

Reads:
    results/runs/N<NNNNNN>.json   (written by run_scalability_gen.py)

Writes:
    results/scalability_summary.json
    results/scalability_time.png    -- Wall-clock time vs substrate side length
    results/scalability_table.csv   -- per-size numeric table (time + memory)

The figure intentionally shows ONLY the generation time vs substrate size
(clean single log-log curve). Memory and other quantities live in the table.

Usage:
    python -m tests.scalability.aggregate_scalability_gen
"""

from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.scalability.config import RESULTS_DIR, RUNS_DIR


def _collect() -> list[dict]:
    rows = []
    for p in sorted(RUNS_DIR.glob("N*.json")):
        rec = json.loads(p.read_text())
        if not rec.get("ok"):
            rows.append(rec)
            continue
        rows.append(rec)
    # Sort by box size when available, else by num_fibers.
    rows.sort(key=lambda r: (r.get("box_L_um") or 0, r.get("num_fibers") or 0))
    return rows


def _fmt_time(s) -> str:
    if s is None:
        return "-"
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s/60:.1f} min"
    return f"{s/3600:.2f} h"


def _plot_time(rows: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r.get("ok") and r.get("box_L_um")
          and (r.get("gen_elapsed_s") or r.get("wall_elapsed_s"))]
    L = [r["box_L_um"] for r in ok]
    t = [r.get("gen_elapsed_s") or r.get("wall_elapsed_s") for r in ok]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    if L:
        ax.plot(L, t, "o-", color="#1f77b4")
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "No successful runs yet.\n"
                "Run: python -m tests.scalability.run_scalability_gen --verify",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    ax.set_xlabel(r"Substrate side length ($\mu$m)")
    ax.set_ylabel("Wall-clock time (sec)")
    ax.set_title("Substrate generation time vs Substrate size")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_table(rows: list[dict], csv_path: Path) -> None:
    cols = ["num_fibers", "box_L_um", "n_spheres", "gen_elapsed_s",
            "wall_elapsed_s", "peak_gpu_alloc_bytes", "peak_gpu_reserved_bytes",
            "peak_host_rss_bytes", "output_bytes", "ok", "oom"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'N_fib':>7} {'L (um)':>9} {'spheres':>10} {'gen time':>10} "
          f"{'GPU peak':>10} {'host peak':>10} {'output':>9} {'status':>11}")
    print("-" * 92)
    for r in rows:
        if not r.get("ok"):
            status = "OOM" if r.get("oom") else "FAIL"
            print(f"{r.get('num_fibers','?'):>7} {'-':>9} {'-':>10} {'-':>10} "
                  f"{'-':>10} {'-':>10} {'-':>9} {status:>11}")
            continue
        gpu = (r.get("peak_gpu_reserved_bytes") or 0) / 1e9
        host = (r.get("peak_host_rss_bytes") or 0) / 1e9
        outmb = (r.get("output_bytes") or 0) / 1e6
        gen = _fmt_time(r.get("gen_elapsed_s") or r.get("wall_elapsed_s"))
        print(f"{r['num_fibers']:>7} {r.get('box_L_um') or 0:>9.1f} "
              f"{r.get('n_spheres') or 0:>10} {gen:>10} "
              f"{gpu:>8.2f}G {host:>8.2f}G {outmb:>7.1f}M {'ok':>11}")


def main() -> None:
    rows = _collect()

    summary_path = RESULTS_DIR / "scalability_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))

    plot_path = RESULTS_DIR / "scalability_time.png"
    _plot_time(rows, plot_path)

    csv_path = RESULTS_DIR / "scalability_table.csv"
    _write_table(rows, csv_path)

    _print_table(rows)
    print(f"\nSummary: {summary_path}")
    print(f"Plot:    {plot_path}")
    print(f"Table:   {csv_path}")


if __name__ == "__main__":
    main()
