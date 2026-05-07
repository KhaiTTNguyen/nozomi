"""
Aggregate the scalability sweep and plot runtime vs box side.

Reads:
    results/scalability/L<LL>/<fw>/run_*.json

Writes:
    results/scalability/scalability_summary.json
    results/scalability/scalability_runtime.png
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from tests.benchmark.config import RESULTS_DIR

SCALE_ROOT = RESULTS_DIR / "scalability"
FRAMEWORKS = ("nozomi", "disimpy", "camino", "mcdc")


def _collect() -> dict:
    """Return {fw: [{L_actual, n_cyl, rt_mean, rt_std, n_runs}, ...]} sorted by L."""
    data: dict = {fw: [] for fw in FRAMEWORKS}
    if not SCALE_ROOT.exists():
        return data

    for L_dir in sorted(SCALE_ROOT.glob("L*")):
        if not L_dir.is_dir():
            continue
        for fw in FRAMEWORKS:
            fw_dir = L_dir / fw
            if not fw_dir.is_dir():
                continue
            runs = []
            for p in sorted(fw_dir.glob("run_*.json")):
                with open(p) as f:
                    runs.append(json.load(f))
            if not runs:
                continue
            times = np.array([r["elapsed_s"] for r in runs])
            data[fw].append({
                "L_target_um": int(runs[0].get("scalability_L_target_um",
                                               int(L_dir.name[1:]))),
                "L_actual_um": float(runs[0].get("scalability_L_actual_um",
                                                 int(L_dir.name[1:]))),
                "n_cyl": int(runs[0].get("scalability_n_cyl", 0)),
                "rt_mean_s": float(times.mean()),
                "rt_std_s": float(times.std(ddof=1)) if len(times) > 1 else 0.0,
                "n_runs": len(runs),
                "n_walkers": int(runs[0].get("n_walkers", 0)),
            })
    for fw in FRAMEWORKS:
        data[fw].sort(key=lambda d: d["L_actual_um"])
    return data


def _plot(data: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Keep the same colour mapping used by the signal benchmark.
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {fw: color_cycle[i % len(color_cycle)]
              for i, fw in enumerate(FRAMEWORKS)}

    fig, ax = plt.subplots(figsize=(7.5, 5))

    any_points = False
    walkers_seen = set()
    for fw in FRAMEWORKS:
        pts = data.get(fw, [])
        if not pts:
            continue
        any_points = True
        L = np.array([p["L_actual_um"] for p in pts])
        mean = np.array([p["rt_mean_s"] for p in pts])
        sd = np.array([p["rt_std_s"] for p in pts])
        walkers_seen.update(p["n_walkers"] for p in pts)
        ax.errorbar(L, mean, yerr=sd, marker="o", capsize=3,
                    color=colors[fw], label=fw, linewidth=1.5)

    if not any_points:
        ax.text(0.5, 0.5, "No scalability runs found.\n"
                          "Run: python -m tests.benchmark.run_scalability",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    # O(N_cyl) = O(L^2) reference line, anchored at the median point of Nozomi
    # (or first framework) for visual comparison.
    ref_fw = next(fw for fw in FRAMEWORKS if data.get(fw))
    ref_pts = data[ref_fw]
    if len(ref_pts) >= 2:
        mid = ref_pts[len(ref_pts) // 2]
        L0, t0 = mid["L_actual_um"], mid["rt_mean_s"]
        Ls = np.array(sorted({p["L_actual_um"]
                              for fw in FRAMEWORKS for p in data.get(fw, [])}))
        ax.plot(Ls, t0 * (Ls / L0) ** 2,
                "k--", linewidth=0.8, alpha=0.6,
                label=r"$\mathcal{O}(L^2) \propto N_\mathrm{cyl}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("box side L (μm)")
    ax.set_ylabel("wall-clock time per run (sec)")
    walker_str = (f"{walkers_seen.pop():,} walkers"
                  if len(walkers_seen) == 1 else "mixed walker counts")
    ax.set_title(f"Scalability: runtime vs substrate size  |  "
                 f"d = 1.8 μm, ICVF = 0.65, {walker_str}")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _print_table(data: dict) -> None:
    print(f"\n{'framework':<10} {'L_actual (um)':>14} {'n_cyl':>8} "
          f"{'walkers':>9} {'runtime (sec)':>20}  {'repeats':>8}")
    print("-" * 78)
    for fw in FRAMEWORKS:
        for pt in data.get(fw, []):
            rt = f"{pt['rt_mean_s']:.2f} ± {pt['rt_std_s']:.2f}"
            print(f"{fw:<10} {pt['L_actual_um']:>14.2f} {pt['n_cyl']:>8} "
                  f"{pt['n_walkers']:>9} {rt:>20}  {pt['n_runs']:>8}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None,
                    help="Results folder to aggregate (defaults to "
                         "$BENCH_RESULTS_DIR or results/latest).")
    args = ap.parse_args()
    if args.run_dir:
        globals()["SCALE_ROOT"] = Path(args.run_dir).resolve() / "scalability"
    print(f"[aggregate_scalability] reading from {SCALE_ROOT}")
    SCALE_ROOT.mkdir(parents=True, exist_ok=True)
    data = _collect()

    summary_path = SCALE_ROOT / "scalability_summary.json"
    summary_path.write_text(json.dumps(data, indent=2))

    plot_path = SCALE_ROOT / "scalability_runtime.png"
    _plot(data, plot_path)

    _print_table(data)
    print(f"\nSummary: {summary_path}")
    print(f"Plot:    {plot_path}")


if __name__ == "__main__":
    main()
