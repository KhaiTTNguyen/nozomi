"""
Aggregate results across frameworks and repeats.

Reads results/<framework>/run_XX.json, computes:
  - mean +/- std runtime per framework
  - per-b-value normalized RMSE vs Camino:
        nRMSE(b) = |S_fw(b) - S_camino(b)| / |S_camino(b)|
  - writes summary JSON and a signal-vs-b plot.

Usage:
    python -m tests.benchmark.aggregate
"""

from __future__ import annotations
import dataclasses
import json
from pathlib import Path
import numpy as np

from tests.benchmark.config import CFG, RESULTS_DIR, ensure_dirs, num_steps, gmax_T_per_m_for_bvalue_s_mm2
from tests.benchmark.analytic import vangelderen_perp


FRAMEWORKS = ("nozomi", "disimpy", "camino", "mcdc")


def _jsonable_cfg(cfg=CFG) -> dict:
    payload = {}
    for key, value in dataclasses.asdict(cfg).items():
        if key in ("camino_bin", "mcdc_bin"):
            continue
        payload[key] = list(value) if isinstance(value, tuple) else value
    return payload


def _load_cfg_for_results(run_dir: Path, fallback=CFG):
    """Prefer the config captured inside a timestamped result folder."""
    meta_path = run_dir / "substrate_cache" / "substrate.json"
    if not meta_path.exists():
        print(f"[warn] no run-local substrate metadata at {meta_path}; using active CFG")
        return fallback

    with open(meta_path) as f:
        payload = json.load(f)
    raw_cfg = payload.get("config", {})
    valid_fields = {field.name for field in dataclasses.fields(fallback)}
    updates = {}
    for key, value in raw_cfg.items():
        if key not in valid_fields or key in ("camino_bin", "mcdc_bin"):
            continue
        if key in ("gradient_axis", "bvals_s_mm2") and isinstance(value, list):
            value = tuple(value)
        updates[key] = value
    return dataclasses.replace(fallback, **updates)


def _load_runs(framework: str) -> list[dict]:
    runs = []
    for p in sorted((RESULTS_DIR / framework).glob("run_*.json")):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def _stack_signals(runs: list[dict]) -> np.ndarray:
    if not runs:
        return np.zeros((0, 0))
    return np.vstack([np.asarray(r["signal"]) for r in runs])


def aggregate(cfg=CFG) -> dict:
    per_fw = {}
    for fw in FRAMEWORKS:
        runs = _load_runs(fw)
        if not runs:
            continue
        times = np.array([r["elapsed_s"] for r in runs])
        sigs = _stack_signals(runs)
        per_fw[fw] = {
            "n_runs": len(runs),
            "runtime_mean_s": float(times.mean()),
            "runtime_std_s": float(times.std(ddof=1)) if len(times) > 1 else 0.0,
            "signal_mean": sigs.mean(axis=0).tolist(),
            "signal_std": (sigs.std(axis=0, ddof=1).tolist()
                           if sigs.shape[0] > 1 else [0.0] * sigs.shape[1]),
        }

    # nRMSE vs Camino (per b-value, across run-averaged signals).
    summary = {
        "bvals_s_mm2": list(cfg.bvals_s_mm2),
        "config": _jsonable_cfg(cfg),
        "per_framework": per_fw,
    }
    if "camino" in per_fw:
        ref = np.asarray(per_fw["camino"]["signal_mean"])
        ref_safe = np.where(np.abs(ref) > 1e-12, ref, 1.0)
        nrmse = {}
        for fw, info in per_fw.items():
            if fw == "camino":
                continue
            s = np.asarray(info["signal_mean"])
            nrmse[fw] = (np.abs(s - ref) / np.abs(ref_safe)).tolist()
        summary["nRMSE_vs_camino_per_b"] = nrmse
    return summary


def _plot(summary: dict, out_path: Path, cfg=CFG) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bvals = np.asarray(summary["bvals_s_mm2"], dtype=float)
    per_fw = summary["per_framework"]
    fws = list(per_fw.keys())

    # Consistent colors across panels.
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {fw: color_cycle[i % len(color_cycle)] for i, fw in enumerate(fws)}

    has_ref = "camino" in per_fw
    if has_ref:
        ref_mean = np.asarray(per_fw["camino"]["signal_mean"])
        ref_std = np.asarray(per_fw["camino"]["signal_std"])

    # Analytic van Gelderen perpendicular signal at the same b-values.
    analytic = np.array([
        vangelderen_perp(
            gmax_T_per_m_for_bvalue_s_mm2(float(b), cfg.delta_s, cfg.Delta_s),
            cfg.cylinder_radius_m, cfg.D0_m2_s, cfg.delta_s, cfg.Delta_s,
        )
        for b in bvals
    ])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_sig, ax_log, ax_res, ax_bar = axes.flat

    # --- (a) S(b) vs b ---
    ax_sig.plot(bvals, analytic, color="k", linestyle="--", linewidth=1.5,
                marker="x", markersize=6, label="van Gelderen (analytic)", zorder=5)
    for fw in fws:
        info = per_fw[fw]
        s = np.asarray(info["signal_mean"])
        e = np.asarray(info["signal_std"])
        ax_sig.errorbar(bvals, s, yerr=e, marker="o", capsize=3,
                        color=colors[fw], label=fw)
    ax_sig.set_xlabel("b-value (s/mm²)")
    ax_sig.set_ylabel("S(b) / S(0)")
    ax_sig.set_title("(a) PGSE signal (mean ± SD)")
    ax_sig.grid(True, linestyle="--", alpha=0.4)
    ax_sig.legend()

    # --- (b) ln S(b) vs b (drop b = 0) ---
    mask = bvals > 0
    a_safe = np.where(analytic > 1e-9, analytic, 1e-9)
    ax_log.plot(bvals[mask], np.log(a_safe[mask]), color="k", linestyle="--",
                linewidth=1.5, marker="x", markersize=6,
                label="van Gelderen (analytic)", zorder=5)
    for fw in fws:
        info = per_fw[fw]
        s = np.asarray(info["signal_mean"])
        e = np.asarray(info["signal_std"])
        # SD of ln S ≈ SD(S) / S  (first-order)
        s_safe = np.where(s > 1e-9, s, 1e-9)
        ln_s = np.log(s_safe)
        ln_e = e / s_safe
        ax_log.errorbar(bvals[mask], ln_s[mask], yerr=ln_e[mask],
                        marker="o", capsize=3, color=colors[fw], label=fw)
    ax_log.set_xlabel("b-value (s/mm²)")
    ax_log.set_ylabel("ln [S(b) / S(0)]")
    ax_log.set_title("(b) semilog view")
    ax_log.grid(True, linestyle="--", alpha=0.4)
    ax_log.legend()

    # --- (c) Residual vs Camino: S_fw - S_camino ---
    if has_ref:
        ax_res.axhline(0.0, color="k", linewidth=0.8)
        for fw in fws:
            if fw == "camino":
                continue
            info = per_fw[fw]
            s = np.asarray(info["signal_mean"])
            e = np.asarray(info["signal_std"])
            # Combined SD for the difference (assume independent).
            e_diff = np.sqrt(e ** 2 + ref_std ** 2)
            ax_res.errorbar(bvals, s - ref_mean, yerr=e_diff,
                            marker="o", capsize=3, color=colors[fw], label=fw)
        # MC noise reference band: ±1/sqrt(N) for b = 0 walker-count.
        n_walk = per_fw.get("camino", {}).get("n_walkers_ref", None)
        ax_res.set_xlabel("b-value (s/mm²)")
        ax_res.set_ylabel("S_fw − S_Camino")
        ax_res.set_title("(c) residual vs Camino")
        ax_res.grid(True, linestyle="--", alpha=0.4)
        ax_res.legend()
    else:
        ax_res.text(0.5, 0.5, "Camino results unavailable",
                    ha="center", va="center", transform=ax_res.transAxes)
        ax_res.set_axis_off()

    # --- (d) nRMSE vs Camino, bar chart (averaged over b > 0) ---
    if has_ref and "nRMSE_vs_camino_per_b" in summary:
        nrmse_per_b = summary["nRMSE_vs_camino_per_b"]
        fw_names = list(nrmse_per_b.keys())
        # Mean over b > 0 (b = 0 is always 1.0 after normalization).
        mean_nrmse = []
        for fw in fw_names:
            vals = np.asarray(nrmse_per_b[fw])
            mean_nrmse.append(float(vals[mask].mean()) if mask.any() else float(vals.mean()))
        xs = np.arange(len(fw_names))
        bars = ax_bar.bar(xs, mean_nrmse,
                          color=[colors[fw] for fw in fw_names])
        ax_bar.set_xticks(xs)
        ax_bar.set_xticklabels(fw_names)
        ax_bar.set_ylabel("mean nRMSE vs Camino (b > 0)")
        ax_bar.set_title("(d) framework agreement")
        ax_bar.grid(True, linestyle="--", alpha=0.4, axis="y")
        for rect, v in zip(bars, mean_nrmse):
            ax_bar.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=9)
    else:
        ax_bar.set_axis_off()

    fig.suptitle(
        "Monte-Carlo diffusion simulator benchmark "
        f"(parallel cylinders, r = {cfg.cylinder_radius_m * 1e6:.2f} μm, "
        f"ICVF = {cfg.icvf:.2f})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_runtime(summary: dict, out_path: Path, cfg=CFG) -> None:
    """Wall-clock runtime comparison (horizontal bar, log-x, mean ± SD)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_fw = summary["per_framework"]
    if not per_fw:
        return None

    # Consistent colour mapping with the signal plot.
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fws_signal_order = list(per_fw.keys())
    colors = {fw: color_cycle[i % len(color_cycle)]
              for i, fw in enumerate(fws_signal_order)}

    # Sort frameworks slowest -> fastest (slowest on top of the bar chart).
    fws = sorted(per_fw.keys(),
                 key=lambda fw: per_fw[fw]["runtime_mean_s"],
                 reverse=True)
    means = np.array([per_fw[fw]["runtime_mean_s"] for fw in fws])
    stds = np.array([per_fw[fw]["runtime_std_s"] for fw in fws])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(fws))
    ax.barh(y, means, xerr=stds, color=[colors[fw] for fw in fws],
            ecolor="k", capsize=4, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(fws)
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock time per run (sec)")
    ax.set_title(f"Runtime  |  {int(cfg.n_walkers)} walkers × "
                 f"{num_steps(cfg)} steps, mean ± SD over "
                 f"{per_fw[fws[0]]['n_runs']} repeats")
    ax.grid(True, axis="x", which="both", linestyle="--", alpha=0.4)
    # Raw-time labels next to each bar.
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m, i, f"  {m:.2f} ± {s:.2f} s",
                va="center", ha="left", fontsize=10)
    # Headroom for right-side labels.
    ax.set_xlim(right=ax.get_xlim()[1] * 4.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _print_table(summary: dict) -> None:
    fws = list(summary["per_framework"].keys())
    bvals = summary["bvals_s_mm2"]
    hdr = f"{'framework':<10} {'runtime_s':>18}   " + "  ".join(
        f"S(b={int(b)})" for b in bvals)
    print(hdr)
    print("-" * len(hdr))
    for fw in fws:
        info = summary["per_framework"][fw]
        rt = f"{info['runtime_mean_s']:.2f} ± {info['runtime_std_s']:.2f}"
        sigs = "  ".join(f"{v:>8.4f}" for v in info["signal_mean"])
        print(f"{fw:<10} {rt:>18}   {sigs}")
    if "nRMSE_vs_camino_per_b" in summary:
        print("\nnRMSE vs Camino (per b):")
        for fw, vals in summary["nRMSE_vs_camino_per_b"].items():
            print(f"  {fw:<10} " + "  ".join(f"{v:.4f}" for v in vals))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None,
                    help="Results folder to aggregate (defaults to "
                         "$BENCH_RESULTS_DIR or results/latest).")
    args = ap.parse_args()
    if args.run_dir:
        globals()["RESULTS_DIR"] = Path(args.run_dir).resolve()
    ensure_dirs()
    print(f"[aggregate] reading from {RESULTS_DIR}")
    cfg = _load_cfg_for_results(RESULTS_DIR)
    summary = aggregate(cfg)
    out_json = RESULTS_DIR / "summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    try:
        _plot(summary, RESULTS_DIR / "signal_vs_b.png", cfg)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] signal plot failed: {exc}")
    try:
        _plot_runtime(summary, RESULTS_DIR / "runtime_comparison.png", cfg)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] runtime plot failed: {exc}")
    _print_table(summary)
    print(f"\nSummary written to {out_json}")


if __name__ == "__main__":
    main()
