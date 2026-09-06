#!/usr/bin/env python
"""Regenerate the four ``figs/substrate_stats`` plot types for previously
generated substrates, on data-driven shared axes.

For every substrate under the given root(s) this script (re)produces:

    1. ODI (global)          - OD_histogram_global_*, FOD_3D_glyph_global_watson_*
    2. CV                    - CV_{outer,inner}_diameter_*
    3. Diameter distribution - {outer,inner}_Diameter_distribution_*
    4. Along-axon variation  - {outer,inner}_Along_axon_radius_variation_*

The CV and diameter-distribution histograms are drawn on *shared* axes so
substrates are directly comparable. The axis limits are computed in a first
pass over the whole dataset (covering the maximum CV / diameter and the maximum
histogram density observed, plus headroom) and persisted to a JSON. A second
pass re-plots every substrate using those limits.

Usage
-----
    cd nozomi
    sim_venv/bin/python3 -m simulation_toolkit.cli.reprocess_substrate_stats \\
        experiment/aim2-prep/data_full

    # reuse previously computed limits instead of rescanning
    sim_venv/bin/python3 -m simulation_toolkit.cli.reprocess_substrate_stats \\
        experiment/aim2-prep/data_full --use-existing-limits

    # only compute + write the shared-limits JSON (no plotting)
    sim_venv/bin/python3 -m simulation_toolkit.cli.reprocess_substrate_stats \\
        experiment/aim2-prep/data_full --only-limits

Notes
-----
* Because the shared axes depend on the whole dataset, the plotting pass
  re-plots *all* substrates by default so every figure uses identical axes.
  ``--skip-existing`` may be used for incremental backfills when the limits are
  known to be unchanged.
* Pre-existing stale/old-variant substrate_stats plots are deleted before
  regeneration; ``Watson_samples_kappa_*.png`` (written at initialization) is
  preserved.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Headless plotting: force Agg before the plotting utilities import pyplot.
import matplotlib
matplotlib.use("Agg")

# Make the repository importable regardless of the cwd.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent  # .../nozomi
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch
from matplotlib.pyplot import cm

import simulation_toolkit.toolkit_params as config_params
import simulation_toolkit.utils.common_utils as common_util
from simulation_toolkit.utils import along_fiber_plot
from simulation_toolkit.utils import orientation_plot


_K_PATTERN = re.compile(r"_K(\d+)")
_TS_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}(?:-\d{2})?)")

_DEFAULT_LIMITS_FILENAME = "substrate_stats_axis_limits.json"

# Histogram bin counts must match the plotting functions so the scanned peak
# densities equal the plotted ones.
_CV_BINS = 55
_DIAM_BINS = 20


# --------------------------------------------------------------------------- #
# Discovery / parsing helpers
# --------------------------------------------------------------------------- #
def _find_substrate_folders(root: Path):
    """Yield (substrate_folder, pkl_path) for folders with a data/array*.pkl."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Root folder does not exist: {root}")
    for data_dir in sorted(root.rglob("data")):
        if not data_dir.is_dir():
            continue
        pkls = sorted(glob.glob(str(data_dir / "*.pkl")))
        pkls = [p for p in pkls if not os.path.basename(p).startswith("init2d")]
        if not pkls:
            continue
        yield data_dir.parent, pkls[-1]


def _parse_kappa_from_name(name: str):
    m = _K_PATTERN.search(name)
    return float(m.group(1)) if m else None


def _parse_timestamp(pkl_basename: str, fallback: str):
    m = _TS_PATTERN.search(pkl_basename)
    return m.group(1) if m else fallback


def _load_component_tensors(pkl_path: str):
    """Return (box_length, lz, {'outer': tensor, 'inner': tensor|None})."""
    sub = common_util.load_substrate_geometry(pkl_path)
    outer = torch.from_numpy(np.ascontiguousarray(sub.outer_fibers))
    inner = None
    if sub.inner_fibers is not None:
        inner = torch.from_numpy(np.ascontiguousarray(sub.inner_fibers))
    lz = float(sub.lz) if sub.lz is not None else float(sub.box_length)
    return float(sub.box_length), lz, {"outer": outer, "inner": inner}


# --------------------------------------------------------------------------- #
# Phase 1 - data-driven shared axis limits
# --------------------------------------------------------------------------- #
def _round_up(value: float, step: float) -> float:
    if step <= 0 or not np.isfinite(value):
        return float(value)
    return float(math.ceil(value / step) * step)


def _hist_peak_density(values: np.ndarray, bins: int, hi: float) -> float:
    if values.size == 0 or hi <= 0:
        return 0.0
    density, _ = np.histogram(values, bins=bins, range=(0.0, hi), density=True)
    return float(density.max()) if density.size else 0.0


def compute_shared_limits(root: Path, percentile=99.5, x_headroom=1.05, y_headroom=1.10,
                          n_outliers=10):
    """Scan all substrates and return the shared-axis-limits dict.

    The x-ranges are set from a high percentile of the pooled CV / diameter
    values (robust to a few pathological substrates whose raw max would stretch
    every axis). The absolute max and the top substrates by max CV / diameter
    are recorded in the JSON so genuine outliers can be inspected. y-ranges are
    the peak histogram density measured on the final fixed x-ranges.
    """
    cv_arrays, diam_arrays = [], []
    per_sub_cv, per_sub_diam = [], []  # (name, max) for outlier reporting
    max_cv, max_diam = 0.0, 0.0
    n_sub = 0

    for substrate_folder, pkl_path in _find_substrate_folders(root):
        try:
            _, _, comps = _load_component_tensors(pkl_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [scan][skip] {substrate_folder.name}: {exc}")
            continue
        n_sub += 1
        sub_cv_max = sub_diam_max = 0.0
        for fibers in comps.values():
            if fibers is None:
                continue
            cv = along_fiber_plot.compute_cv_values(fibers)
            diam = along_fiber_plot.extract_radius_all(fibers) * 2
            if cv.size:
                cv_arrays.append(cv)
                sub_cv_max = max(sub_cv_max, float(cv.max()))
            if diam.size:
                diam_arrays.append(diam)
                sub_diam_max = max(sub_diam_max, float(diam.max()))
        per_sub_cv.append((substrate_folder.name, round(sub_cv_max, 4)))
        per_sub_diam.append((substrate_folder.name, round(sub_diam_max, 4)))
        max_cv = max(max_cv, sub_cv_max)
        max_diam = max(max_diam, sub_diam_max)

    if n_sub == 0:
        raise RuntimeError(f"No substrates found under {root}")

    cv_all = np.concatenate(cv_arrays) if cv_arrays else np.zeros(1)
    diam_all = np.concatenate(diam_arrays) if diam_arrays else np.zeros(1)
    cv_pct = float(np.percentile(cv_all, percentile))
    diam_pct = float(np.percentile(diam_all, percentile))

    cv_xmax = _round_up(cv_pct * x_headroom, 0.05) or 0.05
    diam_xmax = _round_up(diam_pct * x_headroom, 1.0) or 1.0

    max_cv_density = max((_hist_peak_density(a, _CV_BINS, cv_xmax) for a in cv_arrays), default=0.0)
    max_diam_density = max((_hist_peak_density(a, _DIAM_BINS, diam_xmax) for a in diam_arrays), default=0.0)

    cv_ymax = _round_up(max_cv_density * y_headroom, 1.0) or 1.0
    diam_ymax = _round_up(max_diam_density * y_headroom, 0.1) or 0.1

    top_cv = sorted(per_sub_cv, key=lambda t: t[1], reverse=True)[:n_outliers]
    top_diam = sorted(per_sub_diam, key=lambda t: t[1], reverse=True)[:n_outliers]

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "root": str(root),
        "num_substrates": n_sub,
        "percentile": percentile,
        "x_headroom": x_headroom,
        "y_headroom": y_headroom,
        "cv_percentile_value": round(cv_pct, 4),
        "diameter_percentile_value": round(diam_pct, 4),
        "max_cv_observed": round(max_cv, 4),
        "max_diameter_observed": round(max_diam, 4),
        "max_cv_density_observed": round(max_cv_density, 4),
        "max_diameter_density_observed": round(max_diam_density, 4),
        "top_substrates_by_max_cv": top_cv,
        "top_substrates_by_max_diameter": top_diam,
        "CV_DIAMETER_XLIM": [0.0, cv_xmax],
        "CV_DIAMETER_YLIM": [0.0, cv_ymax],
        "DIAMETER_DIST_XLIM": [0.0, diam_xmax],
        "DIAMETER_DIST_YLIM": [0.0, diam_ymax],
    }


# --------------------------------------------------------------------------- #
# Phase 2 - per-substrate plotting
# --------------------------------------------------------------------------- #
_STALE_TOP_PREDICATES = (
    lambda f: f.startswith("CV_") and f.endswith(".png"),
    lambda f: "Diameter_distribution" in f and f.endswith(".png"),
    lambda f: "Along_axon_radius_variation" in f and f.endswith(".png"),
)
_STALE_ODI_PREDICATES = (
    lambda f: f.startswith("FOD_3D_glyph") and f.endswith(".png"),
    lambda f: f.startswith("OD_histogram") and f.endswith(".png"),
    lambda f: f.startswith("Watson_samples_achieved") and f.endswith(".png"),
)


def _delete_stale_plots(substrate_folder: Path):
    """Remove regenerated plot variants; keep Watson_samples_kappa_*.png."""
    stats_dir = substrate_folder / "figs" / "substrate_stats"
    if stats_dir.is_dir():
        for fn in os.listdir(stats_dir):
            if any(pred(fn) for pred in _STALE_TOP_PREDICATES):
                _safe_remove(stats_dir / fn)
    odi_dir = stats_dir / "ODI"
    if odi_dir.is_dir():
        for fn in os.listdir(odi_dir):
            if any(pred(fn) for pred in _STALE_ODI_PREDICATES):
                _safe_remove(odi_dir / fn)


def _safe_remove(path: Path):
    try:
        os.remove(path)
    except OSError as exc:
        print(f"    [warn] could not delete {path}: {exc}")


_PREFERRED_CHECKS = (
    ("substrate_stats", lambda f: f.startswith("outer_Along_axon_radius_variation")),
    ("substrate_stats", lambda f: f.startswith("CV_outer_diameter")),
    ("substrate_stats", lambda f: f.startswith("outer_Diameter_distribution")),
    ("ODI", lambda f: f.startswith("FOD_3D_glyph_global_watson")),
)


def _has_all_preferred(substrate_folder: Path) -> bool:
    stats_dir = substrate_folder / "figs" / "substrate_stats"
    odi_dir = stats_dir / "ODI"
    listings = {
        "substrate_stats": os.listdir(stats_dir) if stats_dir.is_dir() else [],
        "ODI": os.listdir(odi_dir) if odi_dir.is_dir() else [],
    }
    return all(any(pred(f) for f in listings[where]) for where, pred in _PREFERRED_CHECKS)


def _plot_component(fibers, component_label: str):
    fiber_list = common_util.map_matrix_to_list_numpy(fibers)
    if not fiber_list:
        return
    color = cm.rainbow(np.linspace(0.0, 1.0, len(fiber_list)))
    np.random.shuffle(color)
    along_fiber_plot.plot_along_axon_radius_variation(fibers, colors=color, component_label=component_label)
    along_fiber_plot.plot_diameter_CV_distribution(component_label=component_label)
    along_fiber_plot.plot_diameter_GEV_distribution(fibers, component_label=component_label)


def _configure_for_substrate(substrate_folder: Path, pkl_path: str, box_length: float, lz: float):
    config_params.SUBSTRATE_OUTPUT_FOLDER_PATH = str(substrate_folder)
    config_params.BOX_LENGTH = box_length
    # Thin-z chain termini live at +/- lz/2; split_matrix_to_list needs this.
    config_params.BOX_LENGTH_Z = lz
    kappa = _parse_kappa_from_name(substrate_folder.name) or \
        _parse_kappa_from_name(os.path.basename(pkl_path))
    if kappa is not None:
        config_params.ORIENTATION_SHAPE_PARAM = kappa
    config_params.EXP_DATE_TIME = _parse_timestamp(os.path.basename(pkl_path), substrate_folder.name)


def _plot_one(task):
    """Worker: regenerate all four plot types for a single substrate."""
    substrate_folder, pkl_path, limits, skip_existing = task
    substrate_folder = Path(substrate_folder)
    name = substrate_folder.name
    if skip_existing and _has_all_preferred(substrate_folder):
        return ("skipped", name, None)
    try:
        along_fiber_plot.load_shared_axis_limits(None)  # reset
        _apply_limits(limits)
        box_length, lz, comps = _load_component_tensors(pkl_path)
        _configure_for_substrate(substrate_folder, pkl_path, box_length, lz)

        _delete_stale_plots(substrate_folder)

        _plot_component(comps["outer"], "outer")
        if comps["inner"] is not None:
            _plot_component(comps["inner"], "inner")

        orientation_plot.plot_global_axon_OD(comps["outer"], optimized=True)
        return ("processed", name, None)
    except Exception as exc:  # noqa: BLE001
        return ("failed", name, f"{exc}\n{traceback.format_exc()}")


def _apply_limits(limits: dict):
    for key in ("CV_DIAMETER_XLIM", "CV_DIAMETER_YLIM",
                "DIAMETER_DIST_XLIM", "DIAMETER_DIST_YLIM"):
        val = limits.get(key)
        if val is not None:
            setattr(config_params, key, (float(val[0]), float(val[1])))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _resolve_limits_path(root: Path, limits_json: str | None) -> Path:
    if limits_json:
        return Path(limits_json).resolve()
    return root / _DEFAULT_LIMITS_FILENAME


def process_root(root: Path, limits: dict, skip_existing: bool, jobs: int, shard):
    tasks = [
        (str(folder), pkl, limits, skip_existing)
        for folder, pkl in _find_substrate_folders(root)
    ]
    if shard is not None:
        i, n = shard
        tasks = [t for idx, t in enumerate(tasks) if idx % n == i]

    processed = skipped = failed = 0

    def _tally(result):
        nonlocal processed, skipped, failed
        status, name, info = result
        if status == "skipped":
            print(f"  [skip] {name}")
            skipped += 1
        elif status == "processed":
            print(f"  [done] {name}")
            processed += 1
        else:
            print(f"  [ERROR] {name}: {info}")
            failed += 1

    if jobs and jobs > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=jobs) as pool:
            for result in pool.imap_unordered(_plot_one, tasks):
                _tally(result)
    else:
        for task in tasks:
            _tally(_plot_one(task))
    return processed, skipped, failed


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate substrate_stats plots on data-driven shared axes.")
    parser.add_argument("roots", nargs="+",
                        help="One or more root folders (e.g. experiment/aim2-prep/data_full).")
    parser.add_argument("--limits-json", default=None,
                        help="Path to the shared-axis-limits JSON to read/write "
                             "(default: <root>/substrate_stats_axis_limits.json).")
    parser.add_argument("--use-existing-limits", action="store_true",
                        help="Skip the scan and load an existing limits JSON.")
    parser.add_argument("--only-limits", action="store_true",
                        help="Compute and write the limits JSON, then exit (no plotting).")
    parser.add_argument("--percentile", type=float, default=99.5,
                        help="Percentile of pooled CV / diameter values used for the "
                             "shared x-ranges (robust to outliers; default 99.5).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip substrates that already have all four preferred "
                             "plots. May leave inconsistent axes if limits changed.")
    parser.add_argument("--jobs", "-j", type=int, default=1,
                        help="Parallel worker processes for the plotting pass.")
    parser.add_argument("--shard", default=None,
                        help='Process only a disjoint subset "I/N" (plotting pass only).')
    args = parser.parse_args()

    shard = None
    if args.shard:
        i_str, n_str = args.shard.split("/")
        shard = (int(i_str), int(n_str))
        if not (0 <= shard[0] < shard[1]):
            parser.error(f"--shard I/N requires 0 <= I < N (got {args.shard})")

    total_p = total_s = total_f = 0
    for root_str in args.roots:
        root = Path(root_str).resolve()
        limits_path = _resolve_limits_path(root, args.limits_json)
        print(f"\n{'='*70}\nRoot: {root}\nLimits JSON: {limits_path}\n{'='*70}")

        # ---- Phase 1: shared limits ----
        if args.use_existing_limits:
            limits = along_fiber_plot.load_shared_axis_limits(str(limits_path))
            if limits is None:
                parser.error(f"--use-existing-limits set but no readable JSON at {limits_path}")
            print("Loaded existing shared-axis limits.")
        else:
            print("Scanning dataset for shared axis limits ...")
            limits = compute_shared_limits(root, percentile=args.percentile)
            along_fiber_plot.save_shared_axis_limits(str(limits_path), limits)
            print(f"Wrote shared-axis limits -> {limits_path}")
        for key in ("CV_DIAMETER_XLIM", "CV_DIAMETER_YLIM",
                    "DIAMETER_DIST_XLIM", "DIAMETER_DIST_YLIM"):
            print(f"    {key} = {limits.get(key)}")
        if limits.get("max_cv_observed") is not None:
            print(f"    (percentile={limits.get('percentile')}  "
                  f"CV: p={limits.get('cv_percentile_value')} max={limits.get('max_cv_observed')}  "
                  f"diam: p={limits.get('diameter_percentile_value')} max={limits.get('max_diameter_observed')})")
            top_d = limits.get("top_substrates_by_max_diameter") or []
            if top_d:
                print("    Top substrates by max diameter (µm):")
                for nm, val in top_d[:5]:
                    print(f"        {val:>8}  {nm}")
            top_c = limits.get("top_substrates_by_max_cv") or []
            if top_c:
                print("    Top substrates by max CV:")
                for nm, val in top_c[:5]:
                    print(f"        {val:>8}  {nm}")

        if args.only_limits:
            continue

        # ---- Phase 2: plot ----
        print("\nRegenerating substrate_stats plots ...")
        p, s, f = process_root(root, limits, args.skip_existing, args.jobs, shard)
        total_p += p
        total_s += s
        total_f += f

    if not args.only_limits:
        print(f"\n{'='*70}\nDone.  processed={total_p}  skipped={total_s}  failed={total_f}\n{'='*70}")


if __name__ == "__main__":
    main()
