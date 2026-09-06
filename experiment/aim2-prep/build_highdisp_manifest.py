#!/usr/bin/env python3
"""
build_highdisp_manifest.py
==========================
Build the batch manifest for the *missing* Aim2 high-dispersion substrates.

The existing aim2-prep/data tree covers orientation shape parameter K in
{200, 20, 10}. This builder fills in the missing low-K (high angular
dispersion) values K in {2, 4, 7} across the same diameter x bead grid, at a
per-K 3D volume fraction (K=2 -> 0.3, K=4/7 -> 0.5), WITH myelin (g_ratio 0.7).

Grid (exact (mean, sigma) and (bead_alpha_mean, bead_alpha_stdv) pairs taken
from the existing substrate JSONs so the new set matches the old one):

    diameters : (1.68, 0.45), (2.58, 0.69), (3.5, 0.9375), (4.5, 1.205)
    kappas    : 2, 4, 7
    beads     : (0.3, 0.18), (0.5, 0.15), (0.83, 0.08), (1.24, 0.15)
    VF (3D)   : per-K (K=2 -> 0.3, K=4/7 -> 0.5); 2D init VF auto-estimated
    myelin    : g_ratio 0.7 (inner fibers generated)

Use --kappas to split into separate manifests, e.g. --kappas 4,7 and --kappas 2.
=> 4 x len(kappas) x 4 combos x REPEATS reps.

The manifest is written in the schema consumed by
``simulation_toolkit.cli.batch_substrate``. Generate it, then run with e.g.:

    python3 -m simulation_toolkit.cli.batch_substrate run \\
        --manifest experiment/setup/substrate/batch/<date>_manifest_highdisp.json \\
        --gpus 3,4,5,6,7 \\
        --output-root experiment/aim2-prep/data-high-dispersion

Non-convergent or crashing jobs are flagged status="failed" by the orchestrator
and the remaining jobs keep running.
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Make the NOZOMI package importable when run as a script from the repo root
# or from anywhere (this file lives at <repo>/experiment/aim2-prep/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simulation_toolkit.cli.batch_substrate import (  # noqa: E402
    _atomic_write_json,
    _now_stamp,
    _default_manifest_dir,
)

# ---------------------------------------------------------------------------
# Parameter grid (matches existing aim2-prep data)
# ---------------------------------------------------------------------------

# (mean_diameter, sigma_diameter)
DIAMETERS: List[Tuple[float, float]] = [
    (1.68, 0.45),
    (2.58, 0.69),
    (3.5, 0.9375),
    (4.5, 1.205),
]

# orientation_shape_parameter (Watson kappa) -- the MISSING low-K values.
KAPPAS: List[int] = [2, 4, 7]

# (bead_alpha_mean, bead_alpha_stdv)
BEADS: List[Tuple[float, float]] = [
    (0.3, 0.18),
    (0.5, 0.15),
    (0.83, 0.08),
    (1.24, 0.15),
]

# Lookups so custom-mode --diameters/--beads reuse the canonical sigma pairs.
DIAM_SIGMA: Dict[float, float] = {d: s for d, s in DIAMETERS}
BEAD_STDV: Dict[float, float] = {m: s for m, s in BEADS}

# 3D target volume fraction, per Watson kappa (auto-estimates the 2D init VF).
# High-dispersion (low K) packs are only feasible at lower VF, so K=2 uses 0.3.
KAPPA_VF_3D: Dict[int, float] = {2: 0.3, 4: 0.5, 7: 0.5}

# Default experiment_name per K for regen jobs, so regenerated substrates route
# back into the original experiment trees (K2 -> ...140958; K4/K7 -> ...140957).
DEFAULT_EXP_BY_K: Dict[int, str] = {
    2: "highdisp_20260727_140958",
    4: "highdisp_20260727_140957",
    7: "highdisp_20260727_140957",
}

REPEATS_PER_COMBO = 3

# Fixed substrate params. ``g_ratio`` present => myelin (inner fibers) generated.
FIXED_PARAMS: Dict[str, Any] = {
    "box_length_init": 0,
    "box_length_z_init": 0,  # 0 => auto thin-z (anisotropic Lz, decoupled from Lx=Ly)
    "num_fibers": 500,
    "dist_shape": 0.1,
    "space_buffer_starts_ends": 0.23,
    "spheres_spacing": 0.5,
    "space_buffer_repulse": 0.001,
    "w_overlap": 10,
    "w_curve": 3,
    "w_length": 3,
    "bead_spacing_mean": 5.70,
    "bead_spacing_stdv": 2.88,
    "max_vf_refinements": 10,
    "g_ratio": 0.7,
    "inner_sphere_spacing_ratio": 0.5,
    "repeats": 1,  # orchestrator handles repeats
}

# ---------------------------------------------------------------------------
# Smoke-test config (fast validation, NOT physically diffusion-valid)
# ---------------------------------------------------------------------------
# A tiny, quick end-to-end check: exactly SMOKE_NUM_FIBERS axons packed into a
# FIXED small in-plane box (box_length_init > 0 keeps N fixed -- the auto-N
# diffusion-length floor in substrate_main only triggers when box_length_init==0).
# We DROP final_volume_fraction so the VF-refinement loop does a single pass
# instead of retrying up to max_vf_refinements against the fixed box. We also fix
# a SMALL Lz (box_length_z_init > 0). Lz drives the helix arc length -- and thus
# the number of meshed spheres per fiber -- which dominates runtime at low K. A
# small Lz keeps the run fast while still exercising the anisotropic path
# (Lz != Lx=Ly). No g_ratio => no myelin. NOTE: not physically diffusion-valid;
# this is purely a pipeline smoke, NOT a substrate for simulation.
SMOKE_NUM_FIBERS = 10
SMOKE_BOX_LENGTH = 35   # fixed in-plane box (um); low density => fast
SMOKE_BOX_LENGTH_Z = 12  # fixed thin Lz (um); small => short fibers => fast

# (d_mean, d_sig, kappa, bead_m, bead_s) combos for the smoke -- span K and bead.
SMOKE_COMBOS: List[Tuple[float, float, int, float, float]] = [
    (1.68, 0.45, 2, 0.3, 0.18),
    (1.68, 0.45, 7, 1.24, 0.15),
]


def _build_params(d_mean: float, d_sig: float, kappa: float,
                  bead_m: float, bead_s: float,
                  vf_3d: float | None = None,
                  max_vf_refinements: int | None = None) -> Dict[str, Any]:
    """Merge a single combo's varying values with FIXED_PARAMS."""
    p: Dict[str, Any] = dict(FIXED_PARAMS)
    p["mean_diameter"] = d_mean
    p["sigma_diameter"] = d_sig
    p["orientation_shape_parameter"] = kappa
    p["bead_alpha_mean"] = bead_m
    p["bead_alpha_stdv"] = bead_s
    # 3D target -> substrate_main runs estimate_initial_vf() to derive VF_2D.
    if vf_3d is None:
        vf_3d = KAPPA_VF_3D[kappa]
    p["final_volume_fraction"] = vf_3d
    p["target_volume_fraction"] = vf_3d
    if max_vf_refinements is not None:
        p["max_vf_refinements"] = int(max_vf_refinements)
    return p


def _build_smoke_params(d_mean: float, d_sig: float, kappa: int,
                        bead_m: float, bead_s: float,
                        num_fibers: int = SMOKE_NUM_FIBERS,
                        box_length: float = SMOKE_BOX_LENGTH,
                        box_length_z: float = SMOKE_BOX_LENGTH_Z,
                        g_ratio: float | None = None) -> Dict[str, Any]:
    """Params for a fast, tiny smoke: fixed small box + thin Lz, single VF pass, no myelin."""
    p: Dict[str, Any] = dict(FIXED_PARAMS)
    p["num_fibers"] = num_fibers
    p["box_length_init"] = box_length  # >0 keeps N fixed (no auto-N bump)
    p["box_length_z_init"] = box_length_z  # fixed Lz (anisotropic Lz!=Lx)
    if g_ratio is not None:
        p["g_ratio"] = g_ratio  # enable myelin (inner fibers) for this smoke
    p["mean_diameter"] = d_mean
    p["sigma_diameter"] = d_sig
    p["orientation_shape_parameter"] = kappa
    p["bead_alpha_mean"] = bead_m
    p["bead_alpha_stdv"] = bead_s
    # NO final_volume_fraction -> single-pass (avoids VF-refinement thrash on a
    # fixed box). target_volume_fraction is unused for box sizing when box>0.
    p["target_volume_fraction"] = 0.3
    return p


def build_manifest(repeats: int, kappas: List[int]) -> Dict[str, Any]:
    date_tag = _now_stamp()
    exp_name = f"highdisp_{date_tag}"

    jobs: List[Dict[str, Any]] = []
    job_counter = 0
    n_combos = 0
    for d_mean, d_sig in DIAMETERS:
        for kappa in kappas:
            for bead_m, bead_s in BEADS:
                n_combos += 1
                params = _build_params(d_mean, d_sig, kappa, bead_m, bead_s)
                for r in range(repeats):
                    job_counter += 1
                    jobs.append({
                        "job_id": f"j{job_counter:05d}",
                        "sets": ["highdisp"],
                        "repeat_idx": r,
                        "experiment_name": exp_name,
                        "params": params,
                        "status": "pending",
                        "gpu_id": None,
                        "output_folder": None,
                        "start_time": None,
                        "end_time": None,
                        "duration_sec": None,
                        "error": None,
                    })

    return {
        "created": date_tag,
        "kind": "highdisp",
        "fixed_params": FIXED_PARAMS,
        "kappas": list(kappas),
        "vf_by_kappa": {int(k): KAPPA_VF_3D[k] for k in kappas},
        "n_combos": n_combos,
        "repeats_per_combo": repeats,
        "n_jobs": len(jobs),
        "jobs": jobs,
    }


# ---------------------------------------------------------------------------
# Custom free-grid mode (arbitrary float kappas / single VF / experiment_name)
# ---------------------------------------------------------------------------

def build_custom_manifest(kappas: List[float],
                          diameters: List[Tuple[float, float]],
                          beads: List[Tuple[float, float]],
                          vf: float, repeats: int, exp_name: str,
                          max_vf_refinements: int | None = None) -> Dict[str, Any]:
    """Free grid: arbitrary (possibly <1) kappas x diameters x beads at ONE 3D
    target VF, routed to a custom experiment_name (e.g. ferret-brain)."""
    date_tag = _now_stamp()
    jobs: List[Dict[str, Any]] = []
    job_counter = 0
    n_combos = 0
    for d_mean, d_sig in diameters:
        for kappa in kappas:
            for bead_m, bead_s in beads:
                n_combos += 1
                params = _build_params(d_mean, d_sig, kappa, bead_m, bead_s,
                                       vf_3d=vf,
                                       max_vf_refinements=max_vf_refinements)
                for r in range(repeats):
                    job_counter += 1
                    jobs.append({
                        "job_id": f"j{job_counter:05d}",
                        "sets": [exp_name],
                        "repeat_idx": r,
                        "experiment_name": exp_name,
                        "params": params,
                        "status": "pending",
                        "gpu_id": None,
                        "output_folder": None,
                        "start_time": None,
                        "end_time": None,
                        "duration_sec": None,
                        "error": None,
                    })
    return {
        "created": date_tag,
        "kind": "custom",
        "fixed_params": FIXED_PARAMS,
        "experiment_name": exp_name,
        "kappas": list(kappas),
        "vf": vf,
        "diameters": [d for d, _ in diameters],
        "beads": [b for b, _ in beads],
        "max_vf_refinements": max_vf_refinements,
        "n_combos": n_combos,
        "repeats_per_combo": repeats,
        "n_jobs": len(jobs),
        "jobs": jobs,
    }


# ---------------------------------------------------------------------------
# Retry mode: regenerate failed combos at a lower target VF
# ---------------------------------------------------------------------------

def _collect_failed_combos(manifest_path: str) -> Tuple[Dict[Tuple, Dict[str, Any]], int]:
    """Scan a source manifest and return the combos that have >=1 failed job.

    Returns an ordered dict keyed by the varying combo params
    ``(mean_diameter, sigma_diameter, orientation_shape_parameter,
    bead_alpha_mean, bead_alpha_stdv)`` -> ``{experiment_name, sets}`` (taken from
    the failed job so outputs route back to the source experiment tree), plus the
    source's ``repeats_per_combo``.
    """
    with open(manifest_path) as fh:
        data = json.load(fh)
    repeats = int(data.get("repeats_per_combo") or REPEATS_PER_COMBO)
    combos: Dict[Tuple, Dict[str, Any]] = {}
    for job in data.get("jobs", []):
        if job.get("status") != "failed":
            continue
        p = job["params"]
        key = (
            p["mean_diameter"], p["sigma_diameter"],
            p["orientation_shape_parameter"],
            p["bead_alpha_mean"], p["bead_alpha_stdv"],
        )
        if key not in combos:
            combos[key] = {
                "experiment_name": job["experiment_name"],
                "sets": job.get("sets", ["highdisp"]),
            }
    return combos, repeats


def build_retry_manifest(specs: List[Tuple[str, float]]) -> Dict[str, Any]:
    """Build ONE combined manifest that regenerates ALL reps of every combo with
    >=1 failed job across the given ``(source_manifest, vf)`` specs.

    Each spec's combos are emitted at that spec's target VF, preserving the source
    ``experiment_name`` (so outputs land in ``<exp>/...VF_<vf>/`` alongside the
    existing data). Jobs from all specs are merged with sequential job ids.
    """
    date_tag = _now_stamp()

    jobs: List[Dict[str, Any]] = []
    job_counter = 0
    n_combos = 0
    vf_by_kappa: Dict[int, float] = {}
    sources: List[Dict[str, Any]] = []

    for manifest_path, vf in specs:
        combos, repeats = _collect_failed_combos(manifest_path)
        sources.append({
            "manifest": str(manifest_path),
            "vf": vf,
            "n_combos": len(combos),
            "repeats": repeats,
        })
        for key, meta in combos.items():
            d_mean, d_sig, kappa, bead_m, bead_s = key
            n_combos += 1
            params = _build_params(d_mean, d_sig, int(kappa), bead_m, bead_s)
            # Override the per-K default VF with this spec's retry target.
            params["final_volume_fraction"] = vf
            params["target_volume_fraction"] = vf
            vf_by_kappa[int(kappa)] = vf
            for r in range(repeats):
                job_counter += 1
                jobs.append({
                    "job_id": f"j{job_counter:05d}",
                    "sets": meta["sets"],
                    "repeat_idx": r,
                    "experiment_name": meta["experiment_name"],
                    "params": params,
                    "status": "pending",
                    "gpu_id": None,
                    "output_folder": None,
                    "start_time": None,
                    "end_time": None,
                    "duration_sec": None,
                    "error": None,
                })

    repeats_seen = {s["repeats"] for s in sources}
    repeats_per_combo = repeats_seen.pop() if len(repeats_seen) == 1 else None

    return {
        "created": date_tag,
        "kind": "highdisp_retry",
        "fixed_params": FIXED_PARAMS,
        "kappas": sorted(vf_by_kappa),
        "vf_by_kappa": vf_by_kappa,
        "retry_sources": sources,
        "n_combos": n_combos,
        "repeats_per_combo": repeats_per_combo,
        "n_jobs": len(jobs),
        "jobs": jobs,
    }


# ---------------------------------------------------------------------------
# Regen mode: rebuild converged substrates whose realized VF is out of band
# ---------------------------------------------------------------------------

def build_regen_manifest(csv_path: str, tol: float,
                         init_by_k: Dict[int, float],
                         exp_by_k: Dict[int, str]) -> Dict[str, Any]:
    """Emit one regeneration job per converged substrate whose realized VF (avf)
    is OUTSIDE ``[init-tol, init+tol]`` for its K, at that K's calibrated init VF.

    Reads an avf CSV with columns ``K,d,bead,target_vf,rep,avf`` (as produced by
    the convergence scan). ``sigma_diameter`` / ``bead_alpha_stdv`` are looked up
    from the DIAMETERS / BEADS grid; ``experiment_name`` is routed per K via
    ``exp_by_k`` so outputs land back in the original tree.
    """
    dsig = {round(d, 2): s for d, s in DIAMETERS}
    bstd = {round(b, 2): s for b, s in BEADS}
    date_tag = _now_stamp()

    jobs: List[Dict[str, Any]] = []
    job_counter = 0
    combo_rep: Dict[Tuple[float, int, float], int] = defaultdict(int)
    per_k: Dict[int, Dict[str, int]] = defaultdict(lambda: {"below": 0, "above": 0})

    with open(csv_path, newline="") as fh:
        for row in _csv.DictReader(fh):
            K = int(row["K"]); avf = float(row["avf"])
            d = float(row["d"]); bead = float(row["bead"])
            if K not in init_by_k:
                continue
            init = init_by_k[K]
            lo, hi = init - tol, init + tol
            if lo <= avf <= hi:
                continue
            per_k[K]["below" if avf < lo else "above"] += 1
            sig = dsig.get(round(d, 2)); bs = bstd.get(round(bead, 2))
            if sig is None or bs is None:
                raise SystemExit(f"No sigma/bead_stdv mapping for d={d}, bead={bead}")
            params = _build_params(d, sig, K, bead, bs)
            params["final_volume_fraction"] = init
            params["target_volume_fraction"] = init
            key = (round(d, 2), K, round(bead, 2))
            rep = combo_rep[key]; combo_rep[key] += 1
            job_counter += 1
            jobs.append({
                "job_id": f"j{job_counter:05d}",
                "sets": ["highdisp"],
                "repeat_idx": rep,
                "experiment_name": exp_by_k[K],
                "params": params,
                "status": "pending",
                "gpu_id": None,
                "output_folder": None,
                "start_time": None,
                "end_time": None,
                "duration_sec": None,
                "error": None,
                "regen_of": {
                    "K": K, "d": d, "bead": bead,
                    "orig_target_vf": float(row["target_vf"]),
                    "orig_rep": int(row["rep"]), "orig_avf": avf,
                },
            })

    vf_by_kappa = {K: init_by_k[K] for K in sorted(per_k)}
    n_combos = len({(j["params"]["mean_diameter"],
                     j["params"]["orientation_shape_parameter"],
                     j["params"]["bead_alpha_mean"]) for j in jobs})
    return {
        "created": date_tag,
        "kind": "highdisp_regen",
        "fixed_params": FIXED_PARAMS,
        "kappas": sorted(vf_by_kappa),
        "vf_by_kappa": vf_by_kappa,
        "regen_source_csv": str(csv_path),
        "regen_tol": tol,
        "regen_init_by_k": {str(k): v for k, v in init_by_k.items()},
        "regen_summary": {str(k): dict(per_k[k]) for k in sorted(per_k)},
        "n_combos": n_combos,
        "repeats_per_combo": None,
        "n_jobs": len(jobs),
        "jobs": jobs,
    }


def build_topup_manifest(csv_path: str, tol: float,
                         init_by_k: Dict[int, float],
                         exp_by_k: Dict[int, str],
                         min_inband: int = 3, reps: int = 5,
                         max_vf_refinements: int | None = None) -> Dict[str, Any]:
    """Over-provision reps for combos that don't yet have ``min_inband`` in-band reps.

    Reads an avf CSV (cols ``K,d,bead,target_vf,rep,avf``), counts converged reps
    whose realized avf is within ``[init-tol, init+tol]`` per combo, and for every
    combo below ``min_inband`` emits ``reps`` NEW jobs at that K's calibrated init
    VF (routed via ``exp_by_k``). Optionally overrides ``max_vf_refinements`` so the
    VF-refinement loop gets more attempts to land near target. This deliberately
    generates more than the deficit so stochastic failures still leave >=3 in-band.
    """
    dsig = {round(d, 2): s for d, s in DIAMETERS}
    bstd = {round(b, 2): s for b, s in BEADS}
    date_tag = _now_stamp()

    inband: Dict[Tuple[float, int, float], int] = defaultdict(int)
    seen: set = set()
    with open(csv_path, newline="") as fh:
        for row in _csv.DictReader(fh):
            K = int(row["K"])
            if K not in init_by_k:
                continue
            d = float(row["d"]); bead = float(row["bead"]); avf = float(row["avf"])
            key = (round(d, 2), K, round(bead, 2))
            seen.add(key)
            lo, hi = init_by_k[K] - tol, init_by_k[K] + tol
            if lo <= avf <= hi:
                inband[key] += 1

    short = sorted(k for k in seen if inband[k] < min_inband)
    jobs: List[Dict[str, Any]] = []
    job_counter = 0
    for (d, K, bead) in short:
        sig = dsig.get(round(d, 2)); bs = bstd.get(round(bead, 2))
        if sig is None or bs is None:
            raise SystemExit(f"No sigma/bead_stdv mapping for d={d}, bead={bead}")
        params = _build_params(d, sig, K, bead, bs)
        params["final_volume_fraction"] = init_by_k[K]
        params["target_volume_fraction"] = init_by_k[K]
        if max_vf_refinements is not None:
            params["max_vf_refinements"] = int(max_vf_refinements)
        for r in range(reps):
            job_counter += 1
            jobs.append({
                "job_id": f"j{job_counter:05d}",
                "sets": ["highdisp"],
                "repeat_idx": r,
                "experiment_name": exp_by_k[K],
                "params": params,
                "status": "pending",
                "gpu_id": None,
                "output_folder": None,
                "start_time": None,
                "end_time": None,
                "duration_sec": None,
                "error": None,
                "topup_of": {
                    "K": K, "d": d, "bead": bead,
                    "inband_have": inband[(round(d, 2), K, round(bead, 2))],
                    "min_inband": min_inband,
                },
            })

    vf_by_kappa = {K: init_by_k[K] for K in sorted({k[1] for k in short})}
    return {
        "created": date_tag,
        "kind": "highdisp_topup",
        "fixed_params": FIXED_PARAMS,
        "kappas": sorted(vf_by_kappa),
        "vf_by_kappa": vf_by_kappa,
        "topup_source_csv": str(csv_path),
        "topup_tol": tol,
        "topup_min_inband": min_inband,
        "topup_reps_per_combo": reps,
        "topup_max_vf_refinements": max_vf_refinements,
        "topup_init_by_k": {str(k): v for k, v in init_by_k.items()},
        "topup_short_combos": [
            {"K": k[1], "d": k[0], "bead": k[2], "inband_have": inband[k]}
            for k in short
        ],
        "n_combos": len(short),
        "repeats_per_combo": reps,
        "n_jobs": len(jobs),
        "jobs": jobs,
    }


def build_smoke_manifest(num_fibers: int = SMOKE_NUM_FIBERS,
                         box_length: float = SMOKE_BOX_LENGTH,
                         box_length_z: float = SMOKE_BOX_LENGTH_Z,
                         g_ratio: float | None = None) -> Dict[str, Any]:
    """Build a tiny 2-job smoke manifest (fixed box, single VF pass)."""
    date_tag = _now_stamp()
    exp_name = f"highdisp_smoke_{date_tag}"

    jobs: List[Dict[str, Any]] = []
    for i, (d_mean, d_sig, kappa, bead_m, bead_s) in enumerate(SMOKE_COMBOS, start=1):
        params = _build_smoke_params(d_mean, d_sig, kappa, bead_m, bead_s,
                                     num_fibers=num_fibers,
                                     box_length=box_length,
                                     box_length_z=box_length_z,
                                     g_ratio=g_ratio)
        jobs.append({
            "job_id": f"j{i:05d}",
            "sets": ["highdisp_smoke"],
            "repeat_idx": 0,
            "experiment_name": exp_name,
            "params": params,
            "status": "pending",
            "gpu_id": None,
            "output_folder": None,
            "start_time": None,
            "end_time": None,
            "duration_sec": None,
            "error": None,
        })

    return {
        "created": date_tag,
        "kind": "highdisp_smoke",
        "fixed_params": FIXED_PARAMS,
        "n_combos": len(SMOKE_COMBOS),
        "repeats_per_combo": 1,
        "n_jobs": len(jobs),
        "smoke_settings": {
            "num_fibers": num_fibers,
            "box_length_init": box_length,
            "box_length_z_init": box_length_z,
            "g_ratio": g_ratio,
        },
        "jobs": jobs,
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the high-dispersion (K=2,4,7) substrate batch manifest."
    )
    parser.add_argument("--smoke", action="store_true",
                        help=f"Build a tiny 2-job smoke manifest "
                             f"({SMOKE_NUM_FIBERS} axons, fixed {SMOKE_BOX_LENGTH}um box, "
                             f"single VF pass, no myelin).")
    parser.add_argument("--num-fibers", type=int, default=SMOKE_NUM_FIBERS,
                        help=f"Smoke only: number of fibers (default: {SMOKE_NUM_FIBERS}).")
    parser.add_argument("--box", type=float, default=SMOKE_BOX_LENGTH,
                        help=f"Smoke only: fixed in-plane box Lx=Ly in um "
                             f"(default: {SMOKE_BOX_LENGTH}).")
    parser.add_argument("--box-z", type=float, default=SMOKE_BOX_LENGTH_Z,
                        help=f"Smoke only: fixed Lz in um (default: {SMOKE_BOX_LENGTH_Z}). "
                             f"Use 45 to match the production diffusion-length floor.")
    parser.add_argument("--g-ratio", type=float, default=None,
                        help="Smoke only: enable myelin generation at this g_ratio "
                             "(inner/outer diameter, e.g. 0.7). Default: off (no myelin).")
    parser.add_argument("--repeats", type=int, default=REPEATS_PER_COMBO,
                        help=f"Repeats per combo (default: {REPEATS_PER_COMBO}). "
                             f"Ignored with --smoke.")
    parser.add_argument("--kappas", type=str,
                        default=",".join(str(k) for k in KAPPAS),
                        help="Comma-separated Watson kappa values to include "
                             "(default: 2,4,7). Each K uses its VF from KAPPA_VF_3D "
                             "(K=2 -> 0.3, K=4/7 -> 0.5). Use to split into separate "
                             "manifests, e.g. --kappas 4,7 and --kappas 2. Ignored with --smoke.")
    parser.add_argument("--retry-spec", action="append", default=None,
                        metavar="MANIFEST:VF",
                        help="Retry mode: regenerate ALL reps of every combo that "
                             "has >=1 failed job in MANIFEST, at target VF=VF. "
                             "Repeatable to merge multiple sources into ONE combined "
                             "manifest (each at its own VF), e.g. "
                             "--retry-spec K4-7.json:0.4 --retry-spec K2.json:0.25. "
                             "Ignored with --smoke.")
    parser.add_argument("--regen-out-of-band", type=str, default=None, metavar="CSV",
                        help="Regen mode: read an avf CSV (cols K,d,bead,target_vf,rep,avf) "
                             "and emit one job per substrate whose realized avf is OUTSIDE "
                             "[init-tol, init+tol] for its K, at that K's calibrated init VF. "
                             "Pair with --tol and --init. Ignored with --smoke/--retry-spec.")
    parser.add_argument("--regen-short-combos", type=str, default=None, metavar="CSV",
                        help="Top-up mode: read an avf CSV and, for every combo with "
                             "fewer than --min-inband reps whose avf is within "
                             "[init-tol, init+tol], emit --reps NEW jobs at that K's "
                             "calibrated init VF (over-provision). Pair with --tol/--init; "
                             "use --max-vf-refinements to bump refinement attempts.")
    parser.add_argument("--min-inband", type=int, default=3,
                        help="Top-up: target number of in-band reps per combo (default: 3).")
    parser.add_argument("--reps", type=int, default=5,
                        help="Top-up: NEW jobs to emit per short combo (default: 5).")
    parser.add_argument("--max-vf-refinements", type=int, default=None,
                        help="Top-up/regen: override params.max_vf_refinements (more "
                             "attempts to land near target; default: leave FIXED_PARAMS=10).")
    parser.add_argument("--tol", type=float, default=0.1,
                        help="Regen: acceptance half-width around init VF (default: 0.1).")
    parser.add_argument("--init", type=str, default="2:0.3,4:0.4,7:0.5",
                        help="Regen: per-K calibrated init VF map 'K:VF,...' "
                             "(default: 2:0.3,4:0.4,7:0.5).")
    parser.add_argument("--exp-map", type=str, default=None,
                        help="Regen: per-K experiment_name map 'K:name,...'. Default: "
                             "K2->highdisp_20260727_140958, K4/K7->highdisp_20260727_140957.")
    parser.add_argument("--custom", action="store_true",
                        help="Free-grid mode: --kappas (may be <1) x --diameters x "
                             "--beads at ONE --vf, routed to --exp-name (e.g. "
                             "ferret-brain). Requires --vf and --exp-name.")
    parser.add_argument("--vf", type=float, default=None,
                        help="Custom: single 3D target VF applied to all combos.")
    parser.add_argument("--diameters", type=str, default=None,
                        help=f"Custom: comma diameters from the known grid "
                             f"{[d for d, _ in DIAMETERS]} (default: all).")
    parser.add_argument("--beads", type=str, default=None,
                        help=f"Custom: comma bead_alpha means from the known grid "
                             f"{[b for b, _ in BEADS]} (default: all).")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Custom: experiment_name (output subfolder under "
                             "--output-root), e.g. ferret-brain.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output manifest path (default: "
                             "experiment/setup/substrate/batch/<date>_manifest_highdisp[_smoke].json).")
    args = parser.parse_args(argv)

    def _parse_kmap(s: str, cast):
        out: Dict[int, Any] = {}
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                parser.error(f"Bad K-map entry {part!r}; expected K:VALUE")
            k, v = part.split(":", 1)
            out[int(k)] = cast(v)
        return out

    if args.custom:
        if args.vf is None:
            parser.error("--custom requires --vf")
        if not args.exp_name:
            parser.error("--custom requires --exp-name")
        kappas_f = [float(k) for k in args.kappas.split(",") if k.strip()]
        if not kappas_f:
            parser.error("--kappas must list at least one kappa value.")
        if args.diameters:
            dsel = [float(x) for x in args.diameters.split(",") if x.strip()]
            miss = [d for d in dsel if d not in DIAM_SIGMA]
            if miss:
                parser.error(f"--diameters {miss} not in known grid {sorted(DIAM_SIGMA)}.")
            diameters = [(d, DIAM_SIGMA[d]) for d in dsel]
        else:
            diameters = list(DIAMETERS)
        if args.beads:
            bsel = [float(x) for x in args.beads.split(",") if x.strip()]
            miss = [b for b in bsel if b not in BEAD_STDV]
            if miss:
                parser.error(f"--beads {miss} not in known grid {sorted(BEAD_STDV)}.")
            beads = [(b, BEAD_STDV[b]) for b in bsel]
        else:
            beads = list(BEADS)
        data = build_custom_manifest(kappas_f, diameters, beads, args.vf,
                                     repeats=args.repeats, exp_name=args.exp_name,
                                     max_vf_refinements=args.max_vf_refinements)
        default_name = f"{data['created']}_manifest_{args.exp_name.replace('/', '_')}.json"
    elif args.regen_short_combos:
        csvp = Path(args.regen_short_combos)
        if not csvp.is_file():
            parser.error(f"--regen-short-combos CSV not found: {csvp}")
        init_by_k = _parse_kmap(args.init, float)
        exp_by_k = dict(DEFAULT_EXP_BY_K)
        if args.exp_map:
            exp_by_k.update(_parse_kmap(args.exp_map, str))
        missing = [k for k in init_by_k if k not in exp_by_k]
        if missing:
            parser.error(f"No experiment_name for K={missing}; pass --exp-map.")
        data = build_topup_manifest(str(csvp), args.tol, init_by_k, exp_by_k,
                                    min_inband=args.min_inband, reps=args.reps,
                                    max_vf_refinements=args.max_vf_refinements)
        if data["n_jobs"] == 0:
            parser.error("No short combos found in CSV; nothing to top up.")
        default_name = f"{data['created']}_manifest_highdisp_topup.json"
    elif args.regen_out_of_band:
        csvp = Path(args.regen_out_of_band)
        if not csvp.is_file():
            parser.error(f"--regen-out-of-band CSV not found: {csvp}")
        init_by_k = _parse_kmap(args.init, float)
        exp_by_k = dict(DEFAULT_EXP_BY_K)
        if args.exp_map:
            exp_by_k.update(_parse_kmap(args.exp_map, str))
        missing = [k for k in init_by_k if k not in exp_by_k]
        if missing:
            parser.error(f"No experiment_name for K={missing}; pass --exp-map.")
        data = build_regen_manifest(str(csvp), args.tol, init_by_k, exp_by_k)
        if data["n_jobs"] == 0:
            parser.error("No out-of-band substrates found in CSV; nothing to regenerate.")
        default_name = f"{data['created']}_manifest_highdisp_regen.json"
    elif args.retry_spec:
        specs: List[Tuple[str, float]] = []
        for spec in args.retry_spec:
            if ":" not in spec:
                parser.error(f"--retry-spec must be MANIFEST:VF, got {spec!r}")
            path_str, vf_str = spec.rsplit(":", 1)
            src = Path(path_str)
            if not src.is_file():
                parser.error(f"--retry-spec manifest not found: {src}")
            try:
                vf = float(vf_str)
            except ValueError:
                parser.error(f"--retry-spec VF must be a float, got {vf_str!r}")
            if not 0.0 < vf < 1.0:
                parser.error(f"--retry-spec VF must be in (0, 1), got {vf}")
            specs.append((str(src), vf))
        data = build_retry_manifest(specs)
        if data["n_jobs"] == 0:
            parser.error("No failed combos found in the given --retry-spec "
                         "manifest(s); nothing to build.")
        default_name = f"{data['created']}_manifest_highdisp_retry.json"
    elif args.smoke:
        data = build_smoke_manifest(num_fibers=args.num_fibers,
                                    box_length=args.box,
                                    box_length_z=args.box_z,
                                    g_ratio=args.g_ratio)
        default_name = f"{data['created']}_manifest_highdisp_smoke.json"
    else:
        kappas = [int(k) for k in args.kappas.split(",") if k.strip()]
        if not kappas:
            parser.error("--kappas must list at least one kappa value.")
        unknown = [k for k in kappas if k not in KAPPA_VF_3D]
        if unknown:
            parser.error(f"--kappas contains values with no VF mapping in "
                         f"KAPPA_VF_3D: {unknown}. Known: {sorted(KAPPA_VF_3D)}.")
        data = build_manifest(repeats=args.repeats, kappas=kappas)
        ktag = "-".join(str(k) for k in kappas)
        default_name = f"{data['created']}_manifest_highdisp_K{ktag}.json"

    out = Path(args.out) if args.out else (_default_manifest_dir() / default_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out, data)

    print(f"Wrote {out}")
    print(f"  kind={data['kind']}  combos={data['n_combos']}  "
          f"repeats={data['repeats_per_combo']}  jobs={data['n_jobs']}")
    if args.custom:
        print(f"  exp_name={data['experiment_name']}  vf={data['vf']}  "
              f"max_vf_refinements={data['max_vf_refinements']}")
        print(f"  kappas={data['kappas']}  diameters={data['diameters']}  "
              f"beads={data['beads']}")
        print("\nNext:")
        print("  PYTHON=$PWD/sim_venv/bin/python3 ./run-scripts/run-highdisp-substrate.sh \\")
        print(f"      --manifest {out} \\")
        print("      --gpus 2,4")
    elif args.regen_short_combos:
        print(f"  tol=+/-{data['topup_tol']}  min_inband={data['topup_min_inband']}  "
              f"reps/combo={data['topup_reps_per_combo']}  "
              f"max_vf_refinements={data['topup_max_vf_refinements']}")
        print(f"  init_by_k={data['topup_init_by_k']}  kappas={data['kappas']}")
        print("  short combos (in-band have -> emitting reps):")
        for c in data["topup_short_combos"]:
            print(f"      d{c['d']} K{c['K']} bead{c['bead']}: have {c['inband_have']} in-band "
                  f"-> +{data['topup_reps_per_combo']} reps @ initVF={data['topup_init_by_k'][str(c['K'])]}")
        print("\nNext:")
        print("  PYTHON=$PWD/sim_venv/bin/python3 ./run-scripts/run-highdisp-substrate.sh \\")
        print(f"      --manifest {out} \\")
        print("      --gpus <free-gpus>")
    elif args.regen_out_of_band:
        print(f"  tol=+/-{data['regen_tol']}  init_by_k={data['regen_init_by_k']}")
        print("  out-of-band by K (below/above): "
              + ", ".join(f"K{k}:{v['below']}/{v['above']}"
                          for k, v in data['regen_summary'].items()))
        print(f"  kappas={data['kappas']}  vf_by_kappa={data['vf_by_kappa']}")
        for j in data["jobs"]:
            r = j["regen_of"]; p = j["params"]
            print(f"      regen d={r['d']} K={r['K']} bead={r['bead']} "
                  f"orig(avf={r['orig_avf']},VF={r['orig_target_vf']}) "
                  f"-> initVF={p['target_volume_fraction']}  exp={j['experiment_name']}")
        print("\nNext:")
        print("  PYTHON=$PWD/sim_venv/bin/python3 ./run-scripts/run-highdisp-substrate.sh \\")
        print(f"      --manifest {out} \\")
        print("      --gpus <free-gpus>")
    elif args.retry_spec:
        for s in data["retry_sources"]:
            print(f"    from {s['manifest']}  vf={s['vf']}  "
                  f"combos={s['n_combos']}  reps={s['repeats']}")
        print(f"  kappas={data['kappas']}  vf_by_kappa={data['vf_by_kappa']}")
        seen = set()
        for j in data["jobs"]:
            p = j["params"]
            key = (p["mean_diameter"], p["orientation_shape_parameter"],
                   p["bead_alpha_mean"], p["target_volume_fraction"])
            if key not in seen:
                seen.add(key)
                print(f"      d={p['mean_diameter']} K={p['orientation_shape_parameter']} "
                      f"bead={p['bead_alpha_mean']} VF={p['target_volume_fraction']}")
        print("\nNext:")
        print("  ./run-scripts/run-highdisp-substrate.sh \\")
        print(f"      --manifest {out} \\")
        print("      --gpus 6,7")
    elif args.smoke:
        s = data["smoke_settings"]
        print(f"  smoke: num_fibers={s['num_fibers']}  box_length_init={s['box_length_init']}um  "
              f"Lz={s['box_length_z_init']}um  g_ratio={s['g_ratio']}  "
              f"myelin={s['g_ratio'] is not None}")
        print(f"  combos={[(c[0], c[2], c[3]) for c in SMOKE_COMBOS]} (d, K, bead)")
        print("\nNext:")
        print("  python3 -m simulation_toolkit.cli.batch_substrate run \\")
        print(f"      --manifest {out} \\")
        print("      --gpus 6,7 \\")
        print("      --output-root experiment/aim2-prep/data-high-dispersion")
    else:
        vf_by_k = {k: KAPPA_VF_3D[k] for k in kappas}
        print(f"  diameters={[d[0] for d in DIAMETERS]}  kappas={kappas}  "
              f"beads={[b[0] for b in BEADS]}  VF_3D_by_K={vf_by_k}  "
              f"myelin=True g_ratio={FIXED_PARAMS['g_ratio']}")
        print("\nNext:")
        print("  ./run-scripts/run-highdisp-substrate.sh \\")
        print(f"      --manifest {out} \\")
        print("      --gpus 0,1,2,3,4,5,6")


if __name__ == "__main__":
    main()
