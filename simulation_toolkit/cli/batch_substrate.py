"""Batch substrate generation orchestrator for NOZOMI.

This module is *additive*: it does not modify any existing NOZOMI source. It
imports `substrate_main` and `config_params` and drives them per-job from a
pool of GPU worker processes.

Subcommands:
    build        Build the full 3-set master manifest (672 jobs).
    build-smoke  Build a tiny 4-job smoke-test manifest.
    run          Run pending jobs from a manifest across N GPUs.
    status       Print job-status summary for a manifest.

Manifest schema (JSON):
{
    "created": "<YYYYMMDD_HHMMSS>",
    "kind": "full" | "smoke",
    "fixed_params": { ... },
    "jobs": [
        {
            "job_id": "j00001",
            "sets": ["set1_healthy", "set2_axonloss"],
            "repeat_idx": 0,
            "experiment_name": "set1_healthy_<date>" | ...,
            "params": { full NOZOMI substrate params dict },
            "status": "pending" | "running" | "done" | "failed",
            "gpu_id": null | int,
            "output_folder": null | str,
            "start_time": null | str,
            "end_time": null | str,
            "duration_sec": null | float,
            "error": null | str
        },
        ...
    ]
}
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

# (mean_diameter, sigma_diameter)
DIAMETERS: List[Tuple[float, float]] = [
    (0.5, 0.1339),
    (1.0, 0.2678),
    (1.5, 0.4017),
    (2.5, 0.6695),
    (3.5, 0.9373),
    (4.5, 1.2051),
]

# orientation_shape_parameter (Watson kappa)
KAPPAS: List[int] = [200, 100, 20, 10]

# target volume fraction (3D)
VFS: List[float] = [0.65, 0.40, 0.25]

# (bead_alpha_mean, bead_alpha_stdv)
BEADS: List[Tuple[float, float]] = [
    (0.5, 0.048),
    (0.83, 0.08),
    (1.05, 0.101),
    (1.50, 0.1446),
    (2.50, 0.24),
    (3.6, 0.3469),
]

REPEATS_PER_COMBO = 4

# Fixed substrate params (from
# experiment/setup/substrate/single_substrate/2026-05-05-myelin-bead083/
#   d1-K20-bead0.83-substrate.json)
FIXED_PARAMS: Dict[str, Any] = {
    "box_length_init": 0,
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
    "g_ratio": 0.7,
    "inner_sphere_spacing_ratio": 0.5,
    "repeats": 1,  # orchestrator handles repeats
}

# Set 1: healthy WM   -> fix VF=0.65, bead=(0.5,0.048), vary diameter x kappa
SET1_FIXED_VF = 0.65
SET1_FIXED_BEAD = (0.5, 0.048)

# Set 2: axon loss    -> fix bead=(0.5,0.048), vary diameter x kappa x VF
SET2_FIXED_BEAD = (0.5, 0.048)

# Set 3: beading      -> fix VF=0.65, vary diameter x kappa x bead
SET3_FIXED_VF = 0.65


# ---------------------------------------------------------------------------
# Combo enumeration / dedup
# ---------------------------------------------------------------------------


def _combo_key(d_mean: float, d_sig: float, kappa: int, vf: float,
               bead_m: float, bead_s: float) -> Tuple:
    return (round(d_mean, 6), round(d_sig, 6), int(kappa),
            round(vf, 6), round(bead_m, 6), round(bead_s, 6))


def enumerate_full_combos() -> List[Dict[str, Any]]:
    """Enumerate the union of Set 1/2/3 combos, deduped, with set tags."""
    combos: Dict[Tuple, Dict[str, Any]] = {}

    def add(d, k, vf, b, set_tag):
        d_mean, d_sig = d
        bead_m, bead_s = b
        key = _combo_key(d_mean, d_sig, k, vf, bead_m, bead_s)
        if key not in combos:
            combos[key] = {
                "mean_diameter": d_mean,
                "sigma_diameter": d_sig,
                "orientation_shape_parameter": k,
                "target_volume_fraction": vf,
                "bead_alpha_mean": bead_m,
                "bead_alpha_stdv": bead_s,
                "sets": [],
            }
        if set_tag not in combos[key]["sets"]:
            combos[key]["sets"].append(set_tag)

    # Set 1: 6 x 4 = 24
    for d in DIAMETERS:
        for k in KAPPAS:
            add(d, k, SET1_FIXED_VF, SET1_FIXED_BEAD, "set1_healthy")

    # Set 2: 6 x 4 x 3 = 72
    for d in DIAMETERS:
        for k in KAPPAS:
            for vf in VFS:
                add(d, k, vf, SET2_FIXED_BEAD, "set2_axonloss")

    # Set 3: 6 x 4 x 5 = 120
    for d in DIAMETERS:
        for k in KAPPAS:
            for b in BEADS:
                add(d, k, SET3_FIXED_VF, b, "set3_beading")

    return list(combos.values())


def enumerate_smoke_combos() -> List[Dict[str, Any]]:
    """Hardcoded smoke-test combos: d=0.5/sig=0.1339, K=10, VF=0.65, 2 beads."""
    smoke_beads = [(2.50, 0.24), (3.6, 0.3469)]
    out = []
    for b in smoke_beads:
        out.append({
            "mean_diameter": 0.5,
            "sigma_diameter": 0.1339,
            "orientation_shape_parameter": 10,
            "target_volume_fraction": 0.65,
            "bead_alpha_mean": b[0],
            "bead_alpha_stdv": b[1],
            "sets": ["smoke"],
        })
    return out


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_params(combo: Dict[str, Any]) -> Dict[str, Any]:
    """Merge varying combo params with FIXED_PARAMS into a NOZOMI param dict.

    Uses ``final_volume_fraction`` so NOZOMI's
    ``estimate_initial_vf`` 2D->3D pre-calc + iterative refinement runs.
    """
    p: Dict[str, Any] = dict(FIXED_PARAMS)
    p["mean_diameter"] = combo["mean_diameter"]
    p["sigma_diameter"] = combo["sigma_diameter"]
    p["orientation_shape_parameter"] = combo["orientation_shape_parameter"]
    p["bead_alpha_mean"] = combo["bead_alpha_mean"]
    p["bead_alpha_stdv"] = combo["bead_alpha_stdv"]
    # NOTE: use final_volume_fraction (3D target) so the 2D pre-calc kicks in.
    p["final_volume_fraction"] = combo["target_volume_fraction"]
    # Also keep target_volume_fraction so any older code path still has it.
    p["target_volume_fraction"] = combo["target_volume_fraction"]
    return p


def _experiment_name_for_combo(combo: Dict[str, Any], date_tag: str) -> str:
    """Pick a per-set experiment folder name. If combo belongs to multiple
    sets, use the *first* tag (priority: set1 < set2 < set3) so output is in
    one place; the manifest's ``sets`` list still records membership for all
    sets and the per-set MC analysis can filter on it.
    """
    priority = ["set1_healthy", "set2_axonloss", "set3_beading", "smoke"]
    for tag in priority:
        if tag in combo["sets"]:
            return f"{tag}_{date_tag}"
    return f"misc_{date_tag}"


def build_manifest(kind: str, repeats: int) -> Dict[str, Any]:
    """Build a manifest dict for ``kind`` in {"full", "smoke"}."""
    if kind == "full":
        combos = enumerate_full_combos()
    elif kind == "smoke":
        combos = enumerate_smoke_combos()
    else:
        raise ValueError(f"Unknown kind: {kind}")

    date_tag = _now_stamp()
    jobs: List[Dict[str, Any]] = []
    job_counter = 0
    for combo in combos:
        params = _build_params(combo)
        exp_name = _experiment_name_for_combo(combo, date_tag)
        for r in range(repeats):
            job_counter += 1
            jobs.append({
                "job_id": f"j{job_counter:05d}",
                "sets": list(combo["sets"]),
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
        "kind": kind,
        "fixed_params": FIXED_PARAMS,
        "n_combos": len(combos),
        "repeats_per_combo": repeats,
        "n_jobs": len(jobs),
        "jobs": jobs,
    }


# ---------------------------------------------------------------------------
# Atomic manifest I/O
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _read_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def update_job(manifest_path: Path, lock, job_id: str, **fields) -> None:
    """Atomically update a single job's fields in the manifest on disk."""
    with lock:
        data = _read_manifest(manifest_path)
        for j in data["jobs"]:
            if j["job_id"] == job_id:
                j.update(fields)
                break
        _atomic_write_json(manifest_path, data)


# ---------------------------------------------------------------------------
# GPU worker
# ---------------------------------------------------------------------------


def _gpu_worker(gpu_id: int, queue: "mp.Queue", manifest_path: str, lock,
                log_dir: str, output_root: str) -> None:
    """Worker process: pin to one GPU and process jobs until queue is empty."""
    # Must set CUDA_VISIBLE_DEVICES *before* importing torch / NOZOMI modules.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = Path(log_dir) / f"gpu{gpu_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", buffering=1)

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}][gpu{gpu_id}] {msg}"
        log_f.write(line + "\n")
        print(line, flush=True)

    log(f"worker started, CUDA_VISIBLE_DEVICES={gpu_id}")

    # Defer NOZOMI imports until after env var is set.
    try:
        from simulation_toolkit.cli.substrate_main import substrate_main
        import simulation_toolkit.toolkit_params as config_params
    except Exception:
        log("FATAL: failed to import NOZOMI modules:\n" + traceback.format_exc())
        log_f.close()
        return

    config_params.OUTPUT_FOLDER_PATH = output_root

    manifest_path_p = Path(manifest_path)
    durations: List[float] = []

    while True:
        try:
            job = queue.get(timeout=2)
        except Exception:
            log("queue empty, exiting")
            break
        if job is None:
            break

        job_id = job["job_id"]
        params = job["params"]
        exp_name = job["experiment_name"]
        start = time.time()
        start_iso = datetime.now().isoformat(timespec="seconds")

        update_job(manifest_path_p, lock, job_id,
                   status="running", gpu_id=gpu_id, start_time=start_iso)

        log(f"START {job_id} sets={job['sets']} rep={job['repeat_idx']} "
            f"d={params['mean_diameter']} K={params['orientation_shape_parameter']} "
            f"VF={params.get('final_volume_fraction')} "
            f"bead=({params['bead_alpha_mean']},{params['bead_alpha_stdv']}) "
            f"-> {exp_name}")

        try:
            substrate_main(params, exp_name, folder_suffix=f"_rep{job['repeat_idx']}")
            output_folder = getattr(config_params, "SUBSTRATE_OUTPUT_FOLDER_PATH", None)
            duration = time.time() - start
            durations.append(duration)
            eta_per_job = sum(durations) / len(durations)
            update_job(manifest_path_p, lock, job_id,
                       status="done",
                       end_time=datetime.now().isoformat(timespec="seconds"),
                       duration_sec=round(duration, 2),
                       output_folder=output_folder,
                       error=None)
            log(f"DONE  {job_id} in {duration:.1f}s (rolling avg "
                f"{eta_per_job:.1f}s) -> {output_folder}")
        except Exception as e:
            duration = time.time() - start
            tb = traceback.format_exc()
            update_job(manifest_path_p, lock, job_id,
                       status="failed",
                       end_time=datetime.now().isoformat(timespec="seconds"),
                       duration_sec=round(duration, 2),
                       error=f"{type(e).__name__}: {e}\n{tb}")
            log(f"FAIL  {job_id} in {duration:.1f}s: {type(e).__name__}: {e}")

    log_f.close()


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------


def reset_manifest(manifest_path: Path, target_status: str = "running") -> int:
    """Reset jobs stuck at ``target_status`` back to 'pending'.

    Returns the number of jobs reset.
    """
    data = _read_manifest(manifest_path)
    count = 0
    for j in data["jobs"]:
        if j["status"] == target_status:
            j["status"] = "pending"
            j["gpu_id"] = None
            j["start_time"] = None
            j["end_time"] = None
            j["duration_sec"] = None
            j["error"] = None
            count += 1
    if count:
        _atomic_write_json(manifest_path, data)
    return count


def run_manifest(manifest_path: Path, gpus: List[int], limit: Optional[int],
                 retry_failed: bool, recover: bool, dry_run: bool,
                 output_root: str, log_dir: str) -> None:
    data = _read_manifest(manifest_path)
    jobs = data["jobs"]

    statuses = {"pending"}
    if retry_failed:
        statuses.add("failed")
    if recover:
        statuses.add("running")

    pending = [j for j in jobs if j["status"] in statuses]
    if limit is not None:
        pending = pending[:limit]

    print(f"Manifest: {manifest_path}")
    print(f"Total jobs in manifest: {len(jobs)}")
    print(f"Jobs to run (status in {statuses}, limit={limit}): {len(pending)}")
    print(f"GPUs: {gpus}")
    print(f"Output root: {output_root}")

    if dry_run:
        print("\n--dry-run set; not launching workers. Sample of first 5 jobs:")
        for j in pending[:5]:
            p = j["params"]
            print(f"  {j['job_id']} sets={j['sets']} rep={j['repeat_idx']} "
                  f"d={p['mean_diameter']} K={p['orientation_shape_parameter']} "
                  f"VF={p.get('final_volume_fraction')} "
                  f"bead=({p['bead_alpha_mean']},{p['bead_alpha_stdv']}) "
                  f"-> {j['experiment_name']}")
        return

    if not pending:
        print("Nothing to run.")
        return

    Path(output_root).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    queue = manager.Queue()
    lock = manager.Lock()

    # Reset transient fields on jobs we are about to (re)run, so prior
    # failures don't keep their old gpu_id/error visible while running.
    pending_ids = {j["job_id"] for j in pending}
    for j in jobs:
        if j["job_id"] in pending_ids:
            j["status"] = "pending"
            j["gpu_id"] = None
            j["start_time"] = None
            j["end_time"] = None
            j["duration_sec"] = None
            j["error"] = None
    _atomic_write_json(manifest_path, data)

    for j in pending:
        # Hand the worker a shallow copy of just what it needs.
        queue.put({
            "job_id": j["job_id"],
            "sets": j["sets"],
            "repeat_idx": j["repeat_idx"],
            "experiment_name": j["experiment_name"],
            "params": j["params"],
        })

    # Set CUDA_VISIBLE_DEVICES in the PARENT before each p.start() so the
    # spawned subprocess inherits it at birth — before pycuda.autoinit (which
    # is triggered by simulation_toolkit/__init__.py) runs in the child.
    # Setting it only inside _gpu_worker() is too late; the package import
    # during spawn bootstrap already initialises CUDA on GPU 0 by then.
    _parent_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    workers = []
    for gpu_id in gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        p = ctx.Process(
            target=_gpu_worker,
            args=(gpu_id, queue, str(manifest_path), lock, log_dir, output_root),
            daemon=False,
        )
        p.start()
        workers.append(p)
    # Restore the parent's env so it isn't polluted after spawning.
    if _parent_cuda is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _parent_cuda

    try:
        for p in workers:
            p.join()
    except KeyboardInterrupt:
        print("Interrupted; terminating workers...")
        for p in workers:
            p.terminate()
        for p in workers:
            p.join()

    # Final summary
    data = _read_manifest(manifest_path)
    counts: Dict[str, int] = {}
    for j in data["jobs"]:
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    print(f"\nFinal status counts: {counts}")


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------


def status_manifest(manifest_path: Path) -> None:
    data = _read_manifest(manifest_path)
    jobs = data["jobs"]
    by_status: Dict[str, int] = {}
    by_set_status: Dict[Tuple[str, str], int] = {}
    failed: List[Dict[str, Any]] = []
    for j in jobs:
        s = j["status"]
        by_status[s] = by_status.get(s, 0) + 1
        for tag in j["sets"]:
            by_set_status[(tag, s)] = by_set_status.get((tag, s), 0) + 1
        if s == "failed":
            failed.append(j)

    print(f"Manifest: {manifest_path}")
    print(f"Created: {data.get('created')}  Kind: {data.get('kind')}  "
          f"Combos: {data.get('n_combos')}  Repeats: {data.get('repeats_per_combo')}  "
          f"Jobs: {len(jobs)}")
    print("\nBy status:")
    for s, n in sorted(by_status.items()):
        print(f"  {s:8s}  {n}")

    print("\nBy set x status:")
    sets = sorted({tag for (tag, _) in by_set_status})
    statuses = ["pending", "running", "done", "failed"]
    header = "  " + "set".ljust(20) + "".join(s.rjust(10) for s in statuses)
    print(header)
    for tag in sets:
        row = "  " + tag.ljust(20)
        for s in statuses:
            row += str(by_set_status.get((tag, s), 0)).rjust(10)
        print(row)

    stale_running = [j for j in jobs if j["status"] == "running"]
    if stale_running:
        print(f"\nWARNING: {len(stale_running)} job(s) are stuck at 'running' with no "
              f"active process (orphaned from a previous killed run).")
        print("  Run:  batch_substrate reset --manifest <path>  to reset them to 'pending'.")
        for j in stale_running[:10]:
            print(f"  {j['job_id']} gpu={j.get('gpu_id')} started={j.get('start_time')}")
        if len(stale_running) > 10:
            print(f"  ... and {len(stale_running)-10} more")

    if failed:
        print(f"\nFailed jobs ({len(failed)}):")
        for j in failed[:20]:
            err = (j.get("error") or "").splitlines()[0] if j.get("error") else ""
            print(f"  {j['job_id']} gpu={j.get('gpu_id')} -> {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed)-20} more")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_manifest_dir() -> Path:
    return Path("experiment/setup/substrate/batch")


def _default_output_root() -> Path:
    return Path("experiment/result")


def _default_log_dir() -> Path:
    return Path("experiment/setup/substrate/batch/logs")


def _parse_gpu_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _cmd_build(args: argparse.Namespace) -> None:
    data = build_manifest(kind="full", repeats=args.repeats)
    out = Path(args.out) if args.out else (
        _default_manifest_dir() / f"{data['created']}_manifest_full.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out, data)
    print(f"Wrote {out}")
    print(f"  combos={data['n_combos']}  repeats={data['repeats_per_combo']}  "
          f"jobs={data['n_jobs']}")
    # Quick sanity prints
    set_counts: Dict[str, int] = {}
    for j in data["jobs"]:
        for tag in j["sets"]:
            set_counts[tag] = set_counts.get(tag, 0) + 1
    print(f"  jobs tagged per set: {set_counts}")


def _cmd_build_smoke(args: argparse.Namespace) -> None:
    data = build_manifest(kind="smoke", repeats=args.repeats)
    out = Path(args.out) if args.out else (
        _default_manifest_dir() / f"{data['created']}_manifest_smoke.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out, data)
    print(f"Wrote {out}")
    print(f"  combos={data['n_combos']}  repeats={data['repeats_per_combo']}  "
          f"jobs={data['n_jobs']}")
    for j in data["jobs"]:
        p = j["params"]
        print(f"  {j['job_id']} rep={j['repeat_idx']} "
              f"d={p['mean_diameter']} K={p['orientation_shape_parameter']} "
              f"VF={p.get('final_volume_fraction')} "
              f"bead=({p['bead_alpha_mean']},{p['bead_alpha_stdv']})")


def _cmd_run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    gpus = _parse_gpu_list(args.gpus)
    output_root = args.output_root or str(_default_output_root())
    log_dir = args.log_dir or str(_default_log_dir())
    run_manifest(
        manifest_path=manifest_path,
        gpus=gpus,
        limit=args.limit,
        retry_failed=args.retry_failed,
        recover=args.recover,
        dry_run=args.dry_run,
        output_root=output_root,
        log_dir=log_dir,
    )


def _cmd_reset(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    n = reset_manifest(manifest_path, target_status="running")
    if args.also_failed:
        n += reset_manifest(manifest_path, target_status="failed")
    print(f"Reset {n} job(s) to 'pending' in {manifest_path}")


def _cmd_status(args: argparse.Namespace) -> None:
    status_manifest(Path(args.manifest))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Batch substrate generation across multiple GPUs (NOZOMI)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build the full 3-set manifest.")
    p_build.add_argument("--out", type=str, default=None,
                         help="Output manifest path "
                              "(default: experiment/setup/substrate/batch/<date>_manifest_full.json)")
    p_build.add_argument("--repeats", type=int, default=REPEATS_PER_COMBO,
                         help=f"Repeats per unique combo (default: {REPEATS_PER_COMBO}).")
    p_build.set_defaults(func=_cmd_build)

    p_smoke = sub.add_parser("build-smoke", help="Build the smoke-test manifest (4 jobs).")
    p_smoke.add_argument("--out", type=str, default=None,
                         help="Output manifest path "
                              "(default: experiment/setup/substrate/batch/<date>_manifest_smoke.json)")
    p_smoke.add_argument("--repeats", type=int, default=2,
                         help="Repeats per smoke combo (default: 2).")
    p_smoke.set_defaults(func=_cmd_build_smoke)

    p_run = sub.add_parser("run", help="Run pending jobs from a manifest.")
    p_run.add_argument("--manifest", type=str, required=True,
                       help="Path to manifest JSON.")
    p_run.add_argument("--gpus", type=str, default="0,1,2,3,4,5",
                       help="Comma-separated GPU IDs (default: 0,1,2,3,4,5).")
    p_run.add_argument("--limit", type=int, default=None,
                       help="Run only the first N pending jobs.")
    p_run.add_argument("--retry-failed", action="store_true",
                       help="Also re-run jobs marked status=failed.")
    p_run.add_argument("--recover", action="store_true",
                       help="Also pick up stale 'running' jobs left by a "
                            "previously killed run (implies they are orphaned).")
    p_run.add_argument("--dry-run", action="store_true",
                       help="Print what would run, but do not launch workers.")
    p_run.add_argument("--output-root", type=str, default=None,
                       help="Override output root (default: experiment/result).")
    p_run.add_argument("--log-dir", type=str, default=None,
                       help="Override log dir "
                            "(default: experiment/setup/substrate/batch/logs).")
    p_run.set_defaults(func=_cmd_run)

    p_status = sub.add_parser("status", help="Print status summary for a manifest.")
    p_status.add_argument("--manifest", type=str, required=True)
    p_status.set_defaults(func=_cmd_status)

    p_reset = sub.add_parser(
        "reset",
        help="Reset stale 'running' jobs back to 'pending' (use after a crashed run)."
    )
    p_reset.add_argument("--manifest", type=str, required=True)
    p_reset.add_argument("--also-failed", action="store_true",
                         help="Also reset 'failed' jobs back to 'pending'.")
    p_reset.set_defaults(func=_cmd_reset)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    # Make the package importable when running as a script from repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
