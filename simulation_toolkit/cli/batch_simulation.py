"""Batch Monte Carlo simulation orchestrator for NOZOMI.

Mirrors :mod:`simulation_toolkit.cli.batch_substrate`. Reads a substrate
manifest (produced by ``batch_substrate``), enumerates exactly two sim jobs
per completed substrate pickle (one ``compartment="intra"``, one
``compartment="extra"``), and runs them across N GPUs (one job per GPU at
a time). No additional sim-level repetition: variance comes from the
substrate-level ``repeat_idx`` already encoded in the substrate manifest.

Subcommands:
    build        Build a full sim manifest from a substrate manifest.
    build-smoke  Same, but only includes ``status=="done"`` substrates from a
                 ``kind=="smoke"`` substrate manifest (or any manifest if used
                 with ``--limit-substrates``).
    run          Run pending jobs from a sim manifest across N GPUs.
    reset        Reset stuck 'running' jobs back to 'pending'.
    status       Print job-status summary.

Manifest schema (JSON):
{
    "created":   "<YYYYMMDD_HHMMSS>",
    "kind":      "full" | "smoke",
    "source_substrate_manifest": "<path>",
    "sim_params": { time_step, num_spins, D0_intra, D0_extra, nseg, sim_time },
    "n_jobs":    <int>,
    "jobs": [
        {
            "job_id":            "j00001",
            "substrate_job_id":  "j00001",
            "sets":              ["set1_healthy", ...],
            "repeat_idx":        0,
            "compartment":       "intra" | "extra",
            "substrate_pkl":     "<absolute path to substrate .pkl>",
            "params":            { ... full sim params dict ... },
            "status":            "pending" | "running" | "done" | "failed",
            "gpu_id":            null | int,
            "output_pkl":        null | str,
            "start_time":        null | str,
            "end_time":          null | str,
            "duration_sec":      null | float,
            "error":             null | str
        },
        ...
    ]
}
"""

from __future__ import annotations

import argparse
import glob
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
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SIM_CONFIG = "experiment/setup/simulation/default-sim.json"

# Compartments enumerated per substrate. Order matters only for deterministic
# job_id assignment.
COMPARTMENTS: Tuple[str, ...] = ("intra", "extra")


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Sim-param assembly
# ---------------------------------------------------------------------------


def _load_sim_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = json.load(f)
    # Drop pulse-specific keys that belong to gradient_simulation only.
    cfg.pop("pulse", None)
    return cfg


def _build_sim_params(base: Dict[str, Any],
                      sim_time: float,
                      num_spins: Optional[int],
                      time_step: Optional[float],
                      nseg: Optional[int],
                      d0_intra: Optional[float],
                      d0_extra: Optional[float]) -> Dict[str, Any]:
    """Assemble a sim params dict from a base config + CLI overrides."""
    p: Dict[str, Any] = dict(base)
    p["sim_time"] = float(sim_time)
    if num_spins is not None:
        p["num_spins"] = int(num_spins)
    if time_step is not None:
        p["time_step"] = float(time_step)
    if nseg is not None:
        p["nseg"] = int(nseg)
    if d0_intra is not None:
        p["D0_intra"] = float(d0_intra)
    if d0_extra is not None:
        p["D0_extra"] = float(d0_extra)
    # Required keys for simulation_main.simulation_main()
    required = {"time_step", "num_spins", "D0_intra", "D0_extra", "nseg", "sim_time"}
    missing = required - set(p.keys())
    if missing:
        raise ValueError(f"Sim params missing required keys: {sorted(missing)}")
    return p


# ---------------------------------------------------------------------------
# Substrate-pkl discovery
# ---------------------------------------------------------------------------


def _find_substrate_pkl(output_folder: str) -> Optional[str]:
    """Return absolute path of the substrate pickle inside ``output_folder/data``.

    Returns None if the folder does not exist or contains no .pkl files.
    If multiple .pkl files are present, the lexicographically first is chosen.
    """
    if not output_folder:
        return None
    data_dir = Path(output_folder) / "data"
    if not data_dir.is_dir():
        return None
    pkls = sorted(data_dir.glob("*.pkl"))
    if not pkls:
        return None
    return str(pkls[0].resolve())


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def build_sim_manifest(substrate_manifest_path: Path,
                       sim_params_template: Dict[str, Any],
                       kind: str,
                       limit_substrates: Optional[int]) -> Dict[str, Any]:
    """Build a sim manifest by enumerating intra+extra jobs over completed
    substrates from ``substrate_manifest_path``."""
    with open(substrate_manifest_path, "r") as f:
        sub_data = json.load(f)

    sub_jobs = sub_data.get("jobs", [])
    done_subs = [j for j in sub_jobs if j.get("status") == "done"]
    if limit_substrates is not None:
        done_subs = done_subs[:limit_substrates]

    jobs: List[Dict[str, Any]] = []
    skipped: List[str] = []
    job_counter = 0
    for sub_job in done_subs:
        pkl = _find_substrate_pkl(sub_job.get("output_folder") or "")
        if pkl is None:
            skipped.append(sub_job["job_id"])
            continue
        for compartment in COMPARTMENTS:
            job_counter += 1
            params = dict(sim_params_template)
            params["compartment"] = compartment
            jobs.append({
                "job_id": f"j{job_counter:05d}",
                "substrate_job_id": sub_job["job_id"],
                "sets": list(sub_job.get("sets", [])),
                "repeat_idx": sub_job.get("repeat_idx", 0),
                "compartment": compartment,
                "substrate_pkl": pkl,
                "params": params,
                "status": "pending",
                "gpu_id": None,
                "output_pkl": None,
                "start_time": None,
                "end_time": None,
                "duration_sec": None,
                "error": None,
            })

    return {
        "created": _now_stamp(),
        "kind": kind,
        "source_substrate_manifest": str(substrate_manifest_path.resolve()),
        "sim_params": {k: v for k, v in sim_params_template.items()
                       if k != "compartment"},
        "n_substrates_done": len(done_subs),
        "n_substrates_skipped": len(skipped),
        "skipped_substrate_job_ids": skipped,
        "n_jobs": len(jobs),
        "jobs": jobs,
    }


# ---------------------------------------------------------------------------
# Atomic JSON I/O
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


def _newest_output_pkl(substrate_pkl: str, compartment: str,
                       since_ts: float) -> Optional[str]:
    """Return path of the newest ``diffcoeff_<compartment>_*.pkl`` produced
    inside ``<substrate_folder>/sim/ADCdata`` after ``since_ts``."""
    substrate_folder = Path(substrate_pkl).parent.parent  # .../<sub>/data/<x.pkl>
    adc_dir = substrate_folder / "sim" / "ADCdata"
    if not adc_dir.is_dir():
        return None
    candidates = []
    for p in adc_dir.glob(f"diffcoeff_{compartment}_*.pkl"):
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if mtime >= since_ts - 1.0:
            candidates.append((mtime, p))
    if not candidates:
        return None
    candidates.sort()
    return str(candidates[-1][1].resolve())


def _gpu_worker(gpu_id: int, queue: "mp.Queue", manifest_path: str, lock,
                log_dir: str) -> None:
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

    try:
        from simulation_toolkit.cli.simulation_main import simulation_main
    except Exception:
        log("FATAL: failed to import simulation_main:\n" + traceback.format_exc())
        log_f.close()
        return

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
        substrate_pkl = job["substrate_pkl"]
        compartment = job["compartment"]
        start = time.time()
        start_iso = datetime.now().isoformat(timespec="seconds")

        update_job(manifest_path_p, lock, job_id,
                   status="running", gpu_id=gpu_id, start_time=start_iso)

        log(f"START {job_id} sub={job['substrate_job_id']} "
            f"rep={job['repeat_idx']} sets={job['sets']} "
            f"compartment={compartment} -> {substrate_pkl}")

        try:
            simulation_main(params, substrate_pkl)
            output_pkl = _newest_output_pkl(substrate_pkl, compartment, start)
            duration = time.time() - start
            durations.append(duration)
            avg = sum(durations) / len(durations)
            update_job(manifest_path_p, lock, job_id,
                       status="done",
                       end_time=datetime.now().isoformat(timespec="seconds"),
                       duration_sec=round(duration, 2),
                       output_pkl=output_pkl,
                       error=None)
            log(f"DONE  {job_id} in {duration:.1f}s (avg {avg:.1f}s) -> {output_pkl}")
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
# Run / reset / status
# ---------------------------------------------------------------------------


def reset_manifest(manifest_path: Path, target_status: str = "running") -> int:
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
                 log_dir: str) -> None:
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

    if dry_run:
        print("\n--dry-run set; not launching workers. Sample of first 5 jobs:")
        for j in pending[:5]:
            print(f"  {j['job_id']} sub={j['substrate_job_id']} "
                  f"rep={j['repeat_idx']} compartment={j['compartment']} "
                  f"-> {j['substrate_pkl']}")
        return

    if not pending:
        print("Nothing to run.")
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    queue = manager.Queue()
    lock = manager.Lock()

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
        queue.put({
            "job_id": j["job_id"],
            "substrate_job_id": j["substrate_job_id"],
            "sets": j["sets"],
            "repeat_idx": j["repeat_idx"],
            "compartment": j["compartment"],
            "substrate_pkl": j["substrate_pkl"],
            "params": j["params"],
        })

    # Set CUDA_VISIBLE_DEVICES in the PARENT before each p.start() so the
    # spawned child inherits it before pycuda.autoinit fires at import time.
    _parent_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    workers = []
    for gpu_id in gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        p = ctx.Process(
            target=_gpu_worker,
            args=(gpu_id, queue, str(manifest_path), lock, log_dir),
            daemon=False,
        )
        p.start()
        workers.append(p)
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

    data = _read_manifest(manifest_path)
    counts: Dict[str, int] = {}
    for j in data["jobs"]:
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    print(f"\nFinal status counts: {counts}")


def status_manifest(manifest_path: Path) -> None:
    data = _read_manifest(manifest_path)
    jobs = data["jobs"]
    by_status: Dict[str, int] = {}
    by_compartment_status: Dict[Tuple[str, str], int] = {}
    by_set_status: Dict[Tuple[str, str], int] = {}
    failed: List[Dict[str, Any]] = []
    for j in jobs:
        s = j["status"]
        c = j["compartment"]
        by_status[s] = by_status.get(s, 0) + 1
        by_compartment_status[(c, s)] = by_compartment_status.get((c, s), 0) + 1
        for tag in j["sets"]:
            by_set_status[(tag, s)] = by_set_status.get((tag, s), 0) + 1
        if s == "failed":
            failed.append(j)

    print(f"Manifest: {manifest_path}")
    print(f"Created: {data.get('created')}  Kind: {data.get('kind')}  "
          f"Jobs: {len(jobs)}  Source: {data.get('source_substrate_manifest')}")

    print("\nBy status:")
    for s, n in sorted(by_status.items()):
        print(f"  {s:8s}  {n}")

    statuses = ["pending", "running", "done", "failed"]
    print("\nBy compartment x status:")
    print("  " + "compartment".ljust(15) + "".join(s.rjust(10) for s in statuses))
    for c in sorted({c for (c, _) in by_compartment_status}):
        row = "  " + c.ljust(15)
        for s in statuses:
            row += str(by_compartment_status.get((c, s), 0)).rjust(10)
        print(row)

    sets = sorted({tag for (tag, _) in by_set_status})
    if sets:
        print("\nBy set x status:")
        print("  " + "set".ljust(20) + "".join(s.rjust(10) for s in statuses))
        for tag in sets:
            row = "  " + tag.ljust(20)
            for s in statuses:
                row += str(by_set_status.get((tag, s), 0)).rjust(10)
            print(row)

    stale_running = [j for j in jobs if j["status"] == "running"]
    if stale_running:
        print(f"\nWARNING: {len(stale_running)} job(s) stuck at 'running' with no "
              f"active process (orphaned from a previous killed run).")
        print("  Run:  batch_simulation reset --manifest <path>  to reset to 'pending'.")
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
    return Path("experiment/setup/simulation/batch")


def _default_log_dir() -> Path:
    return Path("experiment/setup/simulation/batch/logs")


def _parse_gpu_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _build_sim_params_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    base = _load_sim_config(args.sim_config)
    return _build_sim_params(
        base=base,
        sim_time=args.sim_time,
        num_spins=args.num_spins,
        time_step=args.time_step,
        nseg=args.nseg,
        d0_intra=args.D0_intra,
        d0_extra=args.D0_extra,
    )


def _cmd_build(args: argparse.Namespace) -> None:
    sim_params = _build_sim_params_from_args(args)
    data = build_sim_manifest(
        substrate_manifest_path=Path(args.substrate_manifest),
        sim_params_template=sim_params,
        kind="full",
        limit_substrates=None,
    )
    out = Path(args.out) if args.out else (
        _default_manifest_dir() / f"{data['created']}_manifest_full.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out, data)
    print(f"Wrote {out}")
    print(f"  source={data['source_substrate_manifest']}")
    print(f"  substrates_done={data['n_substrates_done']}  "
          f"skipped={data['n_substrates_skipped']}  jobs={data['n_jobs']}")
    if data["skipped_substrate_job_ids"]:
        print(f"  skipped (no .pkl found): "
              f"{data['skipped_substrate_job_ids'][:10]}"
              + (" ..." if len(data["skipped_substrate_job_ids"]) > 10 else ""))


def _cmd_build_smoke(args: argparse.Namespace) -> None:
    sim_params = _build_sim_params_from_args(args)
    data = build_sim_manifest(
        substrate_manifest_path=Path(args.substrate_manifest),
        sim_params_template=sim_params,
        kind="smoke",
        limit_substrates=args.limit_substrates,
    )
    out = Path(args.out) if args.out else (
        _default_manifest_dir() / f"{data['created']}_manifest_smoke.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out, data)
    print(f"Wrote {out}")
    print(f"  source={data['source_substrate_manifest']}")
    print(f"  substrates_done={data['n_substrates_done']}  "
          f"skipped={data['n_substrates_skipped']}  jobs={data['n_jobs']}")
    for j in data["jobs"]:
        print(f"  {j['job_id']} sub={j['substrate_job_id']} "
              f"rep={j['repeat_idx']} compartment={j['compartment']} "
              f"-> {j['substrate_pkl']}")


def _cmd_run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    gpus = _parse_gpu_list(args.gpus)
    log_dir = args.log_dir or str(_default_log_dir())
    run_manifest(
        manifest_path=manifest_path,
        gpus=gpus,
        limit=args.limit,
        retry_failed=args.retry_failed,
        recover=args.recover,
        dry_run=args.dry_run,
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


def _add_common_build_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--substrate-manifest", type=str, required=True,
                   help="Path to a substrate manifest JSON (input).")
    p.add_argument("--sim-config", type=str, default=DEFAULT_SIM_CONFIG,
                   help=f"Path to sim params JSON (default: {DEFAULT_SIM_CONFIG}).")
    p.add_argument("--sim-time", type=float, required=True,
                   help="Total diffusion time (ms).")
    p.add_argument("--num-spins", type=int, default=None,
                   help="Override num_spins from sim-config.")
    p.add_argument("--time-step", type=float, default=None,
                   help="Override time_step from sim-config.")
    p.add_argument("--nseg", type=int, default=None,
                   help="Override nseg from sim-config.")
    p.add_argument("--D0-intra", type=float, default=None,
                   help="Override D0_intra from sim-config.")
    p.add_argument("--D0-extra", type=float, default=None,
                   help="Override D0_extra from sim-config.")
    p.add_argument("--out", type=str, default=None,
                   help="Output manifest path "
                        "(default: experiment/setup/simulation/batch/<date>_manifest_<kind>.json)")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Batch Monte Carlo simulation across multiple GPUs (NOZOMI)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build full sim manifest from a substrate manifest.")
    _add_common_build_args(p_build)
    p_build.set_defaults(func=_cmd_build)

    p_smoke = sub.add_parser("build-smoke", help="Build a smoke-test sim manifest.")
    _add_common_build_args(p_smoke)
    p_smoke.add_argument("--limit-substrates", type=int, default=None,
                         help="Use only the first N done substrates "
                              "(default: all done substrates in source manifest).")
    p_smoke.set_defaults(func=_cmd_build_smoke)

    p_run = sub.add_parser("run", help="Run pending jobs from a sim manifest.")
    p_run.add_argument("--manifest", type=str, required=True,
                       help="Path to sim manifest JSON.")
    p_run.add_argument("--gpus", type=str, default="0,1,2,3,4,5",
                       help="Comma-separated GPU IDs (default: 0,1,2,3,4,5).")
    p_run.add_argument("--limit", type=int, default=None,
                       help="Run only the first N pending jobs.")
    p_run.add_argument("--retry-failed", action="store_true",
                       help="Also re-run jobs marked status=failed.")
    p_run.add_argument("--recover", action="store_true",
                       help="Also pick up stale 'running' jobs left by a "
                            "previously killed run.")
    p_run.add_argument("--dry-run", action="store_true",
                       help="Print what would run, but do not launch workers.")
    p_run.add_argument("--log-dir", type=str, default=None,
                       help="Override log dir "
                            "(default: experiment/setup/simulation/batch/logs).")
    p_run.set_defaults(func=_cmd_run)

    p_status = sub.add_parser("status", help="Print status summary for a sim manifest.")
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
