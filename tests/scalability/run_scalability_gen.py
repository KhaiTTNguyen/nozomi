"""
Substrate-generation scalability sweep.

For each sweep size (number of fibers), this runs the REAL Nozomi substrate
generator (``substrate_main``) in a fresh spawn subprocess and records:
  * wall-clock time of the whole generation call,
  * the pipeline's own generation-only elapsed time (parsed from the output
    pickle filename),
  * peak GPU memory (torch ``max_memory_allocated`` / ``max_memory_reserved``),
  * peak host RSS (``resource.getrusage``),
  * achieved box side L, sphere count, output file size, convergence.

Why a subprocess per size:
  * a clean CUDA / torch state per run (peak-memory counters reset),
  * an out-of-memory or crash in one size does not abort the whole sweep.

Shared-GPU safety:
  * before each run the parent checks free VRAM and computes a per-process
    memory cap so TOTAL card usage stays <= GPU_BUDGET_FRACTION; the worker
    enforces it via ``torch.cuda.set_per_process_memory_fraction``. If a run
    hits the cap it raises a catchable OOM (recorded as ``oom=True``) instead
    of starving other users' processes.

Checkpoint / resume:
  * each completed size writes ``results/runs/N<NNNNNN>.json``; on restart,
    sizes whose JSON already exists with ``ok=True`` are skipped, so a killed
    sweep never reruns finished sizes.

Usage (from repo root, venv active):
    # verification (just N=100, 500)
    python -m tests.scalability.run_scalability_gen --verify

    # full sweep
    python -m tests.scalability.run_scalability_gen

    # custom sizes
    python -m tests.scalability.run_scalability_gen --num-fibers 100 500 1000
"""

from __future__ import annotations
import argparse
import glob
import json
import multiprocessing as mp
import os
import resource
import sys
import time
import traceback
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.scalability.config import (
    BASE_PARAMS, FULL_NUM_FIBERS, VERIFICATION_NUM_FIBERS,
    GPU_BUDGET_FRACTION, DISABLE_PLOTTING, REPO_ROOT,
    RUNS_DIR, SUBSTRATE_OUT_DIR, params_for, ensure_dirs,
)


# ---------------------------------------------------------------------------
# Worker (runs in a fresh spawn subprocess)
# ---------------------------------------------------------------------------
def _disable_plotting() -> dict:
    """Replace substrate_main's plotting calls with no-ops so only generation
    cost is measured (3D/cross-section rendering is irrelevant here and would
    dominate / destabilize at large N).

    Returns the original callables so the worker can invoke them post-hoc
    after timing has been recorded."""
    def _noop(*args, **kwargs):
        return None

    from simulation_toolkit.cli import substrate_main as sm
    originals = {
        "plot_cross_section_z_mid": sm.cross_section_plot.plot_cross_section_z_mid,
        "plot_fibers": sm.fiber_3D_plot.plot_fibers,
        "plot_myelinated_fibers": sm.fiber_3D_plot.plot_myelinated_fibers,
        "plot_along_axon_OD_arclength": sm.orientation_plot.plot_along_axon_OD_arclength,
    }
    sm.cross_section_plot.plot_cross_section_z_mid = _noop
    sm.fiber_3D_plot.plot_fibers = _noop
    sm.fiber_3D_plot.plot_myelinated_fibers = _noop
    sm.orientation_plot.plot_along_axon_OD_arclength = _noop

    from simulation_toolkit.substrate_generator.initialization_2d import Init2D
    Init2D.plot_PBC = _noop

    return originals


def _bypass_param_validation() -> None:
    """Disable the user-facing parameter validator inside this worker only.

    The shared CLI validator caps num_fibers at 1000 as a guardrail for
    interactive users. The scalability experiment is the deliberate, controlled
    case where we push N beyond that to map the time/memory ceiling, so we
    replace the bound check with a no-op here (the production validator file is
    left untouched)."""
    from simulation_toolkit.cli import substrate_main as sm
    sm.validate_parameters = lambda *a, **k: None


def _parse_output_filename(pkl_path: Path) -> dict:
    """Extract box side L and generation-only elapsed seconds from the output
    pickle filename, e.g.
        array500_fibers_boxL_69.0_..._123.45_sec.pkl
    Falls back to None values if the pattern is absent.
    """
    name = pkl_path.name
    out = {"box_L_um": None, "gen_elapsed_s": None}
    # box length
    if "boxL_" in name:
        try:
            seg = name.split("boxL_", 1)[1]
            out["box_L_um"] = float(seg.split("_", 1)[0])
        except (ValueError, IndexError):
            pass
    # generation-only elapsed seconds (the '<x>_sec' token)
    if "_sec" in name:
        try:
            token = name.split("_sec", 1)[0].rsplit("_", 1)[1]
            out["gen_elapsed_s"] = float(token)
        except (ValueError, IndexError):
            pass
    return out


def _sphere_count_and_L(pkl_path: Path) -> dict:
    """Load the substrate pickle and return sphere count + authoritative L."""
    import pickle
    import numpy as np
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # Non-myelinated format: [optimized_fibers, L]
    if isinstance(data, (list, tuple)) and len(data) == 2:
        fibers, L = data
        arr = fibers.cpu().numpy() if hasattr(fibers, "cpu") else np.asarray(fibers)
        return {"n_spheres": int(arr.shape[0]), "box_L_um": float(L)}
    # Myelinated format (not used here): dict with outer_fibers
    if isinstance(data, dict):
        arr = np.asarray(data.get("outer_fibers"))
        return {"n_spheres": int(arr.shape[0]),
                "box_L_um": float(data.get("box_length"))}
    return {"n_spheres": None, "box_L_um": None}


def _worker(params: dict, out_path: str, gpu_mem_fraction: float) -> None:
    out = Path(out_path)
    try:
        import torch
        import simulation_toolkit.toolkit_params as config_params
        from simulation_toolkit.cli.substrate_main import substrate_main

        # Route all generated artifacts into the scalability results folder.
        config_params.OUTPUT_FOLDER_PATH = str(SUBSTRATE_OUT_DIR) + os.sep

        plot_fns = _disable_plotting() if DISABLE_PLOTTING else None
        _bypass_param_validation()

        device_ok = torch.cuda.is_available()
        if device_ok:
            # Force CUDA context init in this fresh spawn process before any
            # memory-stat / cap calls (otherwise "Invalid device argument").
            torch.cuda.set_device(0)
            torch.cuda.init()
            _ = torch.zeros(1, device="cuda:0")
            # Cap this process so total card usage respects the shared budget.
            frac = max(0.05, min(0.95, float(gpu_mem_fraction)))
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            torch.cuda.reset_peak_memory_stats(0)

        exp_folder = f"scalability_N{int(params['num_fibers']):06d}"

        wall_t0 = time.time()
        substrate_main(params, exp_folder)
        wall_elapsed = time.time() - wall_t0

        # Locate the freshly written substrate pickle.
        out_root = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH
        pkls = sorted(glob.glob(os.path.join(out_root, "data", "*.pkl")),
                      key=os.path.getmtime)
        pkl = Path(pkls[-1]) if pkls else None

        result = {
            "ok": True,
            "oom": False,
            "num_fibers": int(params["num_fibers"]),
            "target_volume_fraction": params["target_volume_fraction"],
            "mean_diameter": params["mean_diameter"],
            "wall_elapsed_s": float(wall_elapsed),
            "gen_elapsed_s": None,
            "box_L_um": None,
            "n_spheres": None,
            "output_pickle": str(pkl) if pkl else None,
            "output_bytes": int(pkl.stat().st_size) if pkl else None,
            "peak_gpu_alloc_bytes": None,
            "peak_gpu_reserved_bytes": None,
            "peak_host_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "gpu_mem_fraction_cap": float(gpu_mem_fraction),
        }

        if device_ok:
            result["peak_gpu_alloc_bytes"] = int(torch.cuda.max_memory_allocated(0))
            result["peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))

        if pkl is not None:
            result.update({k: v for k, v in _parse_output_filename(pkl).items()
                           if v is not None})
            try:
                sc = _sphere_count_and_L(pkl)
                result["n_spheres"] = sc["n_spheres"]
                if sc["box_L_um"] is not None:
                    result["box_L_um"] = sc["box_L_um"]
            except Exception:
                pass  # filename-parsed values still stand

        out.write_text(json.dumps(result, indent=2))

        # --- Post-hoc plots (not counted in wall time) ---
        if pkl is not None and plot_fns is not None:
            try:
                import pickle as _pickle
                import numpy as _np
                from matplotlib import cm as _cm
                with open(pkl, "rb") as _f:
                    _pkl_data = _pickle.load(_f)
                if isinstance(_pkl_data, (list, tuple)) and len(_pkl_data) == 2:
                    _fibers, _box_L = _pkl_data
                elif isinstance(_pkl_data, dict):
                    _fibers = _pkl_data.get("outer_fibers")
                    _box_L = _pkl_data.get("box_length")
                else:
                    _fibers = None
                if _fibers is not None:
                    _n_fibers = int(_np.unique(_np.asarray(_fibers)[:, 4]).shape[0])
                    _color = _cm.rainbow(_np.linspace(0.0, 1.0, _n_fibers))
                    _np.random.shuffle(_color)
                    print("  generating cross-section plot...", flush=True)
                    plot_fns["plot_cross_section_z_mid"](
                        outer_fibers=_fibers,
                        inner_fibers=None,
                        box_length=float(_box_L),
                        color=_color,
                    )
                    print("  generating 3D fiber plot...", flush=True)
                    plot_fns["plot_fibers"](
                        _fibers,
                        overlap_indices=None,
                        color=_color,
                        optimized=True,
                        POV="horizontal_90",
                        component_label="outer",
                    )
                    print("  plots saved.", flush=True)
            except Exception as _plot_exc:
                print(f"  [warning] post-hoc plotting failed: {_plot_exc}", flush=True)

    except Exception as exc:  # includes CUDA OOM
        msg = traceback.format_exc()
        is_oom = "out of memory" in msg.lower() or "CUDA out of memory" in msg
        out.write_text(json.dumps({
            "ok": False,
            "oom": bool(is_oom),
            "num_fibers": int(params.get("num_fibers", -1)),
            "error": msg,
        }, indent=2))


# ---------------------------------------------------------------------------
# Parent driver
# ---------------------------------------------------------------------------
def _gpu_free_total_bytes() -> tuple[int, int] | None:
    """Return (free, total) VRAM in bytes for cuda:0, or None if unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(0)
        return int(free), int(total)
    except Exception:
        return None


def _budget_fraction() -> float:
    """Per-process memory fraction that keeps TOTAL card usage within budget,
    accounting for memory other users already hold.

        allowed_for_us = BUDGET*total - used_by_others
                       = BUDGET*total - (total - free)
                       = free - (1 - BUDGET)*total
        fraction = allowed_for_us / total = free/total - (1 - BUDGET)
    """
    ft = _gpu_free_total_bytes()
    if ft is None:
        return GPU_BUDGET_FRACTION
    free, total = ft
    frac = (free / total) - (1.0 - GPU_BUDGET_FRACTION)
    return max(0.05, min(GPU_BUDGET_FRACTION, frac))


def _run_one(num_fibers: int) -> dict:
    out_path = RUNS_DIR / f"N{num_fibers:06d}.json"
    params = params_for(num_fibers)
    frac = _budget_fraction()

    ft = _gpu_free_total_bytes()
    if ft is not None:
        free_gb, total_gb = ft[0] / 1e9, ft[1] / 1e9
        print(f"  GPU free={free_gb:.1f} GB / total={total_gb:.1f} GB  "
              f"-> per-process cap = {frac*100:.0f}% "
              f"({frac*total_gb:.1f} GB)", flush=True)

    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_worker, args=(params, str(out_path), frac))
    t0 = time.time()
    p.start()
    p.join()
    rc = p.exitcode

    if not out_path.exists():
        result = {"ok": False, "oom": rc == -9, "num_fibers": num_fibers,
                  "error": f"worker produced no output (rc={rc})"}
        out_path.write_text(json.dumps(result, indent=2))
        return result

    result = json.loads(out_path.read_text())
    result.setdefault("driver_wall_s", time.time() - t0)
    out_path.write_text(json.dumps(result, indent=2))
    return result


def sweep(num_fibers_list, force: bool = False) -> None:
    ensure_dirs()
    for n in num_fibers_list:
        out_path = RUNS_DIR / f"N{n:06d}.json"
        if out_path.exists() and not force:
            prev = json.loads(out_path.read_text())
            if prev.get("ok"):
                L = prev.get("box_L_um")
                print(f"[N={n}] already done (L={L} um) -> skip", flush=True)
                continue

        print(f"\n=== N_fibers = {n} ===", flush=True)
        result = _run_one(n)

        if result.get("ok"):
            print(f"  done: L={result.get('box_L_um')} um, "
                  f"spheres={result.get('n_spheres')}, "
                  f"gen={result.get('gen_elapsed_s')} s, "
                  f"wall={result.get('wall_elapsed_s'):.1f} s, "
                  f"GPU peak={(result.get('peak_gpu_reserved_bytes') or 0)/1e9:.2f} GB",
                  flush=True)
        else:
            if result.get("oom"):
                print(f"  OUT OF MEMORY at N={n} (budget reached). "
                      f"Stopping sweep -> this is the feasibility ceiling.",
                      flush=True)
                break
            print(f"  FAILED at N={n}:\n{result.get('error','')[:500]}", flush=True)
            break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="Run only the verification sizes (N=100, 500).")
    ap.add_argument("--num-fibers", type=int, nargs="+", default=None,
                    help="Explicit list of fiber counts to sweep.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run sizes even if a result JSON already exists.")
    args = ap.parse_args()

    if args.num_fibers:
        sizes = args.num_fibers
    elif args.verify:
        sizes = list(VERIFICATION_NUM_FIBERS)
    else:
        sizes = list(FULL_NUM_FIBERS)

    print(f"Scalability sweep over num_fibers = {sizes}")
    print(f"Results -> {RUNS_DIR}")
    sweep(sizes, force=args.force)


if __name__ == "__main__":
    main()
