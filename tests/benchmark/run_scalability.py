"""
Scalability sweep: runtime vs substrate size for all 4 frameworks.

Geometry (fixed across all sizes):
  - square-packed parallel cylinders, z-oriented
  - radius r = 0.9 um (diameter 1.8 um)
  - ICVF = 0.65 -> separation sep = r * sqrt(pi / ICVF) = 1.979 um

Varied:
  - box side L via n_cyl_per_side ~ round(L / sep)
  - default sweep: L in {25, 50, 100} um -> n in {13, 25, 51}

Shot-point: only b = 1000 s/mm^2 is simulated (dropping the other b-values
gives ~4x speedup for Camino/Disimpy/MC-DC with no loss for Nozomi, since
this experiment measures *runtime*, not signal accuracy).

All PGSE, dt, TE are inherited from the main CFG.

Output:
    results/scalability/L<LL>/<fw>/run_NN.json

Usage:
    # default: L = 25, 50, 100, 3 repeats, 10k walkers, all 4 frameworks
    python -m tests.benchmark.run_scalability

    # subset
    python -m tests.benchmark.run_scalability --sides 25 50 --frameworks nozomi camino
"""

from __future__ import annotations
import argparse
import dataclasses
import json
import math
import multiprocessing as mp
import time
import traceback
from pathlib import Path

from tests.benchmark.config import CFG, RESULTS_DIR, BenchConfig

# Scalability-specific constants
ICVF_TARGET = 0.65
CYL_RADIUS_M = 0.9e-6                 # 0.9 um (diameter 1.8 um)
DEFAULT_BOX_SIDES_UM = (25, 50, 100)
DEFAULT_N_WALKERS = 10_000
# Only b = 1000 s/mm^2 is simulated in the scalability sweep. This cuts
# Camino / Disimpy / MC-DC cost ~4x (each of those runs one pass per b-value)
# with no effect on Nozomi timings. b = 1000 is a standard diffusion-MRI
# reference b-value and is enough to exercise restricted diffusion here.
SCALE_BVALS_S_MM2 = (0.0, 1000.0,)   # b=0 lets us report true S/S0

SCALE_ROOT = RESULTS_DIR / "scalability"


def _sep_for_icvf(r_m: float, icvf: float) -> float:
    """Square-lattice separation giving the requested ICVF."""
    return r_m * math.sqrt(math.pi / icvf)


def build_cfg_for_L(L_target_um: float,
                    n_walkers: int = DEFAULT_N_WALKERS,
                    base_cfg: BenchConfig = CFG) -> BenchConfig:
    """Return a BenchConfig overriding substrate params for the target L."""
    sep_m = _sep_for_icvf(CYL_RADIUS_M, ICVF_TARGET)
    n = max(1, int(round(L_target_um * 1e-6 / sep_m)))
    return dataclasses.replace(
        base_cfg,
        cylinder_radius_m=CYL_RADIUS_M,
        cylinder_sep_m=sep_m,
        n_cyl_per_side=n,
        n_walkers=int(n_walkers),
        bvals_s_mm2=SCALE_BVALS_S_MM2,
    )


def _out_dir(L_target_um: int, fw: str) -> Path:
    d = SCALE_ROOT / f"L{L_target_um:03d}" / fw
    d.mkdir(parents=True, exist_ok=True)
    return d


def _worker(fw: str, cfg: BenchConfig, repeat: int, out_path: str) -> None:
    """Executed in a *fresh* child process (spawn) so each framework gets a
    clean Python / CUDA / numba state. Disimpy (numba CUDA) and Nozomi
    (pycuda) both grab the primary CUDA context and cannot coexist inside a
    single interpreter.
    """
    import json as _json
    out = Path(out_path)
    try:
        if fw == "nozomi":
            from tests.benchmark.run_nozomi import run_once
            result = run_once(repeat, cfg=cfg)
        elif fw == "camino":
            from tests.benchmark.run_camino import run_once
            # PLY-mesh mode so geometry-lookup cost scales with N_cyl
            result = run_once(repeat, cfg=cfg, use_ply_mesh=True)
        elif fw == "disimpy":
            from tests.benchmark.run_disimpy import run_once
            result = run_once(repeat, cfg=cfg)
        elif fw == "mcdc":
            from tests.benchmark.run_mcdc import run_once
            result = run_once(repeat, cfg=cfg)
        else:
            raise ValueError(f"unknown framework: {fw}")
        out.write_text(_json.dumps({"ok": True, "result": result}, indent=2))
    except Exception:
        out.write_text(_json.dumps({"ok": False,
                                    "error": traceback.format_exc()}))


def _run_one(fw: str, cfg: BenchConfig, repeat: int) -> dict:
    """Launch the worker in a spawn child process and collect its JSON."""
    import tempfile
    with tempfile.NamedTemporaryFile("r+", suffix=".json", delete=False) as tf:
        tmp_path = tf.name

    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_worker, args=(fw, cfg, repeat, tmp_path))
    p.start()
    p.join()
    rc = p.exitcode

    payload = json.loads(Path(tmp_path).read_text())
    Path(tmp_path).unlink(missing_ok=True)
    if rc != 0 or not payload.get("ok", False):
        raise RuntimeError(f"{fw} worker failed (rc={rc}):\n"
                           + payload.get("error", "no error message"))
    return payload["result"]


def sweep(box_sides_um=DEFAULT_BOX_SIDES_UM,
          frameworks=("nozomi", "disimpy", "camino", "mcdc"),
          n_repeats=3,
          n_walkers=DEFAULT_N_WALKERS) -> None:
    SCALE_ROOT.mkdir(parents=True, exist_ok=True)

    for L_um in box_sides_um:
        cfg = build_cfg_for_L(L_um, n_walkers=n_walkers)
        actual_L_um = cfg.n_cyl_per_side * cfg.cylinder_sep_m * 1e6
        n_cyl = cfg.n_cyl_per_side ** 2
        print(f"\n=== L_target={L_um} um  ->  n_per_side={cfg.n_cyl_per_side}  "
              f"actual_L={actual_L_um:.2f} um  n_cyl={n_cyl}  "
              f"ICVF={cfg.icvf:.3f}  walkers={cfg.n_walkers} ===")

        for fw in frameworks:
            out_d = _out_dir(L_um, fw)
            for r in range(n_repeats):
                print(f"  [{fw}] repeat {r+1}/{n_repeats} ...", flush=True)
                t0 = time.time()
                try:
                    result = _run_one(fw, cfg, r)
                except Exception as exc:
                    print(f"    FAILED: {exc}")
                    continue
                # Annotate with scalability metadata so plotting is easy.
                result["scalability_L_target_um"] = int(L_um)
                result["scalability_L_actual_um"] = float(actual_L_um)
                result["scalability_n_cyl"] = int(n_cyl)
                result["scalability_icvf"] = float(cfg.icvf)
                (out_d / f"run_{r:02d}.json").write_text(
                    json.dumps(result, indent=2))
                print(f"    done in {time.time()-t0:.1f} s "
                      f"(elapsed_s={result['elapsed_s']:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sides", type=int, nargs="+",
                    default=list(DEFAULT_BOX_SIDES_UM),
                    help="Target box sides L in um (e.g. 25 50 100).")
    ap.add_argument("--frameworks", type=str, nargs="+",
                    default=["nozomi", "disimpy", "camino", "mcdc"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--walkers", type=int, default=DEFAULT_N_WALKERS)
    args = ap.parse_args()
    sweep(box_sides_um=args.sides,
          frameworks=args.frameworks,
          n_repeats=args.repeats,
          n_walkers=args.walkers)


if __name__ == "__main__":
    main()
