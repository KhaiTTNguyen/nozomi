"""
Run MC/DC Simulator (jonhrafe/MCDC_Simulator_public) PGSE benchmark.

MC/DC reads a single scheme (.conf) file describing the whole experiment.
We use its native `cylinders_list` substrate (one line per cylinder:
cx cy r, with cylinder axis = z). This matches our canonical geometry
exactly and avoids mesh I/O.

One MC/DC run produces signal for all scheme rows (all b-values) in one pass.

Installation (run once):
    cd tests/benchmark/external
    git clone https://github.com/jonhrafe/MCDC_Simulator_public.git
    cd MCDC_Simulator_public
    mkdir -p build && cd build
    cmake .. && make -j

The resulting binary is ``MC-DC_Simulator`` (adjust ``MCDC_BIN`` in config.py
or via the env var MCDC_BIN if it lives elsewhere).
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np

from tests.benchmark.config import (
    CFG, RESULTS_DIR, SUBSTRATE_DIR, ensure_dirs,
    gmax_T_per_m_for_bvalue_s_mm2, num_steps, seed_for_repeat,
)
from tests.benchmark.substrate import build_cylinder_list_m


def _write_cylinder_list(path: Path, cfg=CFG) -> None:
    """
    MC/DC cylinders_list format (z-oriented):
        first line : ``scale`` factor applied to every coordinate/radius.
        remaining  : ``cx cy cz r``   (detected as z-oriented via 4-token rows)

    MC/DC's internal coordinate units are millimetres, so we store positions
    in metres and use scale = 1e3 (m -> mm) to match the ``scale_from_stu 1``
    conversion applied to diffusivity / duration.
    """
    centers, radii, _, _ = build_cylinder_list_m(cfg)
    with open(path, "w") as f:
        f.write("1000.0\n")  # scale: m -> mm
        for (cx, cy), r in zip(centers, radii):
            f.write(f"{cx:.9e} {cy:.9e} 0.0 {r:.9e}\n")


def _write_scheme(path: Path, cfg=CFG) -> None:
    """MC/DC PGSE scheme (SI). Unit gradient direction + |G| magnitude."""
    lines = ["VERSION: STEJSKALTANNER"]
    gx, gy, gz = cfg.gradient_axis
    for b in cfg.bvals_s_mm2:
        G = gmax_T_per_m_for_bvalue_s_mm2(b)
        if G == 0.0:
            lines.append(
                f"0.0 0.0 0.0 0.0 "
                f"{cfg.Delta_s:.9e} {cfg.delta_s:.9e} {cfg.te_s:.9e}"
            )
        else:
            lines.append(
                f"{gx:.6f} {gy:.6f} {gz:.6f} {G:.9e} "
                f"{cfg.Delta_s:.9e} {cfg.delta_s:.9e} {cfg.te_s:.9e}"
            )
    path.write_text("\n".join(lines) + "\n")


def _write_conf(path: Path, cylinder_list: Path, scheme: Path,
                dwi_out: Path, cfg=CFG, seed: int | None = None,
                ini_walker_flag: str | None = None) -> None:
    """
    MC/DC master config. Voxel coordinates are in the MC/DC internal unit
    system (millimetres). If ``ini_walker_flag`` is 'intra' or 'extra',
    MC/DC seeds walkers exclusively in that compartment against its own
    cylinder geometry (no sampling mismatch). Otherwise walkers are seeded
    uniformly across the voxel.
    """
    _, _, L_m, Lz_m = build_cylinder_list_m(cfg)
    half_L_mm = 1e3 * (L_m / 2.0)
    half_Lz_mm = 1e3 * (Lz_m / 2.0)

    seed_line = f"seed {int(seed)}\n" if (seed is not None and seed > 0) else ""
    ini_line = (f"ini_walkers_pos {ini_walker_flag}\n"
                if ini_walker_flag in ("intra", "extra") else "")
    conf = f"""N {cfg.n_walkers}
T {num_steps(cfg)}
duration {cfg.te_s:.9e}
diffusivity {cfg.D0_m2_s:.9e}
exp_prefix {dwi_out}
scheme_file {scheme}
scale_from_stu 1
write_txt 1
write_bin 0
write_traj_file 0
{seed_line}{ini_line}<obstacle>
cylinders_list {cylinder_list}
</obstacle>
<voxels>
{-half_L_mm:.9e} {-half_L_mm:.9e} {-half_Lz_mm:.9e}
{ half_L_mm:.9e} { half_L_mm:.9e} { half_Lz_mm:.9e}
</voxels>
num_process 0
<END>
"""
    path.write_text(conf)


def run_once(repeat: int, cfg=CFG, compartment: str = "intra") -> dict:
    t_start = time.time()

    work = RESULTS_DIR / "mcdc" / f"run_{repeat:02d}"
    work.mkdir(parents=True, exist_ok=True)
    # Purge any stale outputs so we never pick up a previous run's DWI file.
    for p in work.glob("*DWI*.txt"):
        p.unlink(missing_ok=True)
    for p in work.glob("*simulation_info*.txt"):
        p.unlink(missing_ok=True)

    cyl_path = SUBSTRATE_DIR / "mcdc_cylinders.txt"
    scheme_path = SUBSTRATE_DIR / "mcdc_pgse.scheme"
    conf_path = work / "sim.conf"
    out_prefix = work / "dwi"

    _write_cylinder_list(cyl_path, cfg)
    _write_scheme(scheme_path, cfg)
    seed = seed_for_repeat(repeat, cfg)

    # Write the sim.conf using paths *relative to the conf file's directory*
    # so the artifact does not bake in absolute filesystem paths (which would
    # leak usernames and also break portability when the results tree is
    # copied elsewhere). We run MC/DC below with cwd=work so the relative
    # paths resolve correctly.
    cyl_rel = os.path.relpath(cyl_path, start=work)
    scheme_rel = os.path.relpath(scheme_path, start=work)
    out_prefix_rel = Path("dwi")
    ini_flag = compartment if compartment in ("intra", "extra") else None
    _write_conf(conf_path, Path(cyl_rel), Path(scheme_rel),
                out_prefix_rel, cfg, seed=seed,
                ini_walker_flag=ini_flag)

    mcdc = Path(cfg.mcdc_bin)
    if not mcdc.exists():
        raise FileNotFoundError(
            f"MC/DC binary not found at {mcdc}. See install notes at the top of "
            "tests/benchmark/run_mcdc.py. Set MCDC_BIN env var if built elsewhere."
        )

    proc = subprocess.run([str(mcdc), "--conf", "sim.conf"],
                          cwd=str(work),
                          capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"MC/DC failed (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout.decode(errors='ignore')}\n"
            f"STDERR:\n{proc.stderr.decode(errors='ignore')}"
        )

    # MC/DC writes <prefix>_DWI.txt or <prefix>_rep_NN_DWI.txt depending on
    # build. Prefer the freshest *DWI*.txt in the work dir.
    cands = sorted(work.glob("*DWI*.txt"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise RuntimeError(f"MC/DC did not produce DWI output in {work}")
    dwi_txt = cands[-1]

    signal = np.loadtxt(dwi_txt, dtype=float).reshape(-1)
    s0 = signal[0] if abs(signal[0]) > 1e-12 else 1.0
    signal_norm = (signal / s0).tolist()

    elapsed = time.time() - t_start
    return {
        "framework": "mcdc",
        "repeat": int(repeat),
        "elapsed_s": float(elapsed),
        "bvals_s_mm2": list(cfg.bvals_s_mm2),
        "signal_raw": signal.tolist(),
        "signal": signal_norm,
        "n_walkers": int(cfg.n_walkers),
        "time_step_s": cfg.time_step_s,
        "te_s": cfg.te_s,
        "seed": int(seed),
        "compartment": compartment,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--compartment", type=str,
                    default=os.environ.get("BENCH_COMPARTMENT", "intra"),
                    choices=["all", "intra", "extra"])
    args = ap.parse_args()

    ensure_dirs()
    result = run_once(args.repeat, compartment=args.compartment)

    out = Path(args.out) if args.out else (
        RESULTS_DIR / "mcdc" / f"run_{args.repeat:02d}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
