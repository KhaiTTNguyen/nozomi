"""
Run Camino datasynth PGSE benchmark on a mono-disperse square-packed cylinder
substrate. Camino natively supports this geometry (``-geometry cylinder
-packing SQUARE``), so no mesh export is required.

Outputs:
  results/camino/run_NN.json     timing + signal per b-value

Scheme file format: STEJSKALTANNER (SI units).
    # version: STEJSKALTANNER
    Gx  Gy  Gz  |G|  DELTA  delta  TE

Camino writes a stream of 4-byte floats: one float per scheme line per voxel.
We request 1 voxel so the output is simply len(scheme) floats.
"""

from __future__ import annotations
import argparse
import json
import os
import struct
import subprocess
import time
from pathlib import Path
import numpy as np

from tests.benchmark.config import (
    CFG, RESULTS_DIR, SUBSTRATE_DIR, ensure_dirs,
    gmax_T_per_m_for_bvalue_s_mm2, num_steps, seed_for_repeat,
)


def build_scheme_file(path: Path, cfg=CFG) -> list[float]:
    """
    Camino PGSE scheme with one row per b-value, gradient along +x.
    b = 0 is represented as G = 0 (Camino still needs DELTA/delta/TE).
    Returns the list of |G| values (T/m) for bookkeeping.
    """
    gvals = []
    lines = ["VERSION: STEJSKALTANNER"]
    gx, gy, gz = cfg.gradient_axis  # unit direction (e.g. 1, 0, 0)
    for b_s_mm2 in cfg.bvals_s_mm2:
        G = gmax_T_per_m_for_bvalue_s_mm2(b_s_mm2)
        gvals.append(G)
        # Camino STEJSKALTANNER columns: Gx_unit Gy_unit Gz_unit |G| DELTA delta TE
        # The first three columns must be a unit direction; magnitude is in |G|.
        # For b = 0, use zero direction and zero magnitude.
        if G == 0.0:
            lines.append(
                f"0.0 0.0 0.0 0.0 "
                f"{cfg.Delta_s:.9e} {cfg.delta_s:.9e} {cfg.te_s:.9e}"
            )
        else:
            lines.append(
                f"{gx:.6f} {gy:.6f} {gz:.6f} "
                f"{G:.9e} {cfg.Delta_s:.9e} {cfg.delta_s:.9e} {cfg.te_s:.9e}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return gvals


def _read_bfloat(path: Path, n: int) -> np.ndarray:
    raw = path.read_bytes()
    # Camino native is big-endian 4-byte float.
    return np.array(struct.unpack(f">{n}f", raw[:4 * n]), dtype=float)


def run_once(repeat: int, cfg=CFG, use_ply_mesh: bool = False,
             initial: str = "intra") -> dict:
    """Run Camino datasynth for one repeat.

    use_ply_mesh=True switches from analytical SQUARE-packed cylinders to the
    full cylinder mesh (``-geometry ply``). This is used for the scalability
    benchmark so Camino's geometry-lookup cost grows with the substrate size,
    the same as the other frameworks.

    initial: Camino -initial flag. 'intra' is the default benchmark path;
    'uniform' seeds the whole voxel, and 'extra' seeds outside cylinders.
    """
    t_start = time.time()

    # Paths
    work_dir = RESULTS_DIR / "camino"
    work_dir.mkdir(parents=True, exist_ok=True)
    scheme_path = SUBSTRATE_DIR / "pgse.scheme"
    out_path = work_dir / f"run_{repeat:02d}.Bfloat"

    gvals = build_scheme_file(scheme_path, cfg)
    n_b = len(gvals)
    tmax = num_steps(cfg)

    seed = seed_for_repeat(repeat, cfg)
    datasynth = str(Path(cfg.camino_bin) / "datasynth")
    cmd = [
        datasynth,
        "-walkers", str(cfg.n_walkers),
        "-tmax", str(tmax),
        "-p", "0.0",                       # impermeable membranes
        "-initial", initial,                # uniform | intra | extra
        "-schemefile", str(scheme_path),
        "-voxels", "1",
        "-diffusivity", f"{cfg.D0_m2_s:.9e}",
        "-seed", str(seed),
    ]

    if use_ply_mesh:
        # Regenerate the PLY mesh for this cfg (scalability sweep runs many
        # different substrate sizes through the same SUBSTRATE_DIR).
        # cylinder_length_factor=1.0 makes the mesh z-extent exactly box_z_m
        # so it fits inside Camino's periodic cell (-meshsep). Camino's PBC
        # in z then keeps walkers inside the cylinder cores.
        from tests.benchmark.substrate import export_ply_mesh
        ply_path = SUBSTRATE_DIR / f"substrate_camino_n{cfg.n_cyl_per_side}.ply"
        export_ply_mesh(ply_path, cfg, cylinder_length_factor=1.0)
        Lx = cfg.box_xy_m
        Ly = cfg.box_xy_m
        Lz = cfg.box_z_m
        cmd += [
            "-geometry", "ply",
            "-plyfile", str(ply_path),
            "-meshsep", f"{Lx:.9e}", f"{Ly:.9e}", f"{Lz:.9e}",
        ]
    else:
        cmd += [
            "-geometry", "cylinder",
            "-packing", "SQUARE",
            "-cylinderrad", f"{cfg.cylinder_radius_m:.9e}",
            "-cylindersep", f"{cfg.cylinder_sep_m:.9e}",
        ]

    with open(out_path, "wb") as fout:
        proc = subprocess.run(cmd, stdout=fout, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Camino failed:\n{proc.stderr.decode(errors='ignore')}")

    signal = _read_bfloat(out_path, n_b)
    # Normalize by b=0 signal for fair cross-framework comparison.
    s0 = signal[0] if abs(signal[0]) > 0 else 1.0
    signal_norm = (signal / s0).tolist()

    elapsed = time.time() - t_start
    return {
        "framework": "camino",
        "repeat": int(repeat),
        "elapsed_s": float(elapsed),
        "bvals_s_mm2": list(cfg.bvals_s_mm2),
        "signal_raw": signal.tolist(),
        "signal": signal_norm,
        "n_walkers": int(cfg.n_walkers),
        "time_step_s": cfg.time_step_s,
        "te_s": cfg.te_s,
        "tmax_steps": int(tmax),
        "G_T_per_m": gvals,
        "geometry_mode": "ply_mesh" if use_ply_mesh else "square_packing",
        "seed": int(seed),
        "initial": initial,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    _comp_env = os.environ.get("BENCH_COMPARTMENT", "intra")
    # Camino's flag is 'uniform' for 'all'.
    _camino_initial = {"all": "uniform", "intra": "intra", "extra": "extra"}.get(
        _comp_env, "uniform")
    ap.add_argument("--compartment", type=str, default=_camino_initial,
                    choices=["uniform", "intra", "extra"])
    args = ap.parse_args()

    ensure_dirs()
    result = run_once(args.repeat, initial=args.compartment)

    out = Path(args.out) if args.out else (
        RESULTS_DIR / "camino" / f"run_{args.repeat:02d}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
