"""
Run Nozomi PGSE benchmark.

- Builds parallel-cylinder substrate (test_03-style Structure3D with edge ghosts).
- Seeds walkers inside the cylinder compartment by default, matching the
    intra-axonal analytic benchmark.
- Simulates phase in the x-direction, computes S(b) at configured b-values.
- Records full wall-clock time including substrate build & GPU setup.

CLI:
        python -m tests.benchmark.run_nozomi --repeat 0 --compartment intra
"""

from __future__ import annotations
import argparse
import json
import os
import random
import time
from pathlib import Path
import numpy as np

# nozomi imports - must come after pycuda.autoinit
import pycuda.autoinit  # noqa: F401

from simulation_toolkit.simulation_engine.diffsim3d import DwiSim3d
from simulation_toolkit.simulation_engine.geometry import SimGeometry3D, Structure3D
from simulation_toolkit.simulation_engine.waveforms import PGDiffWaveform

from tests.benchmark.config import CFG, RESULTS_DIR, ensure_dirs, seed_for_repeat
from tests.benchmark.substrate import build_cylinder_list_m


# Nozomi native units: micrometers, milliseconds, mT/m.
M_TO_UM = 1e6
S_TO_MS = 1e3


def _build_sg3(cfg=CFG, n_seg: int = 200, use_ghosts: bool = True):
    centers_m, radii_m, L_m, Lz_m = build_cylinder_list_m(cfg)
    lx = ly = L_m * M_TO_UM
    lz = Lz_m * M_TO_UM
    D0 = cfg.D0_m2_s * 1e12 / 1e3  # m^2/s -> um^2/ms  (1e12 um^2/m^2, 1e-3 s/ms)
    t2 = 1e10  # effectively no T2 decay
    rho = 1.0

    sg3 = SimGeometry3D(lx, ly, lz, D0, t2, rho)

    extension_segments = max(1, int(round(30 * n_seg / 200)))
    base_z = np.linspace(-lz / 2.0, lz / 2.0, n_seg)

    edge_distance_m = L_m / 5.0
    for (cx_m, cy_m), r_m in zip(centers_m, radii_m):
        r_um = r_m * M_TO_UM
        dx_list = (-L_m, 0.0, L_m) if use_ghosts else (0.0,)
        dy_list = (-L_m, 0.0, L_m) if use_ghosts else (0.0,)
        for dx in dx_list:
            if use_ghosts and dx != 0.0 and abs(cx_m) + abs(dx) - L_m / 2.0 > edge_distance_m:
                continue
            for dy in dy_list:
                if use_ghosts and dy != 0.0 and abs(cy_m) + abs(dy) - L_m / 2.0 > edge_distance_m:
                    continue
                sx = np.full_like(base_z, (cx_m + dx) * M_TO_UM)
                sy = np.full_like(base_z, (cy_m + dy) * M_TO_UM)
                sz = base_z.copy()
                sr = np.full_like(base_z, r_um)
                sz_ext = np.concatenate((sz[-extension_segments:] - lz, sz,
                                         sz[:extension_segments] + lz))
                sx_ext = np.concatenate((sx[-extension_segments:], sx,
                                         sx[:extension_segments]))
                sy_ext = np.concatenate((sy[-extension_segments:], sy,
                                         sy[:extension_segments]))
                sr_ext = np.concatenate((sr[-extension_segments:], sr,
                                         sr[:extension_segments]))
                sg3.add_structure(Structure3D(sx_ext, sy_ext, sz_ext, sr_ext,
                                              D0, t2, rho))
    return sg3, (lx, ly, lz)


def run_once(repeat: int, cfg=CFG, n_seg: int = 200,
             use_ghosts: bool = True,
             nsegx: int = 20, nsegy: int = 20, nsegz: int = 10,
             compartment: str = "intra") -> dict:
    """
    compartment: 'intra'  -> seed only inside cylinders (default benchmark path)
    """
    t_start = time.time()

    # Deterministic seeding: Nozomi's position-seeder uses stdlib `random`
    # internally; the GPU walker RNG is seeded via `initstates_int`.
    seed = seed_for_repeat(repeat, cfg)
    random.seed(seed)
    np.random.seed(seed)

    # --- Geometry ---
    sg3, (lx, ly, lz) = _build_sg3(cfg, n_seg=n_seg, use_ghosts=use_ghosts)

    # --- Diffusion direction (from cfg; default x-axis) ---
    _axis = np.asarray(cfg.gradient_axis, dtype=float).reshape(3)
    diffdir = _axis.reshape(3, 1).astype(np.float32)

    # --- Simulator ---
    spins = int(cfg.n_walkers)
    sim = DwiSim3d(sg3, spins)
    sim.set_diffusion_directions(diffdir=diffdir)
    sim.set_segments(nsegx=nsegx, nsegy=nsegy, nsegz=nsegz)

    # Seed walkers into the chosen compartment.
    # Structure indices:
    #   0 .. sg3.nstructures-1   are the individual cylinders (intra)
    #   sg3.nstructures          is the external box (extra)
    if compartment == "all":
        structs = list(np.arange(0, sg3.nstructures + 1))
    elif compartment == "intra":
        structs = list(np.arange(0, sg3.nstructures))
    elif compartment == "extra":
        structs = [int(sg3.nstructures)]
    else:
        raise ValueError(f"compartment must be 'all'|'intra'|'extra', got {compartment!r}")
    sim.setup(structures=structs, initstates_int=np.int32(seed))

    # --- Waveform (units: ms) ---
    big_delta_ms = cfg.Delta_s * S_TO_MS
    little_delta_ms = cfg.delta_s * S_TO_MS
    te_ms = cfg.te_s * S_TO_MS
    time_step_ms = cfg.time_step_s * S_TO_MS

    gwave = PGDiffWaveform(big_delta=big_delta_ms,
                           little_delta=little_delta_ms,
                           te=te_ms,
                           time_step=time_step_ms)

    # --- Simulate phase ---
    phase_sig = sim.simulate_multi_directions(gwave)[0, :]  # (nspins,)

    # --- Map b-values to gmax and compute S(b) = <cos(g * phase)> ---
    b_base_ms_um2 = gwave.calculate_bvalue_from_wave()  # b for gmax=1 mT/m
    signals = []
    for b_s_mm2 in cfg.bvals_s_mm2:
        b_target_ms_um2 = b_s_mm2 / 1000.0
        gmax = float(np.sqrt(b_target_ms_um2 / b_base_ms_um2)) if b_target_ms_um2 > 0 else 0.0
        signals.append(float(np.mean(np.cos(gmax * phase_sig))))

    elapsed = time.time() - t_start
    return {
        "framework": "nozomi",
        "repeat": int(repeat),
        "elapsed_s": float(elapsed),
        "bvals_s_mm2": list(cfg.bvals_s_mm2),
        "seed": int(seed),
        "signal": signals,
        "n_walkers": int(spins),
        "time_step_s": cfg.time_step_s,
        "te_s": cfg.te_s,
        "n_structures": int(sg3.nstructures),
        "box_um": [lx, ly, lz],
        "n_seg": int(n_seg),
        "use_ghosts": bool(use_ghosts),
        "nsegx": int(nsegx), "nsegy": int(nsegy), "nsegz": int(nsegz),
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
        RESULTS_DIR / "nozomi" / f"run_{args.repeat:02d}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
