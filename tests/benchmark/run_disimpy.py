"""
Run Disimpy PGSE benchmark on the same cylinder substrate.

Disimpy's ``substrates.cylinder`` only supports a single infinite cylinder.
For a multi-cylinder square-packed substrate we use ``substrates.mesh``
with the same lateral-surface mesh that powers the other frameworks.

Install once:
    pip install disimpy
"""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np

from tests.benchmark.config import (
    CFG, RESULTS_DIR, ensure_dirs, gmax_T_per_m_for_bvalue_s_mm2, num_steps,
    seed_for_repeat,
)
from tests.benchmark.substrate import build_cylinder_mesh


def _build_disimpy_substrate(cfg=CFG, init_pos="uniform"):
    """
    Build a disimpy mesh substrate. Vertices/faces describe the lateral
    surfaces of all cylinders; ``padding`` is set so the periodic voxel
    matches [-L/2, L/2]^2 in xy (cylinders sit strictly inside the mesh
    bbox by the half-grid-step, so we pad to recover the full box).

    ``init_pos`` may be the strings 'uniform', 'intra', or 'extra': disimpy
    samples initial walker positions against the mesh geometry itself, so
    no external sampling (or geometry-mismatch risk) is introduced.
    """
    from disimpy import substrates

    vertices, faces, (hx, hy, hz) = build_cylinder_mesh(cfg)
    # Mesh bbox half-extents (cylinders don't reach box walls by design).
    bbox_half_x = float(np.max(np.abs(vertices[:, 0])))
    bbox_half_y = float(np.max(np.abs(vertices[:, 1])))
    pad_x = max(0.0, hx - bbox_half_x)
    pad_y = max(0.0, hy - bbox_half_y)
    padding = np.array([pad_x, pad_y, 0.0], dtype=float)

    return substrates.mesh(
        vertices=vertices.astype(np.float64),
        faces=faces.astype(np.int32),
        periodic=True,
        padding=padding,
        init_pos=init_pos,
        quiet=True,
    )


def _build_gradient_array(cfg=CFG):
    """
    Build disimpy gradient array g of shape (n_measurements, n_t, 3) in T/m
    and corresponding dt (seconds). Gradient axis = +x. Rectangular PGSE
    lobes centered on TE/2.
    """
    n_t = num_steps(cfg) + 1
    dt_s = cfg.te_s / (n_t - 1)
    t = np.linspace(0.0, cfg.te_s, n_t)
    axis = np.asarray(cfg.gradient_axis, dtype=float)

    lobe1 = (t >= (cfg.te_s / 2 - cfg.Delta_s / 2 - cfg.delta_s / 2)) & \
            (t <  (cfg.te_s / 2 - cfg.Delta_s / 2 + cfg.delta_s / 2))
    lobe2 = (t >= (cfg.te_s / 2 + cfg.Delta_s / 2 - cfg.delta_s / 2)) & \
            (t <  (cfg.te_s / 2 + cfg.Delta_s / 2 + cfg.delta_s / 2))

    n_b = len(cfg.bvals_s_mm2)
    g = np.zeros((n_b, n_t, 3), dtype=np.float64)
    for i, b in enumerate(cfg.bvals_s_mm2):
        G = gmax_T_per_m_for_bvalue_s_mm2(b)
        waveform = np.zeros(n_t)
        waveform[lobe1] = +G
        waveform[lobe2] = -G
        g[i, :, :] = waveform[:, None] * axis[None, :]
    return g, dt_s


def run_once(repeat: int, cfg=CFG, compartment: str = "intra") -> dict:
    t_start = time.time()

    from disimpy import simulations

    seed = seed_for_repeat(repeat, cfg)
    init_pos = "uniform" if compartment == "all" else compartment
    substrate = _build_disimpy_substrate(cfg, init_pos=init_pos)
    gradient, dt_s = _build_gradient_array(cfg)
    n_t = gradient.shape[1]
    signal = simulations.simulation(
        n_walkers=int(cfg.n_walkers),
        diffusivity=float(cfg.D0_m2_s),
        gradient=gradient,
        dt=float(dt_s),
        substrate=substrate,
        seed=seed,
        quiet=True,
    )
    signal = np.asarray(signal).real

    s0 = signal[0] if abs(signal[0]) > 1e-12 else 1.0
    signal_norm = (signal / s0).tolist()

    elapsed = time.time() - t_start
    return {
        "framework": "disimpy",
        "repeat": int(repeat),
        "elapsed_s": float(elapsed),
        "bvals_s_mm2": list(cfg.bvals_s_mm2),
        "signal_raw": signal.tolist(),
        "signal": signal_norm,
        "n_walkers": int(cfg.n_walkers),
        "time_step_s": float(dt_s),
        "te_s": cfg.te_s,
        "n_timesteps": int(n_t),
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
        RESULTS_DIR / "disimpy" / f"run_{args.repeat:02d}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
