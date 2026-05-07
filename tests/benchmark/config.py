"""
Shared benchmark configuration.

All four frameworks (Nozomi, Disimpy, Camino, MC/DC) consume these numbers so
the comparison is apples-to-apples.

Units:
  - Camino and Disimpy use SI (m, s, T).
  - Nozomi uses um, ms, mT/m (conversions inside run_nozomi.py).
"""

from __future__ import annotations
import os
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = BENCH_DIR / "results"
# Per-invocation results dir. run_all.sh creates a timestamped subfolder and
# exports BENCH_RESULTS_DIR so every runner in that sweep writes into the same
# place. Falls back to results/latest (symlink to most recent sweep) if no env
# override is set, or to results/ itself if nothing has been run yet.
_env_dir = os.environ.get("BENCH_RESULTS_DIR")
if _env_dir:
    RESULTS_DIR = Path(_env_dir).resolve()
elif (RESULTS_ROOT / "latest").exists():
    RESULTS_DIR = (RESULTS_ROOT / "latest").resolve()
else:
    RESULTS_DIR = RESULTS_ROOT
# Keep substrate artifacts inside the timestamped result folder so each sweep
# records the exact geometry and scheme files that were used.
SUBSTRATE_DIR = RESULTS_DIR / "substrate_cache"


@dataclass(frozen=True)
class BenchConfig:
    # --- Substrate (mono-disperse square-packed parallel cylinders, z-oriented) ---
    # r = 4.0 um -> diameter = 8.0 um (large-caliber spinal-cord axon, e.g.
    # ventral corticospinal tract). At Gmax = 80 mT/m (clinical) and b = 2500
    # this size sits just above the resolution floor: the shortest realisable
    # PGSE (delta = Delta = 20.3 ms, td = 13.6 ms) gives ~13% restricted
    # attenuation at b = 2500, well above MC shot noise. sep = r sqrt(pi/ICVF),
    # ICVF = 0.503.
    cylinder_radius_m: float = 4.0e-6       # 4.0 um (diameter 8.0 um)
    cylinder_sep_m: float = 9.996568e-6     # lattice period (ICVF = pi r^2 / s^2 = 0.503)
    n_cyl_per_side: int = 10                # lattice 10x10 -> box ~100 um >> sqrt(2 D TE) ~ 12.7 um
    box_z_m: float = 20.0e-6                # cylinder length / simulation voxel z

    # Derived box
    @property
    def box_xy_m(self) -> float:
        return self.cylinder_sep_m * self.n_cyl_per_side

    @property
    def icvf(self) -> float:
        import math
        return math.pi * self.cylinder_radius_m**2 / self.cylinder_sep_m**2

    # --- Diffusion physics ---
    D0_m2_s: float = 2.0e-9                 # free water @ 37 C (single compartment, no exchange)

    # --- Sequence (PGSE, clinical spinal-cord protocol) ---
    # Floor protocol at Gmax = 80 mT/m for b_max = 2500 s/mm^2:
    # the shortest achievable diffusion time is td = (2/3) * (3 b /
    # (2 gamma^2 G_max^2))^(1/3), reached when delta = Delta (back-to-back
    # gradient pulses, no gap). For b = 2500 this gives delta = Delta ~
    # 20.3 ms, td = 13.6 ms, TE = 40.3 ms, with G_max@b=2500 = 80 mT/m
    # (saturating the hardware). At R = 4 um this yields 13% perpendicular
    # restricted attenuation -- the best a clinical scanner can do.
    delta_s: float = 20.3e-3                # little delta
    Delta_s: float = 20.3e-3                # big delta (back-to-back: Delta = delta)
    te_s: float = 40.6e-3                   # minimum TE for PGSE: Delta + delta
    gradient_axis: tuple = (1.0, 0.0, 0.0)  # x-axis only
    bvals_s_mm2: tuple = (0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0)

    # --- Monte Carlo ---
    n_walkers: int = 50000
    time_step_s: float = 2.0e-6             # 2 us -> step sqrt(6 D dt) = 0.155 um ~ r/6
    seed_mode: str = "intra"                # default benchmark seeding compartment

    # --- Experiment ---
    n_repeats: int = 3
    base_seed: int = 12345                  # deterministic seed per repeat
                                            # (effective seed = base_seed + repeat)

    # --- Paths to external binaries ---
    # Camino: set CAMINO_BIN to the directory containing `datasynth`.
    # Defaults to a sibling `camino/bin/` next to the repo root.
    camino_bin: str = os.environ.get(
        "CAMINO_BIN", str(BENCH_DIR.parents[1].parent / "camino" / "bin")
    )
    mcdc_bin: str = os.environ.get(
        "MCDC_BIN", str(BENCH_DIR / "external" / "MCDC_Simulator_public" / "MC-DC_Simulator")
    )


CFG = BenchConfig()

# Allow quick overrides from the environment (used by trial scripts).
_env_nw = os.environ.get("BENCH_N_WALKERS")
if _env_nw:
    CFG = dataclasses.replace(CFG, n_walkers=int(_env_nw))
_env_nr = os.environ.get("BENCH_N_REPEATS")
if _env_nr:
    CFG = dataclasses.replace(CFG, n_repeats=int(_env_nr))
_env_axis = os.environ.get("BENCH_GRADIENT_AXIS")
if _env_axis:
    parts = [float(v) for v in _env_axis.split(",")]
    if len(parts) != 3:
        raise ValueError(f"BENCH_GRADIENT_AXIS must be 'gx,gy,gz', got {_env_axis!r}")
    CFG = dataclasses.replace(CFG, gradient_axis=tuple(parts))

_env_dt = os.environ.get("BENCH_TIME_STEP_S")
if _env_dt:
    CFG = dataclasses.replace(CFG, time_step_s=float(_env_dt))


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBSTRATE_DIR.mkdir(parents=True, exist_ok=True)
    for fw in ("nozomi", "disimpy", "camino", "mcdc"):
        (RESULTS_DIR / fw).mkdir(parents=True, exist_ok=True)


# --- b-value / G conversions ---------------------------------------------------
# Proton gyromagnetic ratio. All four frameworks agree on this value:
#   Nozomi  : 267.513    rad/(ms mT)     -> 2.67513e8    rad/(s T)
#   Disimpy : 267.513e6  rad/(s T)
#   MC/DC   : 267.51525e3 rad/(ms T)     -> 2.6751525e8  rad/(s T)
#   Camino  : 2.6751525E8 rad/(s T)
GAMMA_RAD_S_T = 2.6751525e8


def gmax_T_per_m_for_bvalue_s_mm2(b_s_mm2: float,
                                  delta_s: float = CFG.delta_s,
                                  Delta_s: float = CFG.Delta_s) -> float:
    """Return gradient amplitude in T/m for a target b-value (s/mm^2)."""
    if b_s_mm2 <= 0:
        return 0.0
    b_s_m2 = b_s_mm2 * 1.0e6          # s/mm^2 -> s/m^2
    denom = (GAMMA_RAD_S_T * delta_s) ** 2 * (Delta_s - delta_s / 3.0)
    import math
    return math.sqrt(b_s_m2 / denom)


def num_steps(cfg: BenchConfig = CFG) -> int:
    import math
    return int(math.ceil(cfg.te_s / cfg.time_step_s))


def seed_for_repeat(repeat: int, cfg: BenchConfig = CFG) -> int:
    """Deterministic per-repeat seed shared by all four frameworks.

    Repeats are offset from base_seed so each repeat is an independent but
    reproducible MC realization.
    """
    return int(cfg.base_seed) + int(repeat)
