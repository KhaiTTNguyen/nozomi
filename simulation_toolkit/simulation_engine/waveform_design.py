"""
waveform_design.py
==================
Automatic, b-value-driven gradient waveform design pipeline.

Given real-scan inputs (target b-value, TE, timing) and a scanner hardware
preset (Gmax / slew rate), this module builds a diffusion gradient waveform,
solves the peak gradient amplitude required to reach the target b-value, and
flags the acquisition as infeasible when the required Gmax exceeds the hardware
limit.

Supported shapes
----------------
- ``pgse``              ideal rectangular PGSE (no slew ramps)
- ``pgse-slew``         slew-limited trapezoidal-lobe PGSE
- ``ogse-apodized``     apodized-cosine OGSE (Does et al. 2003)
- ``ogse-trapezoidal``  Xu-style slew-limited trapezoidal-cosine OGSE
- ``ogse-cosine``       non-apodized cosine OGSE (validation only)

Design mode
-----------
b-value-driven: the normalized (Gmax=1) waveform shape is built, its b-value is
computed numerically from the waveform, and Gmax is solved as
``Gmax = sqrt(b_target / b_at_gmax1)``. For slew-limited shapes the ramp time
``trise = Gmax / slew`` couples back into the shape, so Gmax is found by a
fixed-point iteration.

Outputs
-------
- ``<name>.pkl`` / ``<name>.json`` via :mod:`waveform_io` (JSON carries the
  scalar specs plus the solved physical Gmax).
- ``<name>.png`` waveform plot.
- ``<name>_design.json`` design summary (target/actual b, Gmax, feasibility).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation_toolkit.simulation_engine.waveforms import (
    PGDiffWaveform,
    TrapezoidalPGSEWaveform,
    ApodizedCosineOGSEWaveform,
    CosineOGSEWaveform,
    TrapezoidalCosineOGSEWaveform,
    dt0,
)

# Fixed 180-degree refocusing pulse gap reserved between OGSE encoding blocks.
REFOCUS_GAP_MS = 3.2

# 1 s/mm^2 = 1e-3 ms/um^2
_S_MM2_TO_MS_UM2 = 1e-3


@dataclass(frozen=True)
class HardwarePreset:
    name: str
    gmax_mT_per_m: float
    slew_mT_per_m_per_ms: float


# animal_placeholder holds NaN limits until the real preclinical scanner
# gradient/slew numbers are provided.
HARDWARE_PRESETS: dict[str, HardwarePreset] = {
    "human_80_100": HardwarePreset("human_80_100", 80.0, 100.0),
    "human_40_200": HardwarePreset("human_40_200", 40.0, 200.0),
    # 15.2T Bruker Biospec Avance III preclinical gradient/slew limits.
    "animal_15p2T": HardwarePreset("animal_15p2T", 1000.0, 5000.0),
    "animal_placeholder": HardwarePreset("animal_placeholder", float("nan"), float("nan")),
}

# User-facing shape name -> (internal kind, slew-limited?)
_SHAPES = {
    "pgse": ("pgse_ideal", False),
    "pgse-slew": ("pgse_trap", True),
    "ogse-apodized": ("ogse_apodized", False),
    "ogse-trapezoidal": ("ogse_trap", True),
    "ogse-cosine": ("ogse_cosine", False),
}


@dataclass
class DesignResult:
    shape: str
    preset: str
    te_ms: float
    dt_ms: float
    bvalue_s_mm2_target: float
    bvalue_s_mm2_actual: float
    gmax_mT_per_m: float
    slew_mT_per_m_per_ms: float
    feasible: bool
    trise_ms: Optional[float] = None
    t_eff_ms: Optional[float] = None
    frequency_hz: Optional[float] = None
    n_cycles: Optional[int] = None
    T_duration_ms: Optional[float] = None
    separation_ms: Optional[float] = None
    big_delta_ms: Optional[float] = None
    little_delta_ms: Optional[float] = None
    messages: list[str] = field(default_factory=list)
    waveform: object = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# OGSE timing resolution
# ---------------------------------------------------------------------------
def resolve_ogse_timing(
    n_cycles: int,
    t_eff_ms: Optional[float] = None,
    frequency_hz: Optional[float] = None,
) -> tuple[int, float, float, float]:
    """
    Resolve OGSE timing from N cycles plus exactly one of {t_eff, frequency}.

    Relations (f in cycles/ms, f_hz = 1000 f):
        t_eff = T / (4N) = 1 / (4 f)  ->  t_eff_ms = 250 / f_hz
        T     = 4 N t_eff = N / f

    Returns (N, T_duration_ms, t_eff_ms, frequency_hz).
    """
    n = int(n_cycles)
    if n <= 0:
        raise ValueError("n_cycles must be a positive integer.")
    if (t_eff_ms is None) == (frequency_hz is None):
        raise ValueError("Provide exactly one of t_eff_ms or frequency_hz for OGSE timing.")

    if t_eff_ms is not None:
        t_eff = float(t_eff_ms)
        if t_eff <= 0:
            raise ValueError("t_eff_ms must be positive.")
        f_hz = 250.0 / t_eff
    else:
        f_hz = float(frequency_hz)
        if f_hz <= 0:
            raise ValueError("frequency_hz must be positive.")
        t_eff = 250.0 / f_hz

    T = 4.0 * n * t_eff
    return n, T, t_eff, f_hz


# ---------------------------------------------------------------------------
# Waveform builders (normalized to Gmax = 1)
# ---------------------------------------------------------------------------
def _reposition_apodized(w: ApodizedCosineOGSEWaveform, gap_ms: float) -> None:
    """Place the two apodized blocks symmetrically about TE/2 with a fixed gap."""
    T = w.T_duration
    total = 2.0 * T + gap_ms
    margin = 0.5 * (w.te - total)
    if margin < -1e-9:
        raise ValueError(
            f"te={w.te} ms too short for two OGSE blocks (T={T} ms) plus gap={gap_ms} ms."
        )
    info = w.get_waveform_info()
    apod = info["apodization_duration_ms"]
    cosine = info["cosine_duration_ms"]
    f_cosine = info["cosine_frequency_Hz"] / 1000.0
    w.wave[:] = 0.0
    w._generate_single_waveform(margin, 1.0, apod, cosine, f_cosine)
    w._generate_single_waveform(margin + T + gap_ms, -1.0, apod, cosine, f_cosine)
    w.block1_start_ms = margin
    w.block2_start_ms = margin + T + gap_ms


def _build(kind: str, te: float, dt: float, timing: dict, trise: float):
    """Build a normalized (Gmax=1) waveform object for the requested kind."""
    if kind == "pgse_ideal":
        return PGDiffWaveform(
            big_delta=timing["big_delta"],
            little_delta=timing["little_delta"],
            te=te,
            time_step=dt,
        )
    if kind == "pgse_trap":
        return TrapezoidalPGSEWaveform(
            big_delta=timing["big_delta"],
            little_delta=timing["little_delta"],
            te=te,
            trise=trise,
            gmax=1.0,
            time_step=dt,
        )
    if kind == "ogse_apodized":
        w = ApodizedCosineOGSEWaveform(
            N_cycles=timing["N"],
            T_duration=timing["T"],
            te=te,
            gmax=1.0,
            time_step=dt,
        )
        _reposition_apodized(w, REFOCUS_GAP_MS)
        return w
    if kind == "ogse_trap":
        sep = timing["T"] + REFOCUS_GAP_MS
        return TrapezoidalCosineOGSEWaveform(
            N_cycles=timing["N"],
            T_duration=timing["T"],
            separation_duration=sep,
            trise=trise,
            te=te,
            gmax=1.0,
            time_step=dt,
        )
    if kind == "ogse_cosine":
        return CosineOGSEWaveform(
            N_cycles=timing["N"],
            T_duration=timing["T"],
            te=te,
            gmax=1.0,
            time_step=dt,
        )
    raise ValueError(f"Unknown waveform kind: {kind}")


# ---------------------------------------------------------------------------
# Gmax solver
# ---------------------------------------------------------------------------
def _solve_gmax(
    kind: str,
    has_slew: bool,
    target_ms_um2: float,
    te: float,
    dt: float,
    timing: dict,
    slew: float,
    gmax_guess: float,
    max_iter: int = 200,
    tol: float = 1e-6,
):
    """
    Solve Gmax (mT/m) so the waveform reaches ``target_ms_um2``.

    Returns (waveform, gmax, trise, messages).
    """
    messages: list[str] = []

    if not has_slew:
        w = _build(kind, te, dt, timing, trise=0.0)
        b1 = w.calculate_bvalue_from_wave()
        gmax = float(np.sqrt(target_ms_um2 / b1))
        return w, gmax, 0.0, messages

    if not np.isfinite(slew) or slew <= 0:
        raise ValueError("A finite positive slew rate is required for slew-limited shapes.")

    gmax = float(gmax_guess if np.isfinite(gmax_guess) and gmax_guess > 0 else 50.0)
    w = None
    trise = gmax / slew
    for _ in range(max_iter):
        trise = gmax / slew
        w = _build(kind, te, dt, timing, trise=trise)
        b1 = w.calculate_bvalue_from_wave()
        gmax_new = float(np.sqrt(target_ms_um2 / b1))
        if abs(gmax_new - gmax) <= tol * max(1.0, gmax_new):
            gmax = gmax_new
            break
        gmax = gmax_new
    trise = gmax / slew
    w = _build(kind, te, dt, timing, trise=trise)
    return w, gmax, trise, messages


# ---------------------------------------------------------------------------
# Public design entry point
# ---------------------------------------------------------------------------
def design_waveform(
    shape: str,
    bvalue_s_mm2: float,
    te_ms: float,
    preset: str,
    *,
    little_delta_ms: Optional[float] = None,
    big_delta_ms: Optional[float] = None,
    n_cycles: Optional[int] = None,
    t_eff_ms: Optional[float] = None,
    frequency_hz: Optional[float] = None,
    dt_ms: float = dt0,
) -> DesignResult:
    """
    Design a diffusion gradient waveform for a target b-value under a hardware
    preset. See module docstring for details.
    """
    if shape not in _SHAPES:
        raise ValueError(f"shape must be one of {sorted(_SHAPES)}; got {shape!r}.")
    if preset not in HARDWARE_PRESETS:
        raise ValueError(f"preset must be one of {sorted(HARDWARE_PRESETS)}; got {preset!r}.")

    kind, has_slew = _SHAPES[shape]
    hw = HARDWARE_PRESETS[preset]
    target_ms_um2 = float(bvalue_s_mm2) * _S_MM2_TO_MS_UM2

    messages: list[str] = []
    timing: dict = {}
    t_eff = None
    f_hz = None
    n = None
    T = None
    sep = None

    is_pgse = kind in ("pgse_ideal", "pgse_trap")
    if is_pgse:
        if little_delta_ms is None or big_delta_ms is None:
            raise ValueError("PGSE shapes require little_delta_ms and big_delta_ms.")
        timing["little_delta"] = float(little_delta_ms)
        timing["big_delta"] = float(big_delta_ms)
    else:
        if n_cycles is None:
            raise ValueError("OGSE shapes require n_cycles.")
        n, T, t_eff, f_hz = resolve_ogse_timing(n_cycles, t_eff_ms, frequency_hz)
        timing["N"] = n
        timing["T"] = T
        if kind in ("ogse_apodized", "ogse_trap"):
            sep = T + REFOCUS_GAP_MS

    feasible = True
    try:
        waveform, gmax, trise, solve_msgs = _solve_gmax(
            kind=kind,
            has_slew=has_slew,
            target_ms_um2=target_ms_um2,
            te=float(te_ms),
            dt=float(dt_ms),
            timing=timing,
            slew=hw.slew_mT_per_m_per_ms,
            gmax_guess=hw.gmax_mT_per_m,
        )
        messages.extend(solve_msgs)
    except (ValueError, RuntimeError) as exc:
        return DesignResult(
            shape=shape,
            preset=preset,
            te_ms=float(te_ms),
            dt_ms=float(dt_ms),
            bvalue_s_mm2_target=float(bvalue_s_mm2),
            bvalue_s_mm2_actual=float("nan"),
            gmax_mT_per_m=float("nan"),
            slew_mT_per_m_per_ms=hw.slew_mT_per_m_per_ms,
            feasible=False,
            trise_ms=None,
            t_eff_ms=t_eff,
            frequency_hz=f_hz,
            n_cycles=n,
            T_duration_ms=T,
            separation_ms=sep,
            big_delta_ms=big_delta_ms,
            little_delta_ms=little_delta_ms,
            messages=[f"Infeasible: {exc}"],
            waveform=None,
        )

    b1 = waveform.calculate_bvalue_from_wave()
    b_actual_ms_um2 = b1 * gmax ** 2
    b_actual_s_mm2 = b_actual_ms_um2 / _S_MM2_TO_MS_UM2

    # Hardware Gmax feasibility check.
    if np.isfinite(hw.gmax_mT_per_m) and gmax > hw.gmax_mT_per_m + 1e-9:
        feasible = False
        messages.append(
            f"Required Gmax={gmax:.2f} mT/m exceeds preset limit "
            f"{hw.gmax_mT_per_m:.2f} mT/m ({preset})."
        )
    elif not np.isfinite(hw.gmax_mT_per_m):
        messages.append(f"Preset {preset} has no Gmax limit set; feasibility not checked.")

    # Store the solved physical amplitude so waveform_io serialises it into JSON.
    waveform.gmax_mT_per_m = float(gmax)
    waveform.bvalue_s_mm2 = float(b_actual_s_mm2)
    waveform.bvalue_ms_um2 = float(b_actual_ms_um2)

    trise_out = trise if has_slew else None

    return DesignResult(
        shape=shape,
        preset=preset,
        te_ms=float(te_ms),
        dt_ms=float(dt_ms),
        bvalue_s_mm2_target=float(bvalue_s_mm2),
        bvalue_s_mm2_actual=float(b_actual_s_mm2),
        gmax_mT_per_m=float(gmax),
        slew_mT_per_m_per_ms=hw.slew_mT_per_m_per_ms,
        feasible=feasible,
        trise_ms=trise_out,
        t_eff_ms=t_eff,
        frequency_hz=f_hz,
        n_cycles=n,
        T_duration_ms=T,
        separation_ms=sep,
        big_delta_ms=big_delta_ms,
        little_delta_ms=little_delta_ms,
        messages=messages,
        waveform=waveform,
    )


# ---------------------------------------------------------------------------
# Plotting and saving
# ---------------------------------------------------------------------------
def plot_waveform(result: DesignResult, out_path: str, title: Optional[str] = None) -> str:
    """Plot the physical gradient waveform G(t) = Gmax * shape and save to PNG."""
    if result.waveform is None:
        raise ValueError("DesignResult has no waveform to plot (infeasible design).")
    w = result.waveform
    g_phys = w.wave * result.gmax_mT_per_m

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(w.t, g_phys, linewidth=1.6)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("G (mT/m)")
    if title is None:
        title = (
            f"{result.shape}  |  b={result.bvalue_s_mm2_actual:.0f} s/mm²  "
            f"Gmax={result.gmax_mT_per_m:.1f} mT/m  TE={result.te_ms:.0f} ms"
            f"  [{result.preset}]"
        )
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_design(result: DesignResult, output_dir: str, name: Optional[str] = None) -> dict:
    """
    Save a designed waveform: .pkl/.json (waveform_io) + .png plot +
    <name>_design.json summary. Returns a dict of written paths.
    """
    from simulation_toolkit.simulation_engine.helper import waveform_io

    if result.waveform is None:
        raise ValueError("Cannot save an infeasible design (no waveform).")

    os.makedirs(output_dir, exist_ok=True)
    if name is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f"{result.shape}_{result.preset}_b{int(round(result.bvalue_s_mm2_target))}_{stamp}"

    base = os.path.join(output_dir, name)
    waveform_io.save_waveform_object(result.waveform, base)
    png_path = plot_waveform(result, base + ".png")

    summary = {k: v for k, v in asdict(result).items() if k != "waveform"}
    with open(base + "_design.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    return {
        "pkl": base + ".pkl",
        "json": base + ".json",
        "png": png_path,
        "design_json": base + "_design.json",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _design_from_spec(spec: dict) -> DesignResult:
    return design_waveform(
        shape=spec["shape"],
        bvalue_s_mm2=spec["b"],
        te_ms=spec["te"],
        preset=spec["preset"],
        little_delta_ms=spec.get("little_delta"),
        big_delta_ms=spec.get("big_delta"),
        n_cycles=spec.get("n_cycles"),
        t_eff_ms=spec.get("t_eff"),
        frequency_hz=spec.get("frequency_hz"),
        dt_ms=spec.get("dt", dt0),
    )


def _print_result(result: DesignResult) -> None:
    status = "OK" if result.feasible else "INFEASIBLE"
    print(
        f"[{status}] {result.shape} ({result.preset}): "
        f"b_target={result.bvalue_s_mm2_target:.0f} -> b_actual={result.bvalue_s_mm2_actual:.1f} s/mm², "
        f"Gmax={result.gmax_mT_per_m:.2f} mT/m, "
        f"trise={result.trise_ms if result.trise_ms is None else round(result.trise_ms, 4)} ms, "
        f"t_eff={result.t_eff_ms if result.t_eff_ms is None else round(result.t_eff_ms, 4)} ms"
    )
    for msg in result.messages:
        print(f"    - {msg}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Design diffusion gradient waveforms for a target b-value.")
    parser.add_argument("--config", help="JSON file: a single spec dict or a list of spec dicts.")
    parser.add_argument("--shape", choices=sorted(_SHAPES))
    parser.add_argument("--preset", choices=sorted(HARDWARE_PRESETS))
    parser.add_argument("--b", type=float, help="Target b-value (s/mm^2).")
    parser.add_argument("--te", type=float, help="Echo time / waveform window (ms).")
    parser.add_argument("--little-delta", type=float, help="PGSE delta (ms).")
    parser.add_argument("--big-delta", type=float, help="PGSE Delta (ms).")
    parser.add_argument("--n-cycles", type=int, help="OGSE number of cosine cycles.")
    parser.add_argument("--t-eff", type=float, help="OGSE effective diffusion time (ms).")
    parser.add_argument("--frequency-hz", type=float, help="OGSE oscillation frequency (Hz).")
    parser.add_argument("--dt", type=float, default=dt0, help="Waveform sample step (ms).")
    parser.add_argument("--output-dir", default=".", help="Directory to write outputs.")
    parser.add_argument("--name", help="Base name for the single-design output files.")
    args = parser.parse_args(argv)

    if args.config:
        with open(args.config) as fh:
            data = json.load(fh)
        specs = data if isinstance(data, list) else [data]
        for spec in specs:
            spec.setdefault("dt", args.dt)
            result = _design_from_spec(spec)
            _print_result(result)
            if result.feasible:
                paths = save_design(result, args.output_dir, spec.get("name"))
                print(f"    wrote {paths['pkl']}, {paths['png']}")
        return

    if not (args.shape and args.preset and args.b is not None and args.te is not None):
        parser.error("Without --config, provide at least --shape, --preset, --b and --te.")

    result = design_waveform(
        shape=args.shape,
        bvalue_s_mm2=args.b,
        te_ms=args.te,
        preset=args.preset,
        little_delta_ms=args.little_delta,
        big_delta_ms=args.big_delta,
        n_cycles=args.n_cycles,
        t_eff_ms=args.t_eff,
        frequency_hz=args.frequency_hz,
        dt_ms=args.dt,
    )
    _print_result(result)
    if result.feasible:
        paths = save_design(result, args.output_dir, args.name)
        print(f"wrote {paths['pkl']}, {paths['png']}, {paths['design_json']}")


if __name__ == "__main__":
    main()
