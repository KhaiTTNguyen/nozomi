"""
compute_rdapp_from_narrow_pulse.py
===================================
Post-processing script: converts narrow-pulse RD(t) simulation data to
wide-pulse apparent radial diffusion coefficients (RDapp) using the
Gaussian Phase Approximation (GPA) integral:

    D_app = -γ²/b  ∫₀ᵀ dτ₁ ∫₀^τ₁ G(τ₁)·G(τ₂)·(τ₁-τ₂)·D(τ₁-τ₂) dτ₂

Two acquisition protocols are evaluated per substrate:

    Protocol 1 - b = 300 s/mm², TE = 78 ms
    PGSE : Δ=50 ms, δ=12 ms
    OGSE : ApodizedCosine, T=26 ms, N=1  (t_eff = 6.5 ms)

  Protocol 2 - b = 800 s/mm², TE = 40 ms
    PGSE : Δ=26 ms, δ=3 ms
    OGSE : ApodizedCosine, T=10 ms, N=1  (t_eff = 2.5 ms)

B-values use the analytical formulas:
  PGSE    : b = (gamma* G δ)² (Δ - δ/3)
  OGSE AC : b = (1/4)(gamma* G / πN)² T³ (1 - 1/(8N))
where gamma = 267.513 rad/ms/mT  (= 2π * 42.577 MHz/T for ¹H).

Usage:
    python compute_rdapp_from_narrow_pulse.py <data_root>

Each substrate subfolder that contains sim/ADCdata/*.pkl files will be
processed. Results are saved to  sim/RDapp/rdapp_result.pkl.
"""

import argparse
import os
import re
import sys
import pickle
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# Make sure the nozomi package is importable regardless of cwd
# ------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_NOZOMI_ROOT = _HERE.parent.parent.parent.parent   # nozomi/
if str(_NOZOMI_ROOT) not in sys.path:
    sys.path.insert(0, str(_NOZOMI_ROOT))

from simulation_toolkit.simulation_engine.helper.sim_util import (
    build_gradient_waveform,
    wide_pulse_gradient_integration,
)
from simulation_toolkit.simulation_engine.waveform_design import design_waveform

# ------------------------------------------------------------------
# Gradient configurations
# ------------------------------------------------------------------
# B-values use the analytical formulas (see module docstring).
# RDapp from the GPA autocorrelation ratio is independent of Gmax;
# Gmax is stored for reporting and cross-checking against scanner params.

# ---- Protocol 1: b = 300 s/mm², TE = 78 ms ------------------------
_PGSE_CONFIG = {
    "type": "PGSE",
    "big_delta": 50.0,        # ms  (Δ)
    "little_delta": 12.0,    # ms  (δ)
    "b_value_s_mm2": 300.0, # s/mm²
    "te_ms": 78.0,            # echo time / simulation window (ms)
}

_OGSE_CONFIG = {
    "type": "OGSE",
    "N_cycles": 1,
    "T_duration": 26.0,      # ms  (t_eff = T/(4N) = 6.5 ms)
    "b_value_s_mm2": 300.0, # s/mm²
    "te_ms": 78.0,
}

# ---- Protocol 2: b = 800 s/mm², TE = 40 ms -------------------------
_PGSE_CONFIG_2 = {
    "type": "PGSE",
    "big_delta": 26.0,       # ms  (Δ)
    "little_delta": 3.0,    # ms  (δ)
    "b_value_s_mm2": 800.0, # s/mm²
    "te_ms": 40.0,
}

_OGSE_CONFIG_2 = {
    "type": "OGSE",
    "N_cycles": 1,
    "T_duration": 10.0,     # ms  (t_eff = T/(4N) = 2.5 ms)
    "b_value_s_mm2": 800.0, # s/mm²
    "te_ms": 40.0,
}

# Waveform time step used when building G(t)
_WAVE_DT_MS = 0.01   # ms  (10 µs)

# ---- Human protocol hardware-consistent designs (b-value-driven) -------------
# Human PGSE is now slew-limited; human OGSE adds a trapezoidal-cosine variant
# alongside the apodized one. Both use the 80 mT/m / 100 mT/m/ms scanner preset.
_HUMAN_PRESET = "human_80_100"
_PGSE_SLEW_DESIGN = {
    "shape": "pgse-slew",
    "bvalue_s_mm2": 300.0,
    "te_ms": 78.0,
    "preset": _HUMAN_PRESET,
    "little_delta_ms": 12.0,
    "big_delta_ms": 50.0,
    "dt_ms": _WAVE_DT_MS,
}
_OGSE_TRAP_DESIGN = {
    "shape": "ogse-trapezoidal",
    "bvalue_s_mm2": 300.0,
    "te_ms": 78.0,
    "preset": _HUMAN_PRESET,
    "n_cycles": 1,
    "t_eff_ms": 6.5,
    "dt_ms": _WAVE_DT_MS,
}

# ---- Animal protocol hardware-consistent designs (15.2T Bruker Biospec) -------
# Animal PGSE is now slew-limited; animal OGSE stays apodized-cosine, both
# designed/validated against the 1000 mT/m / 5000 mT/m/ms preclinical preset.
_ANIMAL_PRESET = "animal_15p2T"
_PGSE_SLEW_DESIGN_2 = {
    "shape": "pgse-slew",
    "bvalue_s_mm2": 800.0,
    "te_ms": 40.0,
    "preset": _ANIMAL_PRESET,
    "little_delta_ms": 3.0,
    "big_delta_ms": 26.0,
    "dt_ms": _WAVE_DT_MS,
}
_OGSE_APOD_DESIGN_2 = {
    "shape": "ogse-apodized",
    "bvalue_s_mm2": 800.0,
    "te_ms": 40.0,
    "preset": _ANIMAL_PRESET,
    "n_cycles": 1,
    "t_eff_ms": 2.5,
    "dt_ms": _WAVE_DT_MS,
}

# Save one PGSE/OGSE waveform figure per substrate alongside rdapp_result.pkl.
_SAVE_WAVEFORM_PLOT = True

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_avf(filename: str) -> float:
    """Extract volume fraction from filename: pattern _avf_<float>_"""
    m = re.search(r'_avf_([\d.]+)_', filename)
    if m is None:
        raise ValueError(f"Cannot parse avf from filename: {filename}")
    return float(m.group(1))


def _load_adc_pkl(pkl_path: str):
    """
    Load an ADC pickle file.
    Returns (Dx, Dy, Dz, diff_time) as 1-D float64 arrays.
    Expected pkl shape: (N, 4) with columns [Dx, Dy, Dz, t_ms].
    """
    with open(pkl_path, 'rb') as fh:
        data = pickle.load(fh)
    data = np.array(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(
            f"Unexpected pkl shape {data.shape} in {pkl_path}. "
            "Expected (N, 4) with columns [Dx, Dy, Dz, t_ms]."
        )
    Dx, Dy, Dz, t = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    return Dx, Dy, Dz, t


def _substrate_g_ratio(substrate_dir):
    """Return (g_ratio, is_myelinated) from the substrate's data/array*.pkl,
    or (None, False) if unmyelinated / unavailable."""
    try:
        import glob as _glob
        from simulation_toolkit.utils.common_utils import load_substrate_geometry
        pkls = sorted(_glob.glob(os.path.join(substrate_dir, "data", "array*.pkl")))
        if not pkls:
            return None, False
        s = load_substrate_geometry(pkls[-1])
        return s.g_ratio, bool(s.is_myelinated)
    except Exception as exc:
        print(f"  [warn] could not read g_ratio: {exc}")
        return None, False


def _pick_file(candidates, label):
    """
    Among several candidate files for the same compartment,
    pick the one that appears last alphabetically (i.e. most recent timestamp).
    Warns if more than one file is found.
    """
    if len(candidates) > 1:
        print(f"  [warn] Multiple {label} ADC files found, using most recent: "
              f"{sorted(candidates)[-1]}")
    return sorted(candidates)[-1]


def _find_adc_pairs(adc_dir: str):
    """
    Scan an ADCdata directory and return the selected (intra_path, extra_path) tuple.
    Returns None if either compartment is missing.
    """
    intra_files, extra_files = [], []
    for fn in os.listdir(adc_dir):
        if not fn.endswith('.pkl'):
            continue
        lower = fn.lower()
        # Skip dapp result files that may have been written here by other tools
        if '_dapp' in lower:
            continue
        if '_intra_' in lower:
            intra_files.append(os.path.join(adc_dir, fn))
        elif '_extra_' in lower:
            extra_files.append(os.path.join(adc_dir, fn))

    if not intra_files:
        print(f"  [skip] No intra ADC pkl found in {adc_dir}")
        return None
    if not extra_files:
        print(f"  [skip] No extra ADC pkl found in {adc_dir}")
        return None

    return (
        _pick_file(intra_files, 'intra'),
        _pick_file(extra_files, 'extra'),
    )


def _save_protocol_waveform_plot(
    substrate_dir: str,
    human_pgse,
    human_ogse,
    human_ogse_trap,
    animal_pgse,
    animal_ogse,
) -> str:
    """
    Save a figure of the physical gradient waveforms G(t) = Gmax * shape.

    Top row = human protocol (slew PGSE, apodized OGSE, trapezoidal OGSE);
    bottom row = animal protocol (PGSE, apodized OGSE).

    Returns the saved file path, or an empty string if plotting is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  [warn] Skipping waveform plot (matplotlib unavailable): {exc}")
        return ""

    rdapp_dir = os.path.join(substrate_dir, 'sim', 'RDapp')
    os.makedirs(rdapp_dir, exist_ok=True)
    out_path = os.path.join(rdapp_dir, 'pgse_ogse_waveforms.png')

    panels = [
        (0, "Human PGSE slew (b=300, TE=78)", human_pgse),
        (1, "Human OGSE apodized (b=300, TE=78)", human_ogse),
        (2, "Human OGSE trapezoidal (b=300, TE=78)", human_ogse_trap),
        (3, "Animal PGSE (b=800, TE=40)", animal_pgse),
        (4, "Animal OGSE apodized (b=800, TE=40)", animal_ogse),
    ]
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.4 * nrows), sharex=False, sharey=False)
    axes = axes.flatten()

    for idx, title, wave_obj in panels:
        ax = axes[idx]
        gmax = float(getattr(wave_obj, 'gmax_mT_per_m', 1.0))
        ax.plot(wave_obj.t, wave_obj.wave * gmax, linewidth=1.5)
        ax.axhline(0.0, color='black', linewidth=0.7, linestyle='--', alpha=0.5)
        ax.set_title(f"{title}  |  Gmax={gmax:.1f} mT/m", fontsize=9)
        ax.set_xlabel('Time (ms)', fontsize=9)
        ax.set_ylabel('G (mT/m)', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.35)

    for ax in axes[len(panels):]:
        ax.axis('off')

    fig.suptitle(
        f"PGSE/OGSE gradient waveforms: {os.path.basename(substrate_dir)}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"  waveform plot saved at {out_path}")
    return out_path


def compute_rdapp_for_substrate(substrate_dir: str) -> dict:
    """
    Core computation for one substrate folder.

    Parameters
    ----------
    substrate_dir : str
        Absolute path to the substrate folder.

    Returns
    -------
    dict  with RDapp values for both acquisition protocols.
    """
    adc_dir = os.path.join(substrate_dir, 'sim', 'ADCdata')
    if not os.path.isdir(adc_dir):
        raise FileNotFoundError(f"ADCdata folder not found: {adc_dir}")

    pair = _find_adc_pairs(adc_dir)
    if pair is None:
        raise FileNotFoundError(f"Missing intra or extra ADC file in {adc_dir}")

    intra_path, extra_path = pair
    print(f"  intra  : {os.path.basename(intra_path)}")
    print(f"  extra  : {os.path.basename(extra_path)}")

    # ---- Extract VF (OUTER axon volume fraction) ----
    VF = _extract_avf(os.path.basename(intra_path))
    print(f"  VF     : {VF:.4f}")

    # ---- Myelin-aware water fractions (for the voxel-total signal) ----
    # AVF is the OUTER volume fraction; intra water fills the INNER (g^2*AVF),
    # extra water is (1-AVF); the water-free myelin annulus is excluded, and the
    # total is normalized by the water fraction f_in+f_ex.
    g_ratio, is_myel = _substrate_g_ratio(substrate_dir)
    if is_myel and g_ratio is not None:
        f_in = (float(g_ratio) ** 2) * VF
    else:
        f_in = VF
    f_ex = 1.0 - VF
    f_water = f_in + f_ex
    print(f"  fractions: f_in={f_in:.4f} f_ex={f_ex:.4f} g_ratio={g_ratio}")

    # ---- Load data ----
    Dxi, Dyi, Dzi, t_intra = _load_adc_pkl(intra_path)
    Dxe, Dye, Dze, t_extra = _load_adc_pkl(extra_path)

    if not np.allclose(t_intra, t_extra, rtol=1e-4):
        print("  [info] Time axes differ — interpolating extra onto intra grid")
        Dxe = np.interp(t_intra, t_extra, Dxe)
        Dye = np.interp(t_intra, t_extra, Dye)
        Dze = np.interp(t_intra, t_extra, Dze)
    diff_time = t_intra

    # ---- Radial diffusion ----
    RD_intra = (Dxi + Dyi) / 2.0
    RD_extra = (Dxe + Dye) / 2.0
    RD_total = (f_in * RD_intra + f_ex * RD_extra) / f_water

    # ------------------------------------------------------------------
    # Helper: build waveform once, then run GPA per compartment
    # ------------------------------------------------------------------
    def _build_waveform(cfg, label):
        te = float(cfg["te_ms"])
        gwave = build_gradient_waveform(cfg, te, _WAVE_DT_MS)
        print(f"  {label:<12s}  b={gwave.bvalue_s_mm2:6.1f} s/mm²"
              f"  ({gwave.bvalue_ms_um2:.4e} ms/µm²)"
              f"  Gmax={gwave.gmax_mT_per_m:7.2f} mT/m  TE={te:.0f}ms")

        return gwave

    def _integrate_component(gwave, diffusion_coeff):
        return wide_pulse_gradient_integration(
            diffusion_time_ms=diff_time,
            diffusion_coeff=diffusion_coeff,
            gradient_waveform=gwave.wave,
            gradient_dt_ms=gwave.dt,
        )

    def _build_designed(design_kwargs, label):
        # Hardware-consistent human waveform via the b-value-driven design pipeline.
        res = design_waveform(**design_kwargs)
        gwave = res.waveform
        if gwave is None:
            raise RuntimeError(f"{label} design infeasible: {res.messages}")
        if not res.feasible:
            print(f"  [warn] {label} exceeds hardware limit: {res.messages}")
        gwave.bvalue_s_mm2 = res.bvalue_s_mm2_actual
        gwave.bvalue_ms_um2 = res.bvalue_s_mm2_actual * 1e-3
        gwave.gmax_mT_per_m = res.gmax_mT_per_m
        print(f"  {label:<12s}  b={gwave.bvalue_s_mm2:6.1f} s/mm²"
              f"  Gmax={gwave.gmax_mT_per_m:7.2f} mT/m  TE={float(design_kwargs['te_ms']):.0f}ms")
        return gwave

    def _run_with_gwave(gwave, label):
        rdapp_intra = _integrate_component(gwave, RD_intra)
        rdapp_extra = _integrate_component(gwave, RD_extra)
        rdapp_total = _integrate_component(gwave, RD_total)

        print(f"  {label:<12s}  RDapp_intra = {rdapp_intra:.6f} µm²/ms")
        print(f"  {label:<12s}  RDapp_extra = {rdapp_extra:.6f} µm²/ms")
        print(f"  {label:<12s}  RDapp_total = {rdapp_total:.6f} µm²/ms")

        return {
            "intra": rdapp_intra,
            "extra": rdapp_extra,
            "total": rdapp_total,
            "gwave": gwave,
        }

    def _run_protocol(cfg, label):
        return _run_with_gwave(_build_waveform(cfg, label), label)

    def _run_designed(design_kwargs, label):
        return _run_with_gwave(_build_designed(design_kwargs, label), label)

    # ---- Protocol 1: human, b=300 s/mm², TE=78 ms ----
    # Primary human PGSE is slew-limited (human_80_100). The ideal PGSE and
    # apodized OGSE are retained for reference; a trapezoidal OGSE is added.
    print("  --- Protocol 1: b=300 s/mm², TE=78 ms ---")
    pgse_ideal_res = _run_protocol(_PGSE_CONFIG, "PGSE_ideal")
    pgse_res = _run_designed(_PGSE_SLEW_DESIGN, "PGSE_slew")
    ogse_res = _run_protocol(_OGSE_CONFIG, "OGSE")
    ogse_trap_res = _run_designed(_OGSE_TRAP_DESIGN, "OGSE_trap")

    RDapp_PGSE = pgse_res["total"]
    RDapp_PGSE_intra = pgse_res["intra"]
    RDapp_PGSE_extra = pgse_res["extra"]

    RDapp_PGSE_ideal = pgse_ideal_res["total"]
    RDapp_PGSE_ideal_intra = pgse_ideal_res["intra"]
    RDapp_PGSE_ideal_extra = pgse_ideal_res["extra"]

    RDapp_OGSE = ogse_res["total"]
    RDapp_OGSE_intra = ogse_res["intra"]
    RDapp_OGSE_extra = ogse_res["extra"]

    RDapp_OGSE_trap = ogse_trap_res["total"]
    RDapp_OGSE_trap_intra = ogse_trap_res["intra"]
    RDapp_OGSE_trap_extra = ogse_trap_res["extra"]

    # Apparent radial-diffusion contrast against the slew-limited PGSE baseline.
    DeltaRDapp = RDapp_OGSE - RDapp_PGSE
    DeltaRDapp_intra = RDapp_OGSE_intra - RDapp_PGSE_intra
    DeltaRDapp_extra = RDapp_OGSE_extra - RDapp_PGSE_extra

    DeltaRDapp_trap = RDapp_OGSE_trap - RDapp_PGSE
    DeltaRDapp_trap_intra = RDapp_OGSE_trap_intra - RDapp_PGSE_intra
    DeltaRDapp_trap_extra = RDapp_OGSE_trap_extra - RDapp_PGSE_extra
    print(f"  {'':12s}  ΔRDapp_total(apod) = {DeltaRDapp:.6f} µm²/ms")
    print(f"  {'':12s}  ΔRDapp_total(trap) = {DeltaRDapp_trap:.6f} µm²/ms")

    # ---- Protocol 2: animal, b=800 s/mm², TE=40 ms ----
    # Primary animal PGSE is slew-limited (animal_15p2T); animal OGSE stays
    # apodized-cosine (validated against the same preset). Ideal PGSE retained.
    print("  --- Protocol 2: b=800 s/mm², TE=40 ms ---")
    pgse_ideal_res_2 = _run_protocol(_PGSE_CONFIG_2, "PGSE_2_ideal")
    pgse_res_2 = _run_designed(_PGSE_SLEW_DESIGN_2, "PGSE_2_slew")
    ogse_res_2 = _run_designed(_OGSE_APOD_DESIGN_2, "OGSE_2_apod")

    RDapp_PGSE_2 = pgse_res_2["total"]
    RDapp_OGSE_2 = ogse_res_2["total"]
    RDapp_PGSE_intra_2 = pgse_res_2["intra"]
    RDapp_OGSE_intra_2 = ogse_res_2["intra"]
    RDapp_PGSE_extra_2 = pgse_res_2["extra"]
    RDapp_OGSE_extra_2 = ogse_res_2["extra"]

    RDapp_PGSE_ideal_2 = pgse_ideal_res_2["total"]
    RDapp_PGSE_ideal_intra_2 = pgse_ideal_res_2["intra"]
    RDapp_PGSE_ideal_extra_2 = pgse_ideal_res_2["extra"]

    # Animal contrast against the slew-limited PGSE baseline.
    DeltaRDapp_2 = RDapp_OGSE_2 - RDapp_PGSE_2
    DeltaRDapp_intra_2 = RDapp_OGSE_intra_2 - RDapp_PGSE_intra_2
    DeltaRDapp_extra_2 = RDapp_OGSE_extra_2 - RDapp_PGSE_extra_2
    print(f"  {'':12s}  ΔRDapp_total = {DeltaRDapp_2:.6f} µm²/ms")
    print(f"  {'':12s}  ΔRDapp_intra = {DeltaRDapp_intra_2:.6f} µm²/ms")
    print(f"  {'':12s}  ΔRDapp_extra = {DeltaRDapp_extra_2:.6f} µm²/ms")

    gw_pgse = pgse_res["gwave"]
    gw_pgse_ideal = pgse_ideal_res["gwave"]
    gw_ogse = ogse_res["gwave"]
    gw_ogse_trap = ogse_trap_res["gwave"]
    gw_pgse2 = pgse_res_2["gwave"]
    gw_pgse2_ideal = pgse_ideal_res_2["gwave"]
    gw_ogse2 = ogse_res_2["gwave"]
    waveform_plot_path = ""
    if _SAVE_WAVEFORM_PLOT:
        waveform_plot_path = _save_protocol_waveform_plot(
            substrate_dir=substrate_dir,
            human_pgse=gw_pgse,
            human_ogse=gw_ogse,
            human_ogse_trap=gw_ogse_trap,
            animal_pgse=gw_pgse2,
            animal_ogse=gw_ogse2,
        )

    return {
        'substrate_dir': substrate_dir,
        'intra_file': intra_path,
        'extra_file': extra_path,
        'VF': VF,
        'diff_time_ms': diff_time,
        'RD_intra': RD_intra,
        'RD_extra': RD_extra,
        'RD_total': RD_total,
        # --- Protocol 1 (human, b=300, TE=78) ---
        # Primary PGSE is slew-limited; ideal PGSE + trapezoidal OGSE retained.
        'PGSE_config':          _PGSE_SLEW_DESIGN,
        'PGSE_bvalue_s_mm2':    gw_pgse.bvalue_s_mm2,
        'PGSE_bvalue_ms_um2':   gw_pgse.bvalue_ms_um2,
        'PGSE_gmax_mT_per_m':   gw_pgse.gmax_mT_per_m,
        'PGSE_ideal_config':        _PGSE_CONFIG,
        'PGSE_ideal_bvalue_s_mm2':  gw_pgse_ideal.bvalue_s_mm2,
        'PGSE_ideal_gmax_mT_per_m': gw_pgse_ideal.gmax_mT_per_m,
        'OGSE_config':          _OGSE_CONFIG,
        'OGSE_bvalue_s_mm2':    gw_ogse.bvalue_s_mm2,
        'OGSE_bvalue_ms_um2':   gw_ogse.bvalue_ms_um2,
        'OGSE_gmax_mT_per_m':   gw_ogse.gmax_mT_per_m,
        'OGSE_trap_config':        _OGSE_TRAP_DESIGN,
        'OGSE_trap_bvalue_s_mm2':  gw_ogse_trap.bvalue_s_mm2,
        'OGSE_trap_gmax_mT_per_m': gw_ogse_trap.gmax_mT_per_m,
        'RDapp_PGSE':           RDapp_PGSE,
        'RDapp_OGSE':           RDapp_OGSE,
        'RDapp_PGSE_intra':     RDapp_PGSE_intra,
        'RDapp_OGSE_intra':     RDapp_OGSE_intra,
        'RDapp_PGSE_extra':     RDapp_PGSE_extra,
        'RDapp_OGSE_extra':     RDapp_OGSE_extra,
        'RDapp_PGSE_ideal':        RDapp_PGSE_ideal,
        'RDapp_PGSE_ideal_intra':  RDapp_PGSE_ideal_intra,
        'RDapp_PGSE_ideal_extra':  RDapp_PGSE_ideal_extra,
        'RDapp_OGSE_trap':         RDapp_OGSE_trap,
        'RDapp_OGSE_trap_intra':   RDapp_OGSE_trap_intra,
        'RDapp_OGSE_trap_extra':   RDapp_OGSE_trap_extra,
        'DeltaRDapp':           DeltaRDapp,
        'DeltaRDapp_intra':     DeltaRDapp_intra,
        'DeltaRDapp_extra':     DeltaRDapp_extra,
        'DeltaRDapp_trap':         DeltaRDapp_trap,
        'DeltaRDapp_trap_intra':   DeltaRDapp_trap_intra,
        'DeltaRDapp_trap_extra':   DeltaRDapp_trap_extra,
        # --- Protocol 2 (animal, b=800, TE=40) ---
        # Primary PGSE is slew-limited; ideal PGSE retained. OGSE is apodized.
        'PGSE_config_2':        _PGSE_SLEW_DESIGN_2,
        'PGSE_bvalue_s_mm2_2':  gw_pgse2.bvalue_s_mm2,
        'PGSE_bvalue_ms_um2_2': gw_pgse2.bvalue_ms_um2,
        'PGSE_gmax_mT_per_m_2': gw_pgse2.gmax_mT_per_m,
        'PGSE_ideal_config_2':        _PGSE_CONFIG_2,
        'PGSE_ideal_bvalue_s_mm2_2':  gw_pgse2_ideal.bvalue_s_mm2,
        'PGSE_ideal_gmax_mT_per_m_2': gw_pgse2_ideal.gmax_mT_per_m,
        'OGSE_config_2':        _OGSE_APOD_DESIGN_2,
        'OGSE_bvalue_s_mm2_2':  gw_ogse2.bvalue_s_mm2,
        'OGSE_bvalue_ms_um2_2': gw_ogse2.bvalue_ms_um2,
        'OGSE_gmax_mT_per_m_2': gw_ogse2.gmax_mT_per_m,
        'RDapp_PGSE_2':         RDapp_PGSE_2,
        'RDapp_OGSE_2':         RDapp_OGSE_2,
        'RDapp_PGSE_intra_2':   RDapp_PGSE_intra_2,
        'RDapp_OGSE_intra_2':   RDapp_OGSE_intra_2,
        'RDapp_PGSE_extra_2':   RDapp_PGSE_extra_2,
        'RDapp_OGSE_extra_2':   RDapp_OGSE_extra_2,
        'RDapp_PGSE_ideal_2':        RDapp_PGSE_ideal_2,
        'RDapp_PGSE_ideal_intra_2':  RDapp_PGSE_ideal_intra_2,
        'RDapp_PGSE_ideal_extra_2':  RDapp_PGSE_ideal_extra_2,
        'DeltaRDapp_2':         DeltaRDapp_2,
        'DeltaRDapp_intra_2':   DeltaRDapp_intra_2,
        'DeltaRDapp_extra_2':   DeltaRDapp_extra_2,
        'waveform_plot_path':   waveform_plot_path,
    }


def save_rdapp_result(result: dict, substrate_dir: str) -> str:
    """
    Save result dict to  <substrate_dir>/sim/RDapp/rdapp_result.pkl
    Returns the saved file path.
    """
    rdapp_dir = os.path.join(substrate_dir, 'sim', 'RDapp')
    os.makedirs(rdapp_dir, exist_ok=True)
    out_path = os.path.join(rdapp_dir, 'rdapp_result.pkl')
    with open(out_path, 'wb') as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def process_data_root(data_root: str):
    """
    Walk data_root and process every substrate folder that contains
    sim/ADCdata/*.pkl files.

    Expected structure:
        data_root/
            <experiment_group>/
                <substrate_id>/
                    sim/ADCdata/*.pkl
    """
    data_root = os.path.abspath(data_root)
    if not os.path.isdir(data_root):
        raise NotADirectoryError(f"Not a directory: {data_root}")

    print(f"\n{'='*70}")
    print(f"Processing data root: {data_root}")
    print(f"{'='*70}\n")

    processed, skipped, failed = 0, 0, 0

    for exp_group in sorted(os.listdir(data_root)):
        exp_group_path = os.path.join(data_root, exp_group)
        if not os.path.isdir(exp_group_path):
            continue

        for substrate_id in sorted(os.listdir(exp_group_path)):
            substrate_path = os.path.join(exp_group_path, substrate_id)
            adc_dir = os.path.join(substrate_path, 'sim', 'ADCdata')

            if not os.path.isdir(adc_dir):
                continue  # Not a substrate folder we care about

            print(f"[{exp_group}]  {substrate_id}")

            try:
                result = compute_rdapp_for_substrate(substrate_path)
                out_path = save_rdapp_result(result, substrate_path)
                print(f"  saved at {out_path}\n")
                processed += 1
            except FileNotFoundError as e:
                print(f"  [skip] {e}\n")
                skipped += 1
            except Exception as e:
                print(f"  [ERROR] {e}\n")
                failed += 1

    print(f"{'='*70}")
    print(f"Done.  processed={processed}  skipped={skipped}  failed={failed}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute wide-pulse apparent RD from pre-simulated narrow-pulse "
            "ADCdata across an experiment data folder."
        )
    )
    parser.add_argument(
        'data_root',
        help=(
            "Path to root folder containing experiment subfolders, "
            "e.g. experiment/visualization/data"
        ),
    )
    args = parser.parse_args()
    process_data_root(args.data_root)


if __name__ == '__main__':
    main()
