"""
compute_rdapp_from_narrow_pulse.py
===================================
Post-processing script: converts narrow-pulse RD(t) simulation data to
wide-pulse apparent radial diffusion coefficients (RDapp) using the
Gaussian Phase Approximation (GPA) integral:

    D_app = -γ²/b  ∫₀ᵀ dτ₁ ∫₀^τ₁ G(τ₁)·G(τ₂)·(τ₁-τ₂)·D(τ₁-τ₂) dτ₂

Two acquisition protocols are evaluated per substrate:

  Protocol 1 - b = 300 s/mm², TE = 100 ms
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
    bvalue_pgse_ms_um2,
    bvalue_ogse_apodized_cosine_ms_um2,
)

# ------------------------------------------------------------------
# Gradient configurations
# ------------------------------------------------------------------
# B-values use the analytical formulas (see module docstring).
# RDapp from the GPA autocorrelation ratio is independent of Gmax;
# Gmax is stored for reporting and cross-checking against scanner params.

# ---- Protocol 1: b = 300 s/mm², TE = 100 ms -----------------------
_PGSE_CONFIG = {
    "type": "PGSE",
    "big_delta": 50.0,        # ms  (Δ)
    "little_delta": 12.0,    # ms  (δ)
    "b_value_s_mm2": 300.0, # s/mm²
    "te_ms": 100.0,           # echo time / simulation window (ms)
}

_OGSE_CONFIG = {
    "type": "OGSE",
    "N_cycles": 1,
    "T_duration": 26.0,      # ms  (t_eff = T/(4N) = 6.5 ms)
    "b_value_s_mm2": 300.0, # s/mm²
    "te_ms": 100.0,
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

    # ---- Extract VF ----
    VF = _extract_avf(os.path.basename(intra_path))
    print(f"  VF     : {VF:.4f}")

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
    RD_total = VF * RD_intra + (1.0 - VF) * RD_extra

    # ------------------------------------------------------------------
    # Helper: build waveform, print parameters, run GPA integration
    # ------------------------------------------------------------------
    def _run_gpa(cfg, label):
        te = float(cfg["te_ms"])
        gwave = build_gradient_waveform(cfg, te, _WAVE_DT_MS)
        print(f"  {label:<12s}  b={gwave.bvalue_s_mm2:6.1f} s/mm²"
              f"  ({gwave.bvalue_ms_um2:.4e} ms/µm²)"
              f"  Gmax={gwave.gmax_mT_per_m:7.2f} mT/m  TE={te:.0f}ms")
        rdapp = wide_pulse_gradient_integration(
            diffusion_time_ms=diff_time,
            diffusion_coeff=RD_total,
            gradient_waveform=gwave.wave,
            gradient_dt_ms=gwave.dt,
        )
        print(f"  {label:<12s}  RDapp = {rdapp:.6f} µm²/ms")
        return rdapp, gwave

    # ---- Protocol 1: b=300 s/mm², TE=100 ms ----
    print("  --- Protocol 1: b=300 s/mm², TE=100 ms ---")
    RDapp_PGSE,  gw_pgse  = _run_gpa(_PGSE_CONFIG,   "PGSE")
    RDapp_OGSE,  gw_ogse  = _run_gpa(_OGSE_CONFIG,   "OGSE")
    DeltaRDapp             = RDapp_OGSE - RDapp_PGSE
    print(f"  {'':12s}  ΔRDapp = {DeltaRDapp:.6f} µm²/ms")

    # ---- Protocol 2: b=800 s/mm², TE=40 ms ----
    print("  --- Protocol 2: b=800 s/mm², TE=40 ms ---")
    RDapp_PGSE_2, gw_pgse2 = _run_gpa(_PGSE_CONFIG_2, "PGSE_2")
    RDapp_OGSE_2, gw_ogse2 = _run_gpa(_OGSE_CONFIG_2, "OGSE_2")
    DeltaRDapp_2            = RDapp_OGSE_2 - RDapp_PGSE_2
    print(f"  {'':12s}  ΔRDapp = {DeltaRDapp_2:.6f} µm²/ms")

    return {
        'substrate_dir': substrate_dir,
        'intra_file': intra_path,
        'extra_file': extra_path,
        'VF': VF,
        'diff_time_ms': diff_time,
        'RD_intra': RD_intra,
        'RD_extra': RD_extra,
        'RD_total': RD_total,
        # --- Protocol 1 ---
        'PGSE_config':          _PGSE_CONFIG,
        'PGSE_bvalue_s_mm2':    gw_pgse.bvalue_s_mm2,
        'PGSE_bvalue_ms_um2':   gw_pgse.bvalue_ms_um2,
        'PGSE_gmax_mT_per_m':   gw_pgse.gmax_mT_per_m,
        'OGSE_config':          _OGSE_CONFIG,
        'OGSE_bvalue_s_mm2':    gw_ogse.bvalue_s_mm2,
        'OGSE_bvalue_ms_um2':   gw_ogse.bvalue_ms_um2,
        'OGSE_gmax_mT_per_m':   gw_ogse.gmax_mT_per_m,
        'RDapp_PGSE':           RDapp_PGSE,
        'RDapp_OGSE':           RDapp_OGSE,
        'DeltaRDapp':           DeltaRDapp,
        # --- Protocol 2 ---
        'PGSE_config_2':        _PGSE_CONFIG_2,
        'PGSE_bvalue_s_mm2_2':  gw_pgse2.bvalue_s_mm2,
        'PGSE_bvalue_ms_um2_2': gw_pgse2.bvalue_ms_um2,
        'PGSE_gmax_mT_per_m_2': gw_pgse2.gmax_mT_per_m,
        'OGSE_config_2':        _OGSE_CONFIG_2,
        'OGSE_bvalue_s_mm2_2':  gw_ogse2.bvalue_s_mm2,
        'OGSE_bvalue_ms_um2_2': gw_ogse2.bvalue_ms_um2,
        'OGSE_gmax_mT_per_m_2': gw_ogse2.gmax_mT_per_m,
        'RDapp_PGSE_2':         RDapp_PGSE_2,
        'RDapp_OGSE_2':         RDapp_OGSE_2,
        'DeltaRDapp_2':         DeltaRDapp_2,
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
