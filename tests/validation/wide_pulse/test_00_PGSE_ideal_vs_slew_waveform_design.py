"""
Validation: ideal vs slew-limited PGSE waveform design.

Non-GPU checks that both the ideal rectangular PGSE (PGDiffWaveform) and the
slew-limited trapezoidal PGSE (TrapezoidalPGSEWaveform) reach the target
b-value, that the trapezoidal lobe reduces to the ideal one when trise -> 0, and
that hardware-preset feasibility is enforced.
"""
import os
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from simulation_toolkit.simulation_engine.waveforms import (
    PGDiffWaveform,
    TrapezoidalPGSEWaveform,
)
from simulation_toolkit.simulation_engine.waveform_design import (
    design_waveform,
    HARDWARE_PRESETS,
)

_S_MM2_TO_MS_UM2 = 1e-3


def _b_s_mm2(waveform, gmax):
    return waveform.calculate_bvalue_from_wave() * gmax ** 2 / _S_MM2_TO_MS_UM2


def test_pgse_ideal_reaches_target_bvalue():
    r = design_waveform(
        "pgse", bvalue_s_mm2=300, te_ms=78, preset="human_80_100",
        little_delta_ms=12, big_delta_ms=50, dt_ms=0.005,
    )
    assert r.feasible
    assert r.trise_ms is None
    assert abs(r.bvalue_s_mm2_actual - 300.0) < 1.0


def test_pgse_slew_reaches_target_bvalue():
    r = design_waveform(
        "pgse-slew", bvalue_s_mm2=300, te_ms=78, preset="human_80_100",
        little_delta_ms=12, big_delta_ms=50, dt_ms=0.005,
    )
    assert r.feasible
    assert r.trise_ms is not None
    # trise = Gmax / slew
    slew = HARDWARE_PRESETS["human_80_100"].slew_mT_per_m_per_ms
    assert abs(r.trise_ms - r.gmax_mT_per_m / slew) < 1e-6
    assert abs(r.bvalue_s_mm2_actual - 300.0) < 1.0


def test_slew_pgse_needs_more_gmax_than_ideal():
    common = dict(bvalue_s_mm2=300, te_ms=78, preset="human_80_100",
                  little_delta_ms=12, big_delta_ms=50, dt_ms=0.005)
    ideal = design_waveform("pgse", **common)
    slew = design_waveform("pgse-slew", **common)
    # Finite ramps lower encoding efficiency, so more Gmax is required.
    assert slew.gmax_mT_per_m > ideal.gmax_mT_per_m


def test_trapezoid_pgse_reduces_to_ideal_when_trise_zero():
    te, delta, Delta, dt = 78.0, 12.0, 50.0, 0.005
    ideal = PGDiffWaveform(big_delta=Delta, little_delta=delta, te=te, time_step=dt)
    trap0 = TrapezoidalPGSEWaveform(
        big_delta=Delta, little_delta=delta, te=te, trise=0.0, time_step=dt,
    )
    b_ideal = ideal.calculate_bvalue_from_wave()
    b_trap0 = trap0.calculate_bvalue_from_wave()
    # Only edge-sample discretization differs (open vs half-open lobe masks);
    # the gap shrinks as dt -> 0, so a loose relative tolerance is appropriate.
    assert np.isclose(b_ideal, b_trap0, rtol=2e-3, atol=0.0)


def test_hardware_feasibility_is_enforced():
    # 40 mT/m preset cannot reach b=300 with this short-t_eff trapezoidal OGSE.
    r = design_waveform(
        "ogse-trapezoidal", bvalue_s_mm2=300, te_ms=78, preset="human_40_200",
        n_cycles=1, t_eff_ms=6.5, dt_ms=0.01,
    )
    assert r.gmax_mT_per_m > HARDWARE_PRESETS["human_40_200"].gmax_mT_per_m
    assert r.feasible is False


if __name__ == "__main__":
    test_pgse_ideal_reaches_target_bvalue()
    test_pgse_slew_reaches_target_bvalue()
    test_slew_pgse_needs_more_gmax_than_ideal()
    test_trapezoid_pgse_reduces_to_ideal_when_trise_zero()
    test_hardware_feasibility_is_enforced()
    print("All PGSE ideal-vs-slew waveform-design checks passed.")
