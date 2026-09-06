"""
Render the achieved diffusion gradient waveforms for the human and animal
protocols using the b-value-driven design pipeline, and save the plots under
``tests/validation/wide_pulse/test_figures``.

Protocol timings (from compute_rdapp_from_narrow_pulse.py):
    Human  : b = 300 s/mm^2, TE = 78 ms; PGSE Delta=50, delta=12; OGSE N=1, t_eff=6.5
    Animal : b = 800 s/mm^2, TE = 40 ms; PGSE Delta=26, delta= 3; OGSE N=1, t_eff=2.5

Design mapping:
    Human  -> slew-limited PGSE + trapezoidal-cosine OGSE (ideal PGSE for reference)
    Animal -> apodized-cosine OGSE
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from simulation_toolkit.simulation_engine.waveform_design import (
    design_waveform,
    plot_waveform,
)

_OUT_ROOT = os.path.join(os.path.dirname(__file__), "test_figures")

# (label, kwargs) for each achieved waveform.
_PROTOCOL_DESIGNS = [
    (
        "human_ideal_PGSE",
        dict(shape="pgse", bvalue_s_mm2=300, te_ms=78, preset="human_80_100",
             little_delta_ms=12, big_delta_ms=50, dt_ms=0.005),
    ),
    (
        "human_slew_PGSE",
        dict(shape="pgse-slew", bvalue_s_mm2=300, te_ms=78, preset="human_80_100",
             little_delta_ms=12, big_delta_ms=50, dt_ms=0.005),
    ),
    (
        "human_trapezoidal_OGSE",
        dict(shape="ogse-trapezoidal", bvalue_s_mm2=300, te_ms=78, preset="human_80_100",
             n_cycles=1, t_eff_ms=6.5, dt_ms=0.005),
    ),
    (
        "animal_apodized_OGSE",
        dict(shape="ogse-apodized", bvalue_s_mm2=800, te_ms=40, preset="animal_placeholder",
             n_cycles=1, t_eff_ms=2.5, dt_ms=0.005),
    ),
]


def _panel_title(label, result):
    return (
        f"{label}\n"
        f"b={result.bvalue_s_mm2_actual:.0f} s/mm²  "
        f"Gmax={result.gmax_mT_per_m:.1f} mT/m  TE={result.te_ms:.0f} ms"
    )


def test_render_protocol_waveforms():
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(_OUT_ROOT, f"protocol_waveforms_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for label, kwargs in _PROTOCOL_DESIGNS:
        result = design_waveform(**kwargs)
        assert result.waveform is not None, f"{label} produced no waveform: {result.messages}"
        results.append((label, result))

        # Individual annotated plot.
        plot_waveform(
            result,
            os.path.join(out_dir, f"{label}.png"),
            title=_panel_title(label, result).replace("\n", "  |  "),
        )

    # Combined 2x2 overview of the achieved physical waveforms.
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (label, result) in zip(axes.flatten(), results):
        w = result.waveform
        ax.plot(w.t, w.wave * result.gmax_mT_per_m, linewidth=1.5)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(_panel_title(label, result), fontsize=10)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("G (mT/m)")
        ax.grid(True, linestyle="--", alpha=0.35)
    fig.suptitle("Achieved protocol gradient waveforms (Human & Animal)", fontsize=13)
    fig.tight_layout()
    combined_path = os.path.join(out_dir, "protocol_waveforms_overview.png")
    fig.savefig(combined_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved protocol waveform figures to: {out_dir}")
    for label, result in results:
        flag = "OK" if result.feasible else "INFEASIBLE"
        print(f"  [{flag}] {label}: Gmax={result.gmax_mT_per_m:.2f} mT/m, "
              f"b={result.bvalue_s_mm2_actual:.1f} s/mm², "
              f"trise={None if result.trise_ms is None else round(result.trise_ms, 4)} ms")

    assert os.path.isfile(combined_path)


if __name__ == "__main__":
    test_render_protocol_waveforms()
