"""
plot_deltardapp_vs_diameter_aim2.py
===================================
Aim 2 variant of ``plot_deltardapp_vs_diameter.py``.

The Aim 2 data layout (``experiment/aim2-prep/data``) has three differences
from the original Aim 1 layout (``experiment/visualization/data``):

1.  One extra nesting level (a batch folder), so substrate replicates live at
        ``data_root/<batch>/<parameter_group>/<substrate_replicate>/``
    instead of
        ``data_root/<experiment_group>/<substrate>/``.

2.  The parameter-group folder names follow the pattern
        ``bead<bead>_d<diam>_OD<od>_initVF<vf>_<n>axons``
    rather than
        ``experiment_d<diam>_..._reproducibility_OD<od>``.

3.  The substrate diameter PNG is named
        ``Diameter_distribution_mean<mean>_std<std>_<date>.png``
    rather than
        ``Optimized_diameter_distribution_mean<mean>_std<std>_<date>.png``.

The original script remains untouched so the Aim 1 workflow keeps working.
All plotting functions from ``plot_deltardapp_vs_diameter`` are reused — only
``collect_deltardapp`` (the data-collection walker) and ``main`` (the CLI) are
re-implemented here for the Aim 2 layout.

Usage:
    python plot_deltardapp_vs_diameter_aim2.py experiment/aim2-prep/data \\
        --output experiment/aim2-prep/plots/deltardapp_vs_diameter.png
"""

import argparse
import os
import re
import sys
import pickle
from collections import defaultdict
from pathlib import Path

# ------------------------------------------------------------------
# Make sure the nozomi package is importable regardless of cwd
# ------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_NOZOMI_ROOT = _HERE.parent.parent.parent.parent   # nozomi/
if str(_NOZOMI_ROOT) not in sys.path:
    sys.path.insert(0, str(_NOZOMI_ROOT))

from simulation_toolkit.simulation_engine.helper import plot_deltardapp_vs_diameter as _base
from simulation_toolkit.simulation_engine.helper.plot_deltardapp_vs_diameter import (
    DIAMETER_METRICS,
    plot_deltardapp,
    plot_deltardapp_per_od,
    plot_individual_pgse_ogse_pairs_by_od,
    plot_individual_pgse_ogse_pairs_all_od_combined,
    plot_pgse_ogse_pairs_combined_all,
    plot_deltardapp_combined_all,
    plot_rdapp_components_stacked,
    plot_rdapp_component,
    plot_rdapp_total_by_od,
    plot_rdapp_total_by_od_combined,
    _extract_substrate_mean_diameter as _base_extract_substrate_mean_diameter,
    _extract_effective_diameter_metrics,
)

# ------------------------------------------------------------------
# Aim 2 specific folder-name parsers
# ------------------------------------------------------------------
# Parameter-group folder: e.g. ``bead0.83_d1.68_OD10_initVF0.29_500axons``
_AIM2_PARAM_GROUP_RE = re.compile(
    r'^bead(?P<bead>[\d.]+)_d(?P<diam>[\d.]+)_OD(?P<od>\d+)_initVF[\d.]+_\d+axons$'
)

# Substrate-level diameter histogram PNG (no ``Optimized_`` prefix).
_AIM2_SUBSTRATE_MEAN_RE = re.compile(
    r'Diameter_distribution_mean(?P<mean>[\d.]+)_std[\d.]+_.*\.png$'
)

# Explicit x-axis range (µm) for the d_eff_p3q2 per-OD plot, by bead label.
# Bead groups not listed fall back to the data-derived auto range.
_DEFF_XLIM_BY_BEAD = {
    'bead_0.3': (2, 7),
    'bead_0.5': (2, 8),
}


def _parse_param_group_folder(name: str):
    """Return (diameter_float, od_int) parsed from an Aim 2 param-group folder, else None."""
    m = _AIM2_PARAM_GROUP_RE.match(name)
    if m is None:
        return None
    return float(m.group('diam')), int(m.group('od'))


def _bead_label_for_batch(batch_path: str):
    """
    Return an output-subfolder label of the form ``bead_<value>`` parsed from the
    first matching parameter-group folder inside ``batch_path`` (e.g. ``bead_0.3``),
    or None if no parameter-group folder matches the Aim 2 naming pattern.
    """
    if not os.path.isdir(batch_path):
        return None
    for param_group in sorted(os.listdir(batch_path)):
        m = _AIM2_PARAM_GROUP_RE.match(param_group)
        if m is not None:
            return f"bead_{m.group('bead')}"
    return None


def _extract_substrate_mean_diameter_aim2(substrate_path: str):
    """
    Return the mean diameter (µm) parsed from the substrate-stats PNG filename.

    First tries the Aim 2 pattern (``Diameter_distribution_mean...``);
    falls back to the original Aim 1 pattern (``Optimized_diameter_distribution_mean...``)
    in case both file formats coexist.
    """
    stats_dir = os.path.join(substrate_path, 'figs', 'substrate_stats')
    if not os.path.isdir(stats_dir):
        return None

    candidates = []
    for fn in os.listdir(stats_dir):
        m = _AIM2_SUBSTRATE_MEAN_RE.match(fn)
        if m is not None:
            candidates.append((fn, float(m.group('mean'))))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    # Fall back to the Aim 1 file naming, just in case
    return _base_extract_substrate_mean_diameter(substrate_path)


# ------------------------------------------------------------------
# Data collection for the Aim 2 three-level layout
# ------------------------------------------------------------------

def collect_deltardapp_aim2(data_root: str, only_batch: str = None,
                            human_ogse: str = "apodized") -> dict:
    """
    Walk an Aim 2 ``data`` folder (three nesting levels) and collect
    DeltaRDapp values for both scanner scenarios.

    Parameters
    ----------
    data_root : str
        Root data folder containing batch subfolders.
    only_batch : str, optional
        If given, only the batch subfolder with this exact name is processed
        (used by the ``--per-batch`` workflow). When None, all batches are merged.
    human_ogse : {"apodized", "trapezoidal"}
        Which human OGSE variant to read. ``"trapezoidal"`` uses the slew-limited
        trapezoidal-cosine keys (``RDapp_OGSE_trap*`` / ``DeltaRDapp_trap``);
        ``"apodized"`` (default) uses the apodized keys. Human PGSE is the
        slew-limited ``RDapp_PGSE`` in both cases; animal keys are unchanged.

    Returns a dict with the same shape as the Aim 1 ``collect_deltardapp``,
    so the existing plotting functions can be reused unchanged.
    """
    if human_ogse not in ("apodized", "trapezoidal"):
        raise ValueError(f"human_ogse must be 'apodized' or 'trapezoidal', got {human_ogse!r}")
    _trap = human_ogse == "trapezoidal"
    _human_delta_key = "DeltaRDapp_trap" if _trap else "DeltaRDapp"
    _human_ogse_total = "RDapp_OGSE_trap" if _trap else "RDapp_OGSE"
    _human_ogse_intra = "RDapp_OGSE_trap_intra" if _trap else "RDapp_OGSE_intra"
    _human_ogse_extra = "RDapp_OGSE_trap_extra" if _trap else "RDapp_OGSE_extra"

    data_root = os.path.abspath(data_root)
    results = {
        "human_b300": defaultdict(lambda: defaultdict(list)),
        "animal_b800": defaultdict(lambda: defaultdict(list)),
        "individual_pairs": {
            "human_b300": defaultdict(list),
            "animal_b800": defaultdict(list),
        },
        "components": {
            "intra": {
                "human_b300": {
                    "PGSE": defaultdict(lambda: defaultdict(list)),
                    "OGSE": defaultdict(lambda: defaultdict(list)),
                },
                "animal_b800": {
                    "PGSE": defaultdict(lambda: defaultdict(list)),
                    "OGSE": defaultdict(lambda: defaultdict(list)),
                },
            },
            "extra": {
                "human_b300": {
                    "PGSE": defaultdict(lambda: defaultdict(list)),
                    "OGSE": defaultdict(lambda: defaultdict(list)),
                },
                "animal_b800": {
                    "PGSE": defaultdict(lambda: defaultdict(list)),
                    "OGSE": defaultdict(lambda: defaultdict(list)),
                },
            },
            "total": {
                "human_b300": {
                    "PGSE": defaultdict(lambda: defaultdict(list)),
                    "OGSE": defaultdict(lambda: defaultdict(list)),
                },
                "animal_b800": {
                    "PGSE": defaultdict(lambda: defaultdict(list)),
                    "OGSE": defaultdict(lambda: defaultdict(list)),
                },
            },
        },
    }

    component_key_map = {
        "total": {
            "human_b300": {"PGSE": "RDapp_PGSE", "OGSE": _human_ogse_total},
            "animal_b800": {"PGSE": "RDapp_PGSE_2", "OGSE": "RDapp_OGSE_2"},
        },
        "intra": {
            "human_b300": {"PGSE": "RDapp_PGSE_intra", "OGSE": _human_ogse_intra},
            "animal_b800": {"PGSE": "RDapp_PGSE_intra_2", "OGSE": "RDapp_OGSE_intra_2"},
        },
        "extra": {
            "human_b300": {"PGSE": "RDapp_PGSE_extra", "OGSE": _human_ogse_extra},
            "animal_b800": {"PGSE": "RDapp_PGSE_extra_2", "OGSE": "RDapp_OGSE_extra_2"},
        },
    }

    for batch in sorted(os.listdir(data_root)):
        batch_path = os.path.join(data_root, batch)
        if not os.path.isdir(batch_path):
            continue
        if only_batch is not None and batch != only_batch:
            continue

        for param_group in sorted(os.listdir(batch_path)):
            param_group_path = os.path.join(batch_path, param_group)
            if not os.path.isdir(param_group_path):
                continue

            meta = _parse_param_group_folder(param_group)
            if meta is None:
                continue
            exp_diam, od = meta

            for substrate_id in sorted(os.listdir(param_group_path)):
                substrate_path = os.path.join(param_group_path, substrate_id)
                if not os.path.isdir(substrate_path):
                    continue

                rdapp_pkl = os.path.join(substrate_path, 'sim', 'RDapp', 'rdapp_result.pkl')
                if not os.path.isfile(rdapp_pkl):
                    print(f"  [skip] no rdapp_result.pkl in {batch}/{param_group}/{substrate_id}")
                    continue

                with open(rdapp_pkl, 'rb') as fh:
                    r = pickle.load(fh)

                substrate_mean_diam = _extract_substrate_mean_diameter_aim2(substrate_path)
                if substrate_mean_diam is None:
                    substrate_mean_diam = exp_diam
                    print(
                        f"  [warn] No substrate mean diameter PNG found for "
                        f"{batch}/{param_group}/{substrate_id}; "
                        f"falling back to param-group diameter={exp_diam:.3f}"
                    )

                eff_metrics = _extract_effective_diameter_metrics(substrate_path)
                if eff_metrics is None:
                    # Aim 2 data typically lacks the effective-diameter JSON.
                    # Fall back to the substrate mean diameter for all 3 metrics
                    # so the existing plotting code keeps working.
                    eff_metrics = {
                        "d_eff_p3_q2":     substrate_mean_diam,
                        "d_app_wide_pulse": substrate_mean_diam,
                        "d_app_internal":  substrate_mean_diam,
                    }

                delta_human = r.get(_human_delta_key, None)
                delta_animal = r.get('DeltaRDapp_2', None)
                human_pgse = r.get('RDapp_PGSE', None)
                human_ogse = r.get(_human_ogse_total, None)
                animal_pgse = r.get('RDapp_PGSE_2', None)
                animal_ogse = r.get('RDapp_OGSE_2', None)

                if delta_human is not None:
                    delta_human = float(delta_human)
                    results["human_b300"][od][substrate_mean_diam].append(delta_human)

                if delta_animal is not None:
                    delta_animal = float(delta_animal)
                    results["animal_b800"][od][substrate_mean_diam].append(delta_animal)

                if human_pgse is not None and human_ogse is not None:
                    results["individual_pairs"]["human_b300"][od].append({
                        "diameter": float(substrate_mean_diam),
                        "diameter_d_eff": float(eff_metrics["d_eff_p3_q2"]),
                        "diameter_d_wp":  float(eff_metrics["d_app_wide_pulse"]),
                        "diameter_d_int": float(eff_metrics["d_app_internal"]),
                        "pgse": float(human_pgse),
                        "ogse": float(human_ogse),
                        "substrate_id": substrate_id,
                        "exp_group": f"{batch}/{param_group}",
                    })

                if animal_pgse is not None and animal_ogse is not None:
                    results["individual_pairs"]["animal_b800"][od].append({
                        "diameter": float(substrate_mean_diam),
                        "diameter_d_eff": float(eff_metrics["d_eff_p3_q2"]),
                        "diameter_d_wp":  float(eff_metrics["d_app_wide_pulse"]),
                        "diameter_d_int": float(eff_metrics["d_app_internal"]),
                        "pgse": float(animal_pgse),
                        "ogse": float(animal_ogse),
                        "substrate_id": substrate_id,
                        "exp_group": f"{batch}/{param_group}",
                    })

                for component_name, scenario_map in component_key_map.items():
                    for scenario_key, seq_map in scenario_map.items():
                        for seq_label, rd_key in seq_map.items():
                            value = r.get(rd_key, None)
                            if value is not None:
                                results["components"][component_name][scenario_key][seq_label][od][substrate_mean_diam].append(float(value))

                print(
                    f"  {batch}/{param_group}/{substrate_id}  "
                    f"d={substrate_mean_diam:.3f} OD={od}"
                    f"  ΔRDapp_b300={delta_human if delta_human is not None else 'NA'}"
                    f"  ΔRDapp_b800={delta_animal if delta_animal is not None else 'NA'}"
                )

    return results


# ------------------------------------------------------------------
# Plot generation for one collected results dict
# ------------------------------------------------------------------

def _generate_plots(results: dict, output_path: str, make_summary: bool = True,
                    deff_xlim=None):
    """
    Produce the full ΔD⊥ plot set for one collected ``results`` dict.

    Writes the top-level summary figure to ``output_path`` (when
    ``make_summary`` is True) and the per-OD / per-metric figures (plus the
    ``Dperp_by_ODI`` subfolder) into the same directory as ``output_path``.

    ``deff_xlim`` optionally overrides the x-axis range of the effective
    diameter (``d_eff_p3q2``) per-OD plot, e.g. ``(2, 7)``.
    """
    # Top-level summary plot (mean ± std across replicates)
    if make_summary:
        plot_deltardapp(results, output_path=output_path)

    # Per-OD plots are written into the same plots dir as the main summary
    plots_dir = os.path.dirname(os.path.abspath(output_path))
    output_by_odi_dir = os.path.join(plots_dir, 'Dperp_by_ODI')

    for _, diameter_field, xlabel, tag, long_label in DIAMETER_METRICS:
        output_per_od = os.path.join(
            plots_dir,
            f'deltaDperp_per_od_linear_fit_{tag}.png',
        )
        plot_deltardapp_per_od(
            results,
            output_path=output_per_od,
            diameter_field=diameter_field,
            xlabel=xlabel,
            title_suffix=long_label,
            shared_xaxis=True,
            xlim=deff_xlim if tag == 'd_eff_p3q2' else None,
        )

        plot_individual_pgse_ogse_pairs_by_od(
            results,
            output_dir=output_by_odi_dir,
            diameter_field=diameter_field,
            xlabel=xlabel,
            tag=tag,
        )
        plot_individual_pgse_ogse_pairs_all_od_combined(
            results,
            output_dir=output_by_odi_dir,
            diameter_field=diameter_field,
            xlabel=xlabel,
            tag=tag,
        )


# ------------------------------------------------------------------
# Protocol gradient-waveform rendering (for the all_beads folder)
# ------------------------------------------------------------------

def _render_protocol_waveforms(out_dir: str, human_ogse: str = "trapezoidal") -> None:
    """
    Render the human + animal gradient waveforms used for the all_beads figures
    and save them into ``out_dir`` as physical G(t) = shape * Gmax.

    Human = slew PGSE + (trapezoidal|apodized) OGSE at the human_80_100 preset;
    Animal = ideal PGSE + apodized OGSE. Timings match the protocol configs in
    compute_rdapp_from_narrow_pulse.py.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from simulation_toolkit.simulation_engine.waveform_design import design_waveform
    except Exception as exc:
        print(f"  [warn] Skipping protocol waveform plots: {exc}")
        return

    human_ogse_shape = "ogse-trapezoidal" if human_ogse == "trapezoidal" else "ogse-apodized"
    designs = [
        ("human_pgse_slew", "Human PGSE (slew-limited)",
         dict(shape="pgse-slew", bvalue_s_mm2=300, te_ms=78, preset="human_80_100",
              little_delta_ms=12, big_delta_ms=50, dt_ms=0.005)),
        (f"human_ogse_{human_ogse}", f"Human OGSE ({human_ogse})",
         dict(shape=human_ogse_shape, bvalue_s_mm2=300, te_ms=78, preset="human_80_100",
              n_cycles=1, t_eff_ms=6.5, dt_ms=0.005)),
        ("animal_pgse", "Animal PGSE (ideal)",
         dict(shape="pgse", bvalue_s_mm2=800, te_ms=40, preset="animal_placeholder",
              little_delta_ms=3, big_delta_ms=26, dt_ms=0.005)),
        ("animal_ogse_apodized", "Animal OGSE (apodized)",
         dict(shape="ogse-apodized", bvalue_s_mm2=800, te_ms=40, preset="animal_placeholder",
              n_cycles=1, t_eff_ms=2.5, dt_ms=0.005)),
    ]

    os.makedirs(out_dir, exist_ok=True)
    results = []
    for name, label, kwargs in designs:
        res = design_waveform(**kwargs)
        if res.waveform is None:
            print(f"  [warn] {label} could not be designed: {res.messages}")
            continue
        results.append((name, label, res))

    if not results:
        return

    def _panel_title(label, res):
        return (f"{label}\nb={res.bvalue_s_mm2_actual:.0f} s/mm², "
                f"Gmax={res.gmax_mT_per_m:.1f} mT/m, TE={res.te_ms:.0f} ms")

    # Column 0 = Human (matching TE=78), column 1 = Animal (matching TE=40);
    # PGSE on top row, OGSE on bottom row. sharex per column aligns the TE.
    grid_pos = {
        "human_pgse_slew": (0, 0),
        f"human_ogse_{human_ogse}": (1, 0),
        "animal_pgse": (0, 1),
        "animal_ogse_apodized": (1, 1),
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col")
    used = set()
    for name, label, res in results:
        r, c = grid_pos[name]
        used.add((r, c))
        ax = axes[r, c]
        w = res.waveform
        ax.plot(w.t, w.wave * res.gmax_mT_per_m, linewidth=1.5)
        ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_title(_panel_title(label, res), fontsize=10)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("G (mT/m)")
        ax.grid(True, linestyle="--", alpha=0.35)
    for r in range(2):
        for c in range(2):
            if (r, c) not in used:
                axes[r, c].axis("off")
    fig.suptitle("Protocol gradient waveforms  (left: Human,  right: Animal)", fontsize=13)
    fig.tight_layout()
    combined = os.path.join(out_dir, "protocol_waveforms.png")
    fig.savefig(combined, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Protocol waveforms saved → {combined}")

    for name, label, res in results:
        w = res.waveform
        f, a = plt.subplots(figsize=(9, 3.6))
        a.plot(w.t, w.wave * res.gmax_mT_per_m, linewidth=1.6)
        a.axhline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        a.set_title(_panel_title(label, res).replace("\n", "  |  "), fontsize=10)
        a.set_xlabel("Time (ms)")
        a.set_ylabel("G (mT/m)")
        a.grid(True, linestyle="--", alpha=0.35)
        f.tight_layout()
        f.savefig(os.path.join(out_dir, f"waveform_{name}.png"), dpi=250, bbox_inches="tight")
        plt.close(f)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Plot ΔD⊥ vs axon diameter from pre-computed rdapp_result.pkl files '
            'for the Aim 2 three-level data layout.'
        )
    )
    parser.add_argument(
        'data_root',
        help='Root data folder, e.g. experiment/aim2-prep/data',
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output figure path (e.g. plots/deltaDperp_vs_diameter.png). '
             'If omitted, defaults to ``<data_root>/../plots/deltaDperp_vs_diameter.png``. '
             'Ignored when --per-batch is set.',
    )
    parser.add_argument(
        '--per-batch',
        action='store_true',
        help='Produce a separate plot set per beading batch, written to '
             '``<data_root>/../plots/<bead_label>/`` (e.g. plots/bead_0.3/).',
    )
    parser.add_argument(
        '--all-beads',
        action='store_true',
        help='Produce combined plots pooling ALL bead groups and ALL ODI '
             '(d_eff_p3q2 only), written to ``<data_root>/../plots/all_beads/``.',
    )
    parser.add_argument(
        '--human-ogse',
        choices=['apodized', 'trapezoidal'],
        default='apodized',
        help='Human OGSE variant: "apodized" (default) or "trapezoidal" '
             '(slew-limited, uses RDapp_OGSE_trap / DeltaRDapp_trap). '
             'Human PGSE is the slew-limited RDapp_PGSE in both cases; '
             'animal keys are unchanged.',
    )
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)

    # Default plots dir: next to data_root in a plots subfolder
    plots_dir = os.path.normpath(os.path.join(data_root, '..', 'plots'))

    if args.per_batch:
        any_plotted = False
        for batch in sorted(os.listdir(data_root)):
            batch_path = os.path.join(data_root, batch)
            if not os.path.isdir(batch_path):
                continue

            bead_label = _bead_label_for_batch(batch_path)
            if bead_label is None:
                print(f"  [skip] no Aim 2 param-group folders in batch '{batch}'")
                continue

            print(f"\nCollecting ΔD⊥ (Aim 2 layout) for batch '{batch}' → {bead_label}\n")
            results = collect_deltardapp_aim2(data_root, only_batch=batch,
                                              human_ogse=args.human_ogse)

            if (not results["human_b300"]) and (not results["animal_b800"]):
                print(f"  [skip] no rdapp_result.pkl files found in batch '{batch}'")
                continue

            batch_output = os.path.join(plots_dir, bead_label, 'deltaDperp_vs_diameter.png')
            deff_xlim = _DEFF_XLIM_BY_BEAD.get(bead_label)
            _generate_plots(results, batch_output, make_summary=False,
                            deff_xlim=deff_xlim)
            any_plotted = True

        if not any_plotted:
            print(
                "No rdapp_result.pkl files found in any batch. Run "
                "compute_rdapp_from_narrow_pulse_aim2.py first."
            )
            sys.exit(1)
        return

    if args.all_beads:
        print(f"\nCollecting ΔD⊥ (Aim 2 layout, all beads merged) from: {data_root}\n")
        results = collect_deltardapp_aim2(data_root, human_ogse=args.human_ogse)

        if (not results["human_b300"]) and (not results["animal_b800"]):
            print(
                "No rdapp_result.pkl files found. Run "
                "compute_rdapp_from_narrow_pulse_aim2.py first."
            )
            sys.exit(1)

        # d_eff_p3q2 metric only.
        _, deff_field, _, _, deff_long = DIAMETER_METRICS[0]
        deff_xlabel = r'$\langle d \rangle_{\mathrm{eff}}$ (µm)'
        all_beads_dir = os.path.join(plots_dir, 'all_beads')

        # Combined all-beads figures: protocol encoded by colour (Human/Animal),
        # ODI encoded by marker shape.
        plot_pgse_ogse_pairs_combined_all(
            results,
            output_path=os.path.join(all_beads_dir, 'Dperp_total_d_eff_p3q2.png'),
            diameter_field=deff_field,
            xlabel=deff_xlabel,
            marker_scheme='protocol_color',
        )
        plot_deltardapp_combined_all(
            results,
            output_path=os.path.join(all_beads_dir, 'deltaDperp_vs_deff_p3q2.png'),
            diameter_field=deff_field,
            xlabel=deff_xlabel,
            title_suffix=deff_long,
            marker_scheme='protocol_color',
        )
        _render_protocol_waveforms(all_beads_dir, human_ogse=args.human_ogse)
        return

    print(f"\nCollecting ΔD⊥ (Aim 2 layout) from: {data_root}\n")
    results = collect_deltardapp_aim2(data_root, human_ogse=args.human_ogse)

    if (not results["human_b300"]) and (not results["animal_b800"]):
        print(
            "No rdapp_result.pkl files found. Run "
            "compute_rdapp_from_narrow_pulse_aim2.py first."
        )
        sys.exit(1)

    output = args.output if args.output else os.path.join(plots_dir, 'deltaDperp_vs_diameter.png')
    _generate_plots(results, output)


if __name__ == '__main__':
    main()
