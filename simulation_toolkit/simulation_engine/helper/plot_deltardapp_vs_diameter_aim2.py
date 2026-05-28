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


def _parse_param_group_folder(name: str):
    """Return (diameter_float, od_int) parsed from an Aim 2 param-group folder, else None."""
    m = _AIM2_PARAM_GROUP_RE.match(name)
    if m is None:
        return None
    return float(m.group('diam')), int(m.group('od'))


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

def collect_deltardapp_aim2(data_root: str) -> dict:
    """
    Walk an Aim 2 ``data`` folder (three nesting levels) and collect
    DeltaRDapp values for both scanner scenarios.

    Returns a dict with the same shape as the Aim 1 ``collect_deltardapp``,
    so the existing plotting functions can be reused unchanged.
    """
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
            "human_b300": {"PGSE": "RDapp_PGSE", "OGSE": "RDapp_OGSE"},
            "animal_b800": {"PGSE": "RDapp_PGSE_2", "OGSE": "RDapp_OGSE_2"},
        },
        "intra": {
            "human_b300": {"PGSE": "RDapp_PGSE_intra", "OGSE": "RDapp_OGSE_intra"},
            "animal_b800": {"PGSE": "RDapp_PGSE_intra_2", "OGSE": "RDapp_OGSE_intra_2"},
        },
        "extra": {
            "human_b300": {"PGSE": "RDapp_PGSE_extra", "OGSE": "RDapp_OGSE_extra"},
            "animal_b800": {"PGSE": "RDapp_PGSE_extra_2", "OGSE": "RDapp_OGSE_extra_2"},
        },
    }

    for batch in sorted(os.listdir(data_root)):
        batch_path = os.path.join(data_root, batch)
        if not os.path.isdir(batch_path):
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

                delta_human = r.get('DeltaRDapp', None)
                delta_animal = r.get('DeltaRDapp_2', None)
                human_pgse = r.get('RDapp_PGSE', None)
                human_ogse = r.get('RDapp_OGSE', None)
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
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Plot ΔRDapp vs axon diameter from pre-computed rdapp_result.pkl files '
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
        help='Output figure path (e.g. plots/deltardapp_vs_diameter.png). '
             'If omitted, defaults to ``<data_root>/../plots/deltardapp_vs_diameter.png``.',
    )
    args = parser.parse_args()

    print(f"\nCollecting ΔRDapp (Aim 2 layout) from: {os.path.abspath(args.data_root)}\n")
    results = collect_deltardapp_aim2(args.data_root)

    if (not results["human_b300"]) and (not results["animal_b800"]):
        print(
            "No rdapp_result.pkl files found. Run "
            "compute_rdapp_from_narrow_pulse_aim2.py first."
        )
        sys.exit(1)

    # Default output: next to data_root in a plots subfolder
    plots_dir = os.path.normpath(
        os.path.join(os.path.abspath(args.data_root), '..', 'plots')
    )
    output = args.output if args.output else os.path.join(plots_dir, 'deltardapp_vs_diameter.png')

    # Top-level summary plot (mean ± std across replicates)
    plot_deltardapp(results, output_path=output)

    # Per-OD plots are written into the same plots dir as the main summary
    plots_dir = os.path.dirname(os.path.abspath(output))
    output_by_odi_dir = os.path.join(plots_dir, 'rdapp_by_ODI')

    for _, diameter_field, xlabel, tag, long_label in DIAMETER_METRICS:
        output_per_od = os.path.join(
            plots_dir,
            f'deltardapp_per_od_linear_fit_{tag}.png',
        )
        plot_deltardapp_per_od(
            results,
            output_path=output_per_od,
            diameter_field=diameter_field,
            xlabel=xlabel,
            title_suffix=long_label,
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


if __name__ == '__main__':
    main()
