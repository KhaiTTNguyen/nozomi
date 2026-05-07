"""
plot_deltardapp_vs_diameter.py
==============================
Plot ΔRDapp = RDapp_OGSE - RDapp_PGSE vs mean axon diameter, grouped by
orientation dispersion (OD).

Each experiment group folder encodes both diameter and OD:
    experiment_d<diam>_sig<sigma>_500axons_reproducibility_OD<od>/

Multiple substrate subfolders per group provide reproducibility replicates.
Results are shown as mean ± std across replicates.

Usage:
    python plot_deltardapp_vs_diameter.py <data_root> [--output <fig.png>]

    data_root   e.g.  experiment/visualization/data
"""

import argparse
import json
import os
import re
import sys
import pickle
from collections import defaultdict
from pathlib import Path

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import linregress

# ------------------------------------------------------------------
# Parse folder-name metadata
# ------------------------------------------------------------------
_EXP_RE = re.compile(
    r'(?:VF[\d.]+_)?experiment_d(?P<diam>[\d.]+)_.*_reproducibility_OD(?P<od>\d+)$'
)
_SUBSTRATE_MEAN_RE = re.compile(
    r'Optimized_diameter_distribution_mean(?P<mean>[\d.]+)_std[\d.]+_.*\.png$'
)
_EFF_DIAM_JSON_RE = re.compile(r'outer_effective_axon_diameter_stats.*\.json$')

# Diameter metrics available from the effective-diameter JSON.
# Each entry: (metric_key_in_results, entry_field, x-axis label, filename tag)
DIAMETER_METRICS = [
    (
        "d_eff_p3_q2",
        "diameter_d_eff",
        r"$d_{\mathrm{eff}}$ ($p=3,\,q=2$) (µm)",
        "d_eff_p3q2",
        r"Area-weighted mean: $d_{\mathrm{eff}}$ ($p=3,q=2$)",
    ),
    (
        "d_app_wide_pulse",
        "diameter_d_wp",
        r"$d_{\mathrm{app}}$ (wide pulse) (µm)",
        "d_app_wp",
        r"Wide-pulse apparent diameter: $d_{\mathrm{app,WP}}$",
    ),
    (
        "d_app_internal",
        "diameter_d_int",
        r"$d_{\mathrm{app}}$ (internal) (µm)",
        "d_app_int",
        r"Apparent internal diameter: $d_{\mathrm{app,int}}$",
    ),
]


def _parse_exp_folder(name: str):
    """Return (diameter_float, od_int) or None."""
    m = _EXP_RE.match(name)
    if m is None:
        return None
    return float(m.group('diam')), int(m.group('od'))


def _extract_substrate_mean_diameter(substrate_path: str):
    """Return mean diameter (um) parsed from substrate_stats filename, else None."""
    stats_dir = os.path.join(substrate_path, 'figs', 'substrate_stats')
    if not os.path.isdir(stats_dir):
        return None

    candidates = []
    for fn in os.listdir(stats_dir):
        m = _SUBSTRATE_MEAN_RE.match(fn)
        if m is not None:
            candidates.append((fn, float(m.group('mean'))))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _extract_effective_diameter_metrics(substrate_path: str):
    """
    Read ``outer_effective_axon_diameter_stats*.json`` and return a dict with
    the three bundle-level diameter metrics, or None if the file is missing.

    Returns
    -------
    dict with keys:
        ``d_eff_p3_q2``, ``d_app_wide_pulse``, ``d_app_internal``  (all in µm)
    """
    stats_dir = os.path.join(substrate_path, 'figs', 'substrate_stats')
    if not os.path.isdir(stats_dir):
        return None

    candidates = sorted(
        fn for fn in os.listdir(stats_dir) if _EFF_DIAM_JSON_RE.match(fn)
    )
    if not candidates:
        return None

    json_path = os.path.join(stats_dir, candidates[-1])
    with open(json_path) as fh:
        stats = json.load(fh)

    bundle = stats.get("bundle", {})
    d_eff = bundle.get("d_eff_p3_q2_um")
    d_wp  = bundle.get("d_app_wide_pulse_um")
    d_int = bundle.get("d_app_internal_um")

    if any(v is None for v in (d_eff, d_wp, d_int)):
        return None

    return {
        "d_eff_p3_q2":    float(d_eff),
        "d_app_wide_pulse": float(d_wp),
        "d_app_internal":  float(d_int),
    }


# ------------------------------------------------------------------
# Data collection
# ------------------------------------------------------------------

def collect_deltardapp(data_root: str) -> dict:
    """
    Walk data_root and collect DeltaRDapp values for both scanner scenarios.

    Returns
    -------
        dict keyed by scenario:
            - "human_b300"  → {od: {diam: [DeltaRDapp]}}
            - "animal_b800" → {od: {diam: [DeltaRDapp_2]}}
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

    for exp_group in sorted(os.listdir(data_root)):
        meta = _parse_exp_folder(exp_group)
        if meta is None:
            continue
        exp_diam, od = meta
        exp_path = os.path.join(data_root, exp_group)

        for substrate_id in sorted(os.listdir(exp_path)):
            substrate_path = os.path.join(exp_path, substrate_id)
            if not os.path.isdir(substrate_path):
                continue

            rdapp_pkl = os.path.join(substrate_path, 'sim', 'RDapp', 'rdapp_result.pkl')
            if not os.path.isfile(rdapp_pkl):
                print(f"  [skip] no rdapp_result.pkl in {substrate_id}")
                continue

            with open(rdapp_pkl, 'rb') as fh:
                r = pickle.load(fh)

            substrate_mean_diam = _extract_substrate_mean_diameter(substrate_path)
            if substrate_mean_diam is None:
                substrate_mean_diam = exp_diam
                print(
                    f"  [warn] No substrate mean diameter PNG found for {substrate_id}; "
                    f"falling back to experiment diameter={exp_diam:.3f}"
                )

            eff_metrics = _extract_effective_diameter_metrics(substrate_path)
            if eff_metrics is None:
                print(
                    f"  [warn] No effective diameter JSON found for {substrate_id}; "
                    f"falling back to experiment diameter for all 3 metrics"
                )
                eff_metrics = {
                    "d_eff_p3_q2":     exp_diam,
                    "d_app_wide_pulse": exp_diam,
                    "d_app_internal":  exp_diam,
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
                    "exp_group": exp_group,
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
                    "exp_group": exp_group,
                })

            for component_name, scenario_map in component_key_map.items():
                for scenario_key, seq_map in scenario_map.items():
                    for seq_label, rd_key in seq_map.items():
                        value = r.get(rd_key, None)
                        if value is not None:
                            results["components"][component_name][scenario_key][seq_label][od][substrate_mean_diam].append(float(value))

            print(
                f"  {exp_group}/{substrate_id}  d={substrate_mean_diam:.3f} OD={od}"
                f"  ΔRDapp_b300={delta_human if delta_human is not None else 'NA'}"
                f"  ΔRDapp_b800={delta_animal if delta_animal is not None else 'NA'}"
            )

    return results


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

# Colour and marker map per OD value; fallback for unexpected OD values
_OD_STYLES = {
    10:  {'color': '#2166ac', 'marker': 'o', 'label': 'OD10  (κ=10, ODI≈0.064)'},
    20:  {'color': '#f4a582', 'marker': 's', 'label': 'OD20  (κ=20, ODI≈0.032)'},
    200: {'color': '#d6604d', 'marker': '^', 'label': 'OD200 (κ=200, ODI≈0.003)'},
}


def plot_deltardapp(results: dict, output_path: Optional[str] = None):
    """
    Parameters
    ----------
    results     : output of collect_deltardapp()
    output_path : file to save figure; if None, shows interactively
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), sharey=True)
    scenario_meta = [
        ("human_b300", "Human scanner", r"$b=300$ s/mm$^2$"),
        ("animal_b800", "Animal scanner", r"$b=800$ s/mm$^2$"),
    ]

    for ax, (scenario_key, scanner_label, b_label) in zip(axes, scenario_meta):
        scenario_results = results.get(scenario_key, {})
        od_values = sorted(scenario_results.keys())

        for od in od_values:
            diam_dict = scenario_results[od]
            diams_sorted = sorted(diam_dict.keys())
            if not diams_sorted:
                continue

            means = np.array([np.mean(diam_dict[d]) for d in diams_sorted])
            stds = np.array([
                np.std(diam_dict[d], ddof=1) if len(diam_dict[d]) > 1 else 0.0
                for d in diams_sorted
            ])
            counts = [len(diam_dict[d]) for d in diams_sorted]

            style = _OD_STYLES.get(od, {'color': 'gray', 'marker': 'x',
                                         'label': f'OD{od}'})

            ax.errorbar(
                diams_sorted, means, yerr=stds,
                color=style['color'],
                marker=style['marker'],
                label=f"{style['label']}  (n={counts[0]})",
                linewidth=1.8,
                markersize=7,
                capsize=5,
                capthick=1.5,
                elinewidth=1.2,
            )

        ax.set_xlabel('Mean axon diameter (µm)', fontsize=12)
        ax.set_title(f'{scanner_label}\n{b_label}', fontsize=12)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax.grid(True, which='major', linestyle='--', alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', alpha=0.25)
        ax.legend(fontsize=9, framealpha=0.9)

    axes[0].set_ylabel(
        r'$\Delta RD^{app} = RD^{app}_{\mathrm{OGSE}} - RD^{app}_{\mathrm{PGSE}}$'
        '\n(µm²/ms)',
        fontsize=12,
    )
    fig.suptitle(
        r'Wide-pulse apparent radial diffusion contrast: '
        r'$\Delta RD^{app}$ vs mean axon diameter',
        fontsize=13,
    )
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved → {output_path}")
    else:
        plt.show()

    return fig, ax


def plot_rdapp_component(results: dict, component: str, output_path: Optional[str] = None):
    """
    Plot RDapp_PGSE and RDapp_OGSE versus mean diameter for one compartment.

    Parameters
    ----------
    results     : output of collect_deltardapp()
    component   : one of {"intra", "extra", "total"}
    output_path : file to save figure; if None, shows interactively
    """
    component_results = results.get("components", {}).get(component, {})
    if not component_results:
        raise ValueError(f"No component results found for '{component}'.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    scenario_meta = [
        ("human_b300", "Human scanner", r"$b=300$ s/mm$^2$"),
        ("animal_b800", "Animal scanner", r"$b=800$ s/mm$^2$"),
    ]

    component_label_map = {
        "intra": "Intra-axonal",
        "extra": "Extra-axonal",
        "total": "Volume-fraction weighted total",
    }
    component_label = component_label_map.get(component, component)

    for ax, (scenario_key, scanner_label, b_label) in zip(axes, scenario_meta):
        seq_maps = component_results.get(scenario_key, {})
        pgse_map = seq_maps.get("PGSE", {})
        ogse_map = seq_maps.get("OGSE", {})
        od_values = sorted(set(pgse_map.keys()) | set(ogse_map.keys()))

        # Plot sequence groups in order so legend is grouped by OGSE then PGSE.
        for seq_name, seq_map in [
            ("OGSE", ogse_map),
            ("PGSE", pgse_map),
        ]:
            for od in od_values:
                style = _OD_STYLES.get(od, {'color': 'gray', 'marker': 'x',
                                            'label': f'OD{od}'})
                diam_dict = seq_map.get(od, {})
                diams_sorted = sorted(diam_dict.keys())
                if not diams_sorted:
                    continue

                means = np.array([np.mean(diam_dict[d]) for d in diams_sorted])
                stds = np.array([
                    np.std(diam_dict[d], ddof=1) if len(diam_dict[d]) > 1 else 0.0
                    for d in diams_sorted
                ])
                counts = [len(diam_dict[d]) for d in diams_sorted]

                marker_face = 'white' if seq_name == "OGSE" else style['color']
                ax.errorbar(
                    diams_sorted, means, yerr=stds,
                    color=style['color'],
                    marker=style['marker'],
                    markerfacecolor=marker_face,
                    markeredgecolor=style['color'],
                    linestyle='None',
                    linewidth=1.8,
                    markersize=7,
                    capsize=5,
                    capthick=1.5,
                    elinewidth=1.2,
                    label=f"{style['label']} {seq_name} (n={counts[0]})",
                )

        ax.set_xlabel('Mean axon diameter (µm)', fontsize=12)
        ax.set_title(f'{scanner_label}\n{b_label}', fontsize=12)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax.grid(True, which='major', linestyle='--', alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', alpha=0.25)
        ax.legend(fontsize=8, framealpha=0.9)

    axes[0].set_ylabel(r'$RD^{app}_{\perp}$ (µm²/ms)', fontsize=12)
    fig.suptitle(
        f'{component_label} compartment: ' +
        r'$RD^{app}_{\mathrm{PGSE}}$ and $RD^{app}_{\mathrm{OGSE}}$ vs mean axon diameter',
        fontsize=13,
    )
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved → {output_path}")
    else:
        plt.show()

    return fig, axes


def plot_rdapp_total_by_od(results: dict, output_dir: str):
    """
    Create one figure per OD value for the total compartment.
    Each figure contains two panels (human and animal), and each panel shows
    OGSE vs PGSE RDapp as marker-only points with error bars.

    Parameters
    ----------
    results    : output of collect_deltardapp()
    output_dir : directory where figures are saved
    """
    total_results = results.get("components", {}).get("total", {})
    human_maps = total_results.get("human_b300", {})
    animal_maps = total_results.get("animal_b800", {})

    human_pgse = human_maps.get("PGSE", {})
    human_ogse = human_maps.get("OGSE", {})
    animal_pgse = animal_maps.get("PGSE", {})
    animal_ogse = animal_maps.get("OGSE", {})

    od_values = sorted(
        set(human_pgse.keys()) | set(human_ogse.keys()) |
        set(animal_pgse.keys()) | set(animal_ogse.keys())
    )
    if not od_values:
        print("No OD-specific total RDapp values found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    scenario_meta = [
        ("human_b300", "Human scanner", r"$b=300$ s/mm$^2$"),
        ("animal_b800", "Animal scanner", r"$b=800$ s/mm$^2$"),
    ]

    for od in od_values:
        style = _OD_STYLES.get(od, {'color': 'gray', 'marker': 'x', 'label': f'OD{od}'})
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), sharey=True)
        fig.subplots_adjust(wspace=0.08)

        for ax, (scenario_key, scanner_label, b_label) in zip(axes, scenario_meta):
            seq_maps = total_results.get(scenario_key, {})
            pgse_map = seq_maps.get("PGSE", {})
            ogse_map = seq_maps.get("OGSE", {})

            for seq_name, diam_dict in [
                ("OGSE", ogse_map.get(od, {})),
                ("PGSE", pgse_map.get(od, {})),
            ]:
                diams_sorted = sorted(diam_dict.keys())
                if not diams_sorted:
                    continue

                means = np.array([np.mean(diam_dict[d]) for d in diams_sorted])
                stds = np.array([
                    np.std(diam_dict[d], ddof=1) if len(diam_dict[d]) > 1 else 0.0
                    for d in diams_sorted
                ])
                counts = [len(diam_dict[d]) for d in diams_sorted]

                marker_face = 'white' if seq_name == "OGSE" else style['color']
                ax.errorbar(
                    diams_sorted, means, yerr=stds,
                    color=style['color'],
                    marker=style['marker'],
                    markerfacecolor=marker_face,
                    markeredgecolor=style['color'],
                    linestyle='None',
                    linewidth=1.8,
                    markersize=7,
                    capsize=5,
                    capthick=1.5,
                    elinewidth=1.2,
                    label=f"{seq_name} (n={counts[0]})",
                )

            ax.set_xlabel('Mean axon diameter (µm)', fontsize=12)
            ax.set_title(f'{scanner_label}\n{b_label}', fontsize=12)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            ax.grid(True, which='major', linestyle='--', alpha=0.5)
            ax.grid(True, which='minor', linestyle=':', alpha=0.25)
            ax.legend(fontsize=10, framealpha=0.9)

        axes[0].set_ylabel(r'$RD^{app}_{\perp}$ (µm²/ms)', fontsize=12)
        fig.suptitle(
            f"{style['label']} total compartment: " +
            r'$RD^{app}_{\mathrm{PGSE}}$ and $RD^{app}_{\mathrm{OGSE}}$ vs mean axon diameter',
            fontsize=14,
        )
        fig.tight_layout()

        out_path = os.path.join(output_dir, f'rdapp_total_{od}.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure saved → {out_path}")


def plot_rdapp_total_by_od_combined(results: dict, output_dir: str):
    """
    Create one combined figure with one row per OD value and two columns
    (human and animal scanner panels) for total-compartment RDapp.

    Parameters
    ----------
    results    : output of collect_deltardapp()
    output_dir : directory where the combined figure is saved
    """
    total_results = results.get("components", {}).get("total", {})
    human_maps = total_results.get("human_b300", {})
    animal_maps = total_results.get("animal_b800", {})

    human_pgse = human_maps.get("PGSE", {})
    human_ogse = human_maps.get("OGSE", {})
    animal_pgse = animal_maps.get("PGSE", {})
    animal_ogse = animal_maps.get("OGSE", {})

    od_values = sorted(
        set(human_pgse.keys()) | set(human_ogse.keys()) |
        set(animal_pgse.keys()) | set(animal_ogse.keys())
    )
    if not od_values:
        print("No OD-specific total RDapp values found for combined figure.")
        return

    os.makedirs(output_dir, exist_ok=True)
    n_od = len(od_values)
    fig, axes = plt.subplots(n_od, 2, figsize=(11, 5.0 * n_od), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.08, hspace=0.22)
    if n_od == 1:
        axes = np.array([axes])

    scenario_meta = [
        ("human_b300", "Human scanner", r"$b=300$ s/mm$^2$"),
        ("animal_b800", "Animal scanner", r"$b=800$ s/mm$^2$"),
    ]

    for row_idx, od in enumerate(od_values):
        style = _OD_STYLES.get(od, {'color': 'gray', 'marker': 'x', 'label': f'OD{od}'})
        for col_idx, (scenario_key, scanner_label, b_label) in enumerate(scenario_meta):
            ax = axes[row_idx, col_idx]
            seq_maps = total_results.get(scenario_key, {})
            pgse_map = seq_maps.get("PGSE", {})
            ogse_map = seq_maps.get("OGSE", {})

            for seq_name, diam_dict in [
                ("OGSE", ogse_map.get(od, {})),
                ("PGSE", pgse_map.get(od, {})),
            ]:
                diams_sorted = sorted(diam_dict.keys())
                if not diams_sorted:
                    continue

                means = np.array([np.mean(diam_dict[d]) for d in diams_sorted])
                stds = np.array([
                    np.std(diam_dict[d], ddof=1) if len(diam_dict[d]) > 1 else 0.0
                    for d in diams_sorted
                ])
                counts = [len(diam_dict[d]) for d in diams_sorted]

                marker_face = 'white' if seq_name == "OGSE" else style['color']
                ax.errorbar(
                    diams_sorted, means, yerr=stds,
                    color=style['color'],
                    marker=style['marker'],
                    markerfacecolor=marker_face,
                    markeredgecolor=style['color'],
                    linestyle='None',
                    linewidth=1.8,
                    markersize=7,
                    capsize=5,
                    capthick=1.5,
                    elinewidth=1.2,
                    label=f"{seq_name} (n={counts[0]})",
                )

            if row_idx == 0:
                ax.set_title(f'{scanner_label}\n{b_label}', fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{style['label']}\n" + r'$RD^{app}_{\perp}$ (µm²/ms)',
                    fontsize=11,
                )
            if row_idx == n_od - 1:
                ax.set_xlabel('Mean axon diameter (µm)', fontsize=12)

            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            ax.grid(True, which='major', linestyle='--', alpha=0.5)
            ax.grid(True, which='minor', linestyle=':', alpha=0.25)
            ax.legend(fontsize=10, framealpha=0.9)

    fig.suptitle(
        r'Total compartment: $RD^{app}_{\mathrm{PGSE}}$ and '
        r'$RD^{app}_{\mathrm{OGSE}}$ by OD',
        fontsize=14,
    )
    fig.tight_layout()

    out_path = os.path.join(output_dir, 'rdapp_total_all_ODI_combined.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved → {out_path}")


def plot_individual_pgse_ogse_pairs_by_od(
        results: dict,
        output_dir: str,
        diameter_field: str = "diameter",
        xlabel: str = "Mean axon diameter (µm)",
        tag: str = "",
):
    """
    Create one figure per OD value with two panels (human/animal), plotting
    individual substrate PGSE/OGSE pairs at each substrate's mean diameter.

    Parameters
    ----------
    diameter_field : key in each entry dict to use as x-axis value
    xlabel         : x-axis label string
    tag            : short string appended to output filename (e.g. ``d_eff_p3q2``)
    """
    pairs = results.get("individual_pairs", {})
    human_pairs_by_od = pairs.get("human_b300", {})
    animal_pairs_by_od = pairs.get("animal_b800", {})
    od_values = sorted(set(human_pairs_by_od.keys()) | set(animal_pairs_by_od.keys()))
    if not od_values:
        print("No individual PGSE/OGSE pairs found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    scenario_meta = [
        ("human_b300", "Human scanner", r"$b=300$ s/mm$^2$"),
        ("animal_b800", "Animal scanner", r"$b=800$ s/mm$^2$"),
    ]

    for od in od_values:
        style = _OD_STYLES.get(od, {'color': 'gray', 'marker': 'x', 'label': f'OD{od}'})
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

        for ax, (scenario_key, scanner_label, b_label) in zip(axes, scenario_meta):
            entries = sorted(
                pairs.get(scenario_key, {}).get(od, []),
                key=lambda item: item[diameter_field],
            )

            for idx, item in enumerate(entries):
                x = item[diameter_field]
                y_pgse = item["pgse"]
                y_ogse = item["ogse"]

                pgse_label = 'PGSE' if idx == 0 else '_nolegend_'
                ogse_label = 'OGSE' if idx == 0 else '_nolegend_'
                ax.scatter(
                    x, y_pgse,
                    s=36,
                    marker=style['marker'],
                    facecolor='white',
                    edgecolor=style['color'],
                    linewidth=1.4,
                    label=pgse_label,
                    zorder=3,
                )
                ax.scatter(
                    x, y_ogse,
                    marker=style['marker'],
                    color=style['color'],
                    edgecolor=style['color'],
                    label=ogse_label,
                    zorder=3,
                )

            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_title(f'{scanner_label}\n{b_label}', fontsize=12)
            ax.grid(True, which='major', linestyle='--', alpha=0.5)
            ax.grid(True, which='minor', linestyle=':', alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ordered_handles = []
                ordered_labels = []
                for target in ('OGSE', 'PGSE'):
                    for h, l in zip(handles, labels):
                        if l == target and l not in ordered_labels:
                            ordered_handles.append(h)
                            ordered_labels.append(l)
                ax.legend(
                    ordered_handles,
                    ordered_labels,
                    fontsize=14,
                    framealpha=0.9,
                    loc='upper left',
                )

        axes[0].set_ylabel(r'$RD^{app}_{\perp}$ (µm²/ms)', fontsize=12)
        fig.suptitle(
            f"{style['label']} individual substrate pairs: "
            r"$RD^{app}_{\mathrm{PGSE}}$ and $RD^{app}_{\mathrm{OGSE}}$",
            fontsize=13,
        )
        fig.tight_layout()

        fname_tag = f'_{tag}' if tag else ''
        out_path = os.path.join(output_dir, f'rdapp_total_individual_pairs_OD{od}{fname_tag}.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure saved → {out_path}")


def plot_individual_pgse_ogse_pairs_all_od_combined(
        results: dict,
        output_dir: str,
        diameter_field: str = "diameter",
        xlabel: str = "Mean axon diameter (µm)",
        tag: str = "",
):
    """
    Create one large combined figure containing all OD values.
    Rows correspond to OD values and columns are human/animal panels.

    Parameters
    ----------
    diameter_field : key in each entry dict to use as x-axis value
    xlabel         : x-axis label string
    tag            : short string appended to output filename
    """
    pairs = results.get("individual_pairs", {})
    human_pairs_by_od = pairs.get("human_b300", {})
    animal_pairs_by_od = pairs.get("animal_b800", {})
    od_values = sorted(set(human_pairs_by_od.keys()) | set(animal_pairs_by_od.keys()))
    if not od_values:
        print("No individual PGSE/OGSE pairs found for combined figure.")
        return

    os.makedirs(output_dir, exist_ok=True)
    scenario_meta = [
        ("human_b300", "Human scanner", r"$b=300$ s/mm$^2$"),
        ("animal_b800", "Animal scanner", r"$b=800$ s/mm$^2$"),
    ]

    n_od = len(od_values)
    fig, axes = plt.subplots(n_od, 2, figsize=(13, 4.4 * n_od), sharex=True, sharey=True)
    if n_od == 1:
        axes = np.array([axes])

    for row_idx, od in enumerate(od_values):
        style = _OD_STYLES.get(od, {'color': 'gray', 'marker': 'x', 'label': f'OD{od}'})

        for col_idx, (scenario_key, scanner_label, b_label) in enumerate(scenario_meta):
            ax = axes[row_idx, col_idx]
            entries = sorted(
                pairs.get(scenario_key, {}).get(od, []),
                key=lambda item: item[diameter_field],
            )

            for idx, item in enumerate(entries):
                x = item[diameter_field]
                y_pgse = item["pgse"]
                y_ogse = item["ogse"]

                pgse_label = 'PGSE' if idx == 0 else '_nolegend_'
                ogse_label = 'OGSE' if idx == 0 else '_nolegend_'
                ax.scatter(
                    x, y_pgse,
                    s=36,
                    marker=style['marker'],
                    facecolor='white',
                    edgecolor=style['color'],
                    linewidth=1.4,
                    label=pgse_label,
                    zorder=3,
                )
                ax.scatter(
                    x, y_ogse,
                    s=40,
                    marker=style['marker'],
                    color=style['color'],
                    edgecolor=style['color'],
                    label=ogse_label,
                    zorder=3,
                )

            if row_idx == 0:
                ax.set_title(f'{scanner_label}\n{b_label}', fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{style['label']}\n" + r'$RD^{app}_{\perp}$ (µm²/ms)',
                    fontsize=11,
                )
            if row_idx == n_od - 1:
                ax.set_xlabel(xlabel, fontsize=12)

            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            ax.grid(True, which='major', linestyle='--', alpha=0.5)
            ax.grid(True, which='minor', linestyle=':', alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ordered_handles = []
                ordered_labels = []
                for target in ('OGSE', 'PGSE'):
                    for h, l in zip(handles, labels):
                        if l == target and l not in ordered_labels:
                            ordered_handles.append(h)
                            ordered_labels.append(l)
                ax.legend(
                    ordered_handles,
                    ordered_labels,
                    fontsize=14,
                    framealpha=0.9,
                    loc='upper left',
                )

    fig.suptitle(
        r'All OD values: individual substrate ' +
        r'$RD^{app}_{\mathrm{PGSE}}$ and $RD^{app}_{\mathrm{OGSE}}$',
        fontsize=14,
    )
    fig.tight_layout()

    fname_tag = f'_{tag}' if tag else ''
    out_path = os.path.join(output_dir, f'rdapp_total_individual_pairs_all_OD_combined{fname_tag}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved → {out_path}")


def plot_rdapp_components_stacked(results: dict, output_path: Optional[str] = None):
    """
    Plot a single figure with 3 stacked rows (intra, extra, total), each with
    PGSE and OGSE RDapp versus mean diameter.

    Parameters
    ----------
    results     : output of collect_deltardapp()
    output_path : file to save figure; if None, shows interactively
    """
    components = [
        ("intra", "Intra-axonal"),
        ("extra", "Extra-axonal"),
        ("total", "Volume-fraction weighted total"),
    ]
    scenario_meta = [
        ("human_b300", "Human scanner", r"$b=300$ s/mm$^2$"),
        ("animal_b800", "Animal scanner", r"$b=800$ s/mm$^2$"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True)

    for row_idx, (component, component_label) in enumerate(components):
        component_results = results.get("components", {}).get(component, {})

        for col_idx, (scenario_key, scanner_label, b_label) in enumerate(scenario_meta):
            ax = axes[row_idx, col_idx]

            seq_maps = component_results.get(scenario_key, {})
            pgse_map = seq_maps.get("PGSE", {})
            ogse_map = seq_maps.get("OGSE", {})
            od_values = sorted(set(pgse_map.keys()) | set(ogse_map.keys()))

            for od in od_values:
                style = _OD_STYLES.get(od, {'color': 'gray', 'marker': 'x',
                                            'label': f'OD{od}'})

                for seq_name, diam_dict, line_style, marker_face in [
                    ("PGSE", pgse_map.get(od, {}), '-', style['color']),
                    ("OGSE", ogse_map.get(od, {}), '--', 'white'),
                ]:
                    diams_sorted = sorted(diam_dict.keys())
                    if not diams_sorted:
                        continue

                    means = np.array([np.mean(diam_dict[d]) for d in diams_sorted])
                    stds = np.array([
                        np.std(diam_dict[d], ddof=1) if len(diam_dict[d]) > 1 else 0.0
                        for d in diams_sorted
                    ])
                    counts = [len(diam_dict[d]) for d in diams_sorted]

                    ax.errorbar(
                        diams_sorted, means, yerr=stds,
                        color=style['color'],
                        marker=style['marker'],
                        markerfacecolor=marker_face,
                        markeredgecolor=style['color'],
                        linestyle=line_style,
                        linewidth=1.6,
                        markersize=6,
                        capsize=4,
                        capthick=1.2,
                        elinewidth=1.0,
                        label=f"{style['label']} {seq_name} (n={counts[0]})",
                    )

            if row_idx == 0:
                ax.set_title(f'{scanner_label}\n{b_label}', fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f'{component_label}\n$RD^{{app}}_{{\\perp}}$ (µm²/ms)', fontsize=11)

            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            ax.grid(True, which='major', linestyle='--', alpha=0.5)
            ax.grid(True, which='minor', linestyle=':', alpha=0.25)
            ax.legend(fontsize=7, framealpha=0.9, ncol=1)

    for ax in axes[-1, :]:
        ax.set_xlabel('Mean axon diameter (µm)', fontsize=12)

    fig.suptitle(
        r'Compartment-wise $RD^{app}_{\mathrm{PGSE}}$ and '
        r'$RD^{app}_{\mathrm{OGSE}}$ vs mean axon diameter',
        fontsize=14,
    )
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved → {output_path}")
    else:
        plt.show()

    return fig, axes


# ------------------------------------------------------------------
# Per-OD plot with linear fit
# ------------------------------------------------------------------

def plot_deltardapp_per_od(
        results: dict,
        output_path: Optional[str] = None,
        diameter_field: str = "diameter",
        xlabel: str = "Mean axon diameter (µm)",
        title_suffix: str = "",
):
    """
    One subplot per OD value, each showing:
      - Individual replicate scatter points
    - Linear fits computed from individual scattered points
      - Linear fit  ΔRDapp = α·d + β
      - Pearson r annotation

    Parameters
    ----------
    results        : output of collect_deltardapp()
    output_path    : file to save figure; if None, shows interactively
    diameter_field : key in each ``individual_pairs`` entry to use as x-axis
    xlabel         : x-axis label string
    title_suffix   : appended to the figure suptitle (e.g. diameter type name)
    """
    pair_results = results.get("individual_pairs", {})
    human_pairs = pair_results.get("human_b300", {})
    animal_pairs = pair_results.get("animal_b800", {})
    od_values = sorted(set(human_pairs.keys()) | set(animal_pairs.keys()))
    n_od = len(od_values)

    fig, axes = plt.subplots(1, n_od, figsize=(5 * n_od, 5), sharey=True)
    if n_od == 1:
        axes = [axes]

    for ax, od in zip(axes, od_values):
        style = _OD_STYLES.get(od, {'color': 'gray', 'marker': 'o',
                                     'label': f'OD{od}'})
        color = style['color']

        scenario_specs = [
            ("human_b300", human_pairs.get(od, []), '-', 'Human b=300', 'D'),
            ("animal_b800", animal_pairs.get(od, []), '--', 'Animal b=800', 'X'),
        ]

        ann_y = 0.96
        for _, entries, line_style, scenario_label, scatter_marker in scenario_specs:
            if not entries:
                continue

            # Build per-substrate scattered delta values: ΔRDapp = OGSE - PGSE.
            all_x = np.array([float(item[diameter_field]) for item in entries], dtype=float)
            all_y = np.array(
                [float(item["ogse"]) - float(item["pgse"]) for item in entries],
                dtype=float,
            )
            if len(all_x) < 2:
                continue

            slope, intercept, r_value, p_value, _ = linregress(all_x, all_y)

            ax.scatter(
                all_x, all_y,
                color=color, marker=scatter_marker,
                s=40, alpha=0.55, zorder=3,
                label=f'{scenario_label}: data (n={len(all_x)})',
            )

            x_fit = np.linspace(
                float(np.min(all_x)) * 0.88,
                float(np.max(all_x)) * 1.06,
                200,
            )
            y_fit = slope * x_fit + intercept
            sign = '+' if intercept >= 0 else '-'
            ax.plot(
                x_fit, y_fit,
                color=color, linestyle=line_style, linewidth=1.6, zorder=3,
                label=(
                    f'{scenario_label} fit: '
                    rf'$\beta{sign if sign=="+" else "-"}{abs(intercept):.4f}$'
                ),
            )

            ax.text(
                0.05, ann_y,
                f'{scenario_label}: ' + rf'$r={r_value:.4f}$' + ('*' if p_value < 0.05 else ''),
                transform=ax.transAxes,
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.7),
            )
            ann_y -= 0.11

        ax.set_title(f'{style["label"]}', fontsize=12)
        ax.set_xlabel(xlabel, fontsize=12)
        if od == od_values[0]:
            ax.set_ylabel(
                r'$\Delta RD^{app}$ (µm²/ms)',
                fontsize=12,
            )
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax.grid(True, which='major', linestyle='--', alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', alpha=0.25)
        ax.legend(fontsize=9, framealpha=0.9)

    suptitle = (
        r'$\Delta RD^{app} = \alpha \cdot d + \beta$ — linear fit per OD '
        r'(human $b=300$ and animal $b=800$)'
    )
    if title_suffix:
        suptitle += f'\n{title_suffix}'
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved → {output_path}")
    else:
        plt.show()

    return fig, axes


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot ΔRDapp vs axon diameter from pre-computed rdapp_result.pkl files.'
    )
    parser.add_argument(
        'data_root',
        help='Root data folder, e.g. experiment/visualization/data',
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output figure path (e.g. plots/deltardapp_vs_diameter.png). '
             'If omitted, shows figure interactively.',
    )
    args = parser.parse_args()

    # Make nozomi importable when run as a standalone script
    _nozomi_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(_nozomi_root) not in sys.path:
        sys.path.insert(0, str(_nozomi_root))

    print(f"\nCollecting ΔRDapp from: {os.path.abspath(args.data_root)}\n")
    results = collect_deltardapp(args.data_root)

    if (not results["human_b300"]) and (not results["animal_b800"]):
        print("No rdapp_result.pkl files found. Run compute_rdapp_from_narrow_pulse.py first.")
        sys.exit(1)

    # Default output: next to the data_root in a plots subfolder
    plots_dir = os.path.normpath(
        os.path.join(os.path.abspath(args.data_root), '..', 'plots')
    )
    output = args.output if args.output else os.path.join(plots_dir, 'deltardapp_vs_diameter.png')

    output_by_odi_dir = os.path.join(
        _nozomi_root,
        'experiment', 'visualization', 'plots', 'rdapp_by_ODI',
    )

    # ------------------------------------------------------------------
    # Generate plots for each of the 3 effective diameter metrics
    # (metric_key, entry_field, xlabel, filename_tag, long_label)
    # ------------------------------------------------------------------
    for _, diameter_field, xlabel, tag, long_label in DIAMETER_METRICS:

        # ΔRDapp vs diameter — linear fit, per-OD subplots
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

        # RDapp PGSE/OGSE vs diameter — per-OD individual substrate scatter
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
