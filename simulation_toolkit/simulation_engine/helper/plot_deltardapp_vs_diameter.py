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
    r'experiment_d(?P<diam>[\d.]+)_.*_reproducibility_OD(?P<od>\d+)$'
)


def _parse_exp_folder(name: str):
    """Return (diameter_float, od_int) or None."""
    m = _EXP_RE.match(name)
    if m is None:
        return None
    return float(m.group('diam')), int(m.group('od'))


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
        diam, od = meta
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

            delta_human = r.get('DeltaRDapp', None)
            delta_animal = r.get('DeltaRDapp_2', None)

            if delta_human is not None:
                delta_human = float(delta_human)
                results["human_b300"][od][diam].append(delta_human)

            if delta_animal is not None:
                delta_animal = float(delta_animal)
                results["animal_b800"][od][diam].append(delta_animal)

            for component_name, scenario_map in component_key_map.items():
                for scenario_key, seq_map in scenario_map.items():
                    for seq_label, rd_key in seq_map.items():
                        value = r.get(rd_key, None)
                        if value is not None:
                            results["components"][component_name][scenario_key][seq_label][od][diam].append(float(value))

            print(
                f"  {exp_group}/{substrate_id}  d={diam} OD={od}"
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
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
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

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
    fig, axes = plt.subplots(n_od, 2, figsize=(13, 4.4 * n_od), sharex=True, sharey=True)
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

def plot_deltardapp_per_od(results: dict, output_path: Optional[str] = None):
    """
    One subplot per OD value, each showing:
      - Individual replicate scatter points
      - Mean ± std error bars per diameter
      - Linear fit  ΔRDapp = α·d + β
      - Pearson r annotation

    Parameters
    ----------
    results     : output of collect_deltardapp()
    output_path : file to save figure; if None, shows interactively
    """
    human_results = results.get("human_b300", {})
    animal_results = results.get("animal_b800", {})
    od_values = sorted(set(human_results.keys()) | set(animal_results.keys()))
    n_od = len(od_values)

    fig, axes = plt.subplots(1, n_od, figsize=(5 * n_od, 5), sharey=True)
    if n_od == 1:
        axes = [axes]

    for ax, od in zip(axes, od_values):
        style = _OD_STYLES.get(od, {'color': 'gray', 'marker': 'o',
                                     'label': f'OD{od}'})
        color = style['color']
        marker = style['marker']

        scenario_specs = [
            ("human_b300", human_results.get(od, {}), '-', 'Human b=300'),
            ("animal_b800", animal_results.get(od, {}), '--', 'Animal b=800'),
        ]

        ann_y = 0.96
        for _, diam_dict, line_style, scenario_label in scenario_specs:
            diams_sorted = sorted(diam_dict.keys())
            if not diams_sorted:
                continue

            # Flatten all replicates → used for regression & correlation
            all_x, all_y = [], []
            for d in diams_sorted:
                for v in diam_dict[d]:
                    all_x.append(d)
                    all_y.append(v)
            all_x = np.array(all_x)
            all_y = np.array(all_y)
            if len(all_x) < 2:
                continue

            slope, intercept, r_value, p_value, _ = linregress(all_x, all_y)

            means = np.array([np.mean(diam_dict[d]) for d in diams_sorted])
            stds = np.array([
                np.std(diam_dict[d], ddof=1) if len(diam_dict[d]) > 1 else 0.0
                for d in diams_sorted
            ])
            counts = [len(diam_dict[d]) for d in diams_sorted]

            ax.scatter(
                all_x, all_y,
                color=color, marker=marker,
                s=26, alpha=0.35, zorder=2,
                label='_nolegend_',
            )

            ax.errorbar(
                diams_sorted, means, yerr=stds,
                color=color, marker=marker,
                linestyle='None',
                label=f'{scenario_label}: mean ± std (n={counts[0]})',
                linewidth=1.8, markersize=7,
                capsize=5, capthick=1.5, elinewidth=1.2,
                zorder=4,
            )

            x_fit = np.linspace(
                min(diams_sorted) * 0.88,
                max(diams_sorted) * 1.06,
                200,
            )
            y_fit = slope * x_fit + intercept
            sign = '+' if intercept >= 0 else '-'
            ax.plot(
                x_fit, y_fit,
                color=color, linestyle=line_style, linewidth=1.6, zorder=3,
                label=(
                    f'{scenario_label} fit: '
                    rf'$\alpha={slope:.4f}$, '
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
        ax.set_xlabel('Mean axon diameter (µm)', fontsize=12)
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

    fig.suptitle(
        r'$\Delta RD^{app} = \alpha \cdot d + \beta$ — linear fit per OD '
        r'(human $b=300$ and animal $b=800$)',
        fontsize=13, y=1.02,
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
    output = args.output
    if output is None:
        output = os.path.join(
            os.path.abspath(args.data_root),
            '..', 'plots', 'deltardapp_vs_diameter.png'
        )
        output = os.path.normpath(output)

    # plot_deltardapp(results, output_path=output)

    # Per-OD plot with linear fit (saved alongside the grouped plot)
    output_per_od = os.path.join(
        os.path.dirname(os.path.abspath(output)),
        'deltardapp_per_od_linear_fit.png',
    )
    plot_deltardapp_per_od(results, output_path=output_per_od)

    # output_intra = os.path.join(
    #     os.path.dirname(os.path.abspath(output)),
    #     'rdapp_intra_pgse_ogse_vs_diameter.png',
    # )
    # plot_rdapp_component(results, component='intra', output_path=output_intra)

    # output_extra = os.path.join(
    #     os.path.dirname(os.path.abspath(output)),
    #     'rdapp_extra_pgse_ogse_vs_diameter.png',
    # )
    # plot_rdapp_component(results, component='extra', output_path=output_extra)

    # output_total = os.path.join(
    #     os.path.dirname(os.path.abspath(output)),
    #     'rdapp_total_pgse_ogse_vs_diameter.png',
    # )
    # plot_rdapp_component(results, component='total', output_path=output_total)

    output_by_odi_dir = os.path.join(
        _nozomi_root,
        'experiment', 'visualization', 'plots', 'rdapp_by_ODI',
    )
    plot_rdapp_total_by_od(results, output_dir=output_by_odi_dir)
    plot_rdapp_total_by_od_combined(results, output_dir=output_by_odi_dir)

    # output_stacked = os.path.join(
    #     os.path.dirname(os.path.abspath(output)),
    #     'rdapp_components_stacked_pgse_ogse_vs_diameter.png',
    # )
    # plot_rdapp_components_stacked(results, output_path=output_stacked)


if __name__ == '__main__':
    main()
