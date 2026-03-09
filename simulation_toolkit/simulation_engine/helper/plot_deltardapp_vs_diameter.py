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

    plot_deltardapp(results, output_path=output)

    # Per-OD plot with linear fit (saved alongside the grouped plot)
    output_per_od = os.path.join(
        os.path.dirname(os.path.abspath(output)),
        'deltardapp_per_od_linear_fit.png',
    )
    plot_deltardapp_per_od(results, output_path=output_per_od)


if __name__ == '__main__':
    main()
