#!/usr/bin/env python3
"""
plot_set_study.py
=================
Phase E of the Set 1/2/3 study: plot apparent radial diffusion (Dperp) and its
OGSE-minus-PGSE contrast (DeltaRDapp) against the effective axon diameter Deff.

For every substrate under one or more set roots it reads:
  * x = inner effective axon diameter, p=3/q=2 only
        (figs/substrate_stats/inner_effective_axon_diameter_stats_*.json ->
         ["bundle"]["d_eff_p3_q2_um"]).
  * y = RDapp values from sim/RDapp/rdapp_result.pkl (written by
        compute_rdapp_from_narrow_pulse.py), for both protocols:
        Human (b300, Protocol 1) and Animal (b800, Protocol 2).

Both compartments are plotted (intra, extra) plus total, so each
microstructure's effect per compartment is visible. Per component and per
protocol two figure families are produced in --output-dir:
  * Dperp_<protocol>_<component>.png   : Dperp_PGSE and Dperp_OGSE vs Deff,
                                         on the SAME axes.
  * DeltaRDapp_<protocol>_<component>.png : (OGSE - PGSE) vs Deff.
Plus a combined DeltaRDapp_<protocol>_all_components.png overlaying components.

Usage
-----
    cd nozomi
    sim_venv/bin/python3 experiment/aim2-prep/plot_set_study.py \\
        --output-dir experiment/aim2-prep/plots/set2-study
"""
import argparse
import glob
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve()
_NOZOMI_ROOT = _HERE.parents[2]

_DEFAULT_SET_ROOTS = [
    str(_HERE.parent / "data_full" / "set1"),
    str(_HERE.parent / "data_full" / "set2"),
    str(_HERE.parent / "data_full" / "set3"),
]

# Protocol label -> (RDapp key suffix, human-readable title)
_PROTOCOLS = {
    "human": ("", "Human (b=300 s/mm\u00b2, TE=78 ms)"),
    "animal": ("_2", "Animal (b=800 s/mm\u00b2, TE=40 ms)"),
}
_COMPONENTS = ("intra", "extra", "total")
# Stable colors per set label so overlaid figures stay consistent.
_SET_COLORS = {}
_COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _set_color(set_label):
    if set_label not in _SET_COLORS:
        _SET_COLORS[set_label] = _COLOR_CYCLE[len(_SET_COLORS) % len(_COLOR_CYCLE)]
    return _SET_COLORS[set_label]


def _rdapp_key(kind, component, suffix):
    """kind in {'PGSE','OGSE','Delta'}; component in {'intra','extra','total'}."""
    if kind == "Delta":
        base = "DeltaRDapp" if component == "total" else f"DeltaRDapp_{component}"
    else:
        base = f"RDapp_{kind}" if component == "total" else f"RDapp_{kind}_{component}"
    return f"{base}{suffix}"


def _read_inner_deff(substrate_dir):
    stats_dir = os.path.join(substrate_dir, "figs", "substrate_stats")
    if not os.path.isdir(stats_dir):
        return None
    cands = sorted(glob.glob(os.path.join(stats_dir, "inner_effective_axon_diameter_stats_*.json")))
    if not cands:
        return None
    import json
    with open(cands[-1]) as f:
        stats = json.load(f)
    return (stats.get("bundle") or {}).get("d_eff_p3_q2_um")


def _read_rdapp(substrate_dir):
    pkl = os.path.join(substrate_dir, "sim", "RDapp", "rdapp_result.pkl")
    if not os.path.isfile(pkl):
        return None
    with open(pkl, "rb") as f:
        return pickle.load(f)


def _iter_substrate_dirs(set_root):
    set_root = os.path.abspath(set_root)
    if not os.path.isdir(set_root):
        print(f"  !! missing set root: {set_root}", file=sys.stderr)
        return
    for substrate_id in sorted(os.listdir(set_root)):
        substrate_dir = os.path.join(set_root, substrate_id)
        if os.path.isdir(os.path.join(substrate_dir, "data")):
            yield substrate_dir


def collect(set_roots):
    """Return list of records: {set, deff, rdapp} for substrates that have both
    an inner Deff and an rdapp_result."""
    records = []
    missing_deff = missing_rdapp = 0
    for set_root in set_roots:
        set_label = os.path.basename(os.path.abspath(set_root))
        for substrate_dir in _iter_substrate_dirs(set_root):
            deff = _read_inner_deff(substrate_dir)
            rdapp = _read_rdapp(substrate_dir)
            if deff is None:
                missing_deff += 1
                continue
            if rdapp is None:
                missing_rdapp += 1
                continue
            records.append({"set": set_label, "deff": float(deff), "rdapp": rdapp})
    print(f"  collected {len(records)} substrate(s); "
          f"missing inner Deff={missing_deff}, missing RDapp={missing_rdapp}")
    return records


def _group_by_set(records):
    by_set = defaultdict(list)
    for r in records:
        by_set[r["set"]].append(r)
    return by_set


def plot_dperp(records, protocol, component, out_dir):
    suffix, title = _PROTOCOLS[protocol]
    pgse_key = _rdapp_key("PGSE", component, suffix)
    ogse_key = _rdapp_key("OGSE", component, suffix)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for set_label, recs in sorted(_group_by_set(records).items()):
        color = _set_color(set_label)
        x = [r["deff"] for r in recs]
        y_pgse = [r["rdapp"].get(pgse_key) for r in recs]
        y_ogse = [r["rdapp"].get(ogse_key) for r in recs]
        ax.scatter(x, y_pgse, facecolors="none", edgecolors=color, s=36,
                   label=f"{set_label} PGSE")
        ax.scatter(x, y_ogse, color=color, s=30, marker="s",
                   label=f"{set_label} OGSE")

    ax.set_xlabel("Effective axon diameter $D_{eff}$ (p=3, q=2) [\u00b5m]")
    ax.set_ylabel("$D_\\perp^{app}$ [\u00b5m\u00b2/ms]")
    ax.set_title(f"{title}\nDperp PGSE vs OGSE \u2014 {component} compartment")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = os.path.join(out_dir, f"Dperp_{protocol}_{component}.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_delta(records, protocol, component, out_dir):
    suffix, title = _PROTOCOLS[protocol]
    key = _rdapp_key("Delta", component, suffix)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for set_label, recs in sorted(_group_by_set(records).items()):
        color = _set_color(set_label)
        x = [r["deff"] for r in recs]
        y = [r["rdapp"].get(key) for r in recs]
        ax.scatter(x, y, color=color, s=32, label=set_label)

    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Effective axon diameter $D_{eff}$ (p=3, q=2) [\u00b5m]")
    ax.set_ylabel("$\\Delta D_\\perp^{app}$ (OGSE \u2212 PGSE) [\u00b5m\u00b2/ms]")
    ax.set_title(f"{title}\n\u0394RDapp \u2014 {component} compartment")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(out_dir, f"DeltaRDapp_{protocol}_{component}.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_delta_all_components(records, protocol, out_dir):
    suffix, title = _PROTOCOLS[protocol]
    markers = {"intra": "o", "extra": "^", "total": "s"}

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for set_label, recs in sorted(_group_by_set(records).items()):
        color = _set_color(set_label)
        x = [r["deff"] for r in recs]
        for component in _COMPONENTS:
            key = _rdapp_key("Delta", component, suffix)
            y = [r["rdapp"].get(key) for r in recs]
            ax.scatter(x, y, color=color, s=28, marker=markers[component],
                       alpha=0.8, label=f"{set_label} {component}")

    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Effective axon diameter $D_{eff}$ (p=3, q=2) [\u00b5m]")
    ax.set_ylabel("$\\Delta D_\\perp^{app}$ (OGSE \u2212 PGSE) [\u00b5m\u00b2/ms]")
    ax.set_title(f"{title}\n\u0394RDapp \u2014 all compartments")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=7, ncol=len(_COMPONENTS))
    fig.tight_layout()
    out = os.path.join(out_dir, f"DeltaRDapp_{protocol}_all_components.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set_roots", nargs="*", default=_DEFAULT_SET_ROOTS,
                    help="One or more set roots (default: data_full/set1,set2,set3).")
    ap.add_argument("--output-dir", "-o",
                    default=str(_HERE.parent / "plots" / "set2-study"),
                    help="Directory to write figures into.")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    records = collect(args.set_roots)
    if not records:
        print("No substrates with both inner Deff and RDapp results; nothing to plot.")
        return

    written = []
    for protocol in _PROTOCOLS:
        for component in _COMPONENTS:
            written.append(plot_dperp(records, protocol, component, out_dir))
            written.append(plot_delta(records, protocol, component, out_dir))
        written.append(plot_delta_all_components(records, protocol, out_dir))

    print(f"\nWrote {len(written)} figure(s):")
    for p in written:
        print(f"  {os.path.relpath(p, _NOZOMI_ROOT)}")


if __name__ == "__main__":
    main()
