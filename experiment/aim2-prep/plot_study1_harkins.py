#!/usr/bin/env python3
"""
plot_study1_harkins.py
======================
Study 1 (volume-fraction effect, Set 2 / K=200), plotted in the style of
Harkins et al. 2021 (NeuroImage 227:117619) Fig 4 & 5 to test whether the 2D
findings hold in realistic 3D substrates:

  P1  RD_intra / RD_extra for PGSE (hollow) and OGSE (filled) vs <d>eff.
  P2  ΔRDapp_intra and ΔRDapp_extra vs <d>eff  (Harkins Fig 5; crossover).
  P3  RD_total PGSE (hollow) and OGSE (filled) vs <d>eff  (Harkins Fig 4 mid).
  P4  ΔRDapp_total vs <d>eff  (Harkins Fig 4 bottom; VF-independence test).

Grouped by VF arm (healthy VF70 vs pathological VF30); one column per protocol
(human trapezoidal, animal apodized). ΔRDapp panels get a 2nd-order fit.
For the extra-axonal panel, ΔRDapp_extra is also shown vs the OUTER Deff
(Harkins App. A1: extra-axonal water reports the outer diameter).

Usage:
    sim_venv/bin/python3 experiment/aim2-prep/plot_study1_harkins.py
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_NOZOMI_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DEFAULT_DATA = os.path.join(_HERE, "data_full", "set2-K200")

KEYS = {
    "human": {
        "pgse": {"total": "RDapp_PGSE", "intra": "RDapp_PGSE_intra", "extra": "RDapp_PGSE_extra"},
        "ogse": {"total": "RDapp_OGSE_trap", "intra": "RDapp_OGSE_trap_intra", "extra": "RDapp_OGSE_trap_extra"},
        "delta": {"total": "DeltaRDapp_trap", "intra": "DeltaRDapp_trap_intra", "extra": "DeltaRDapp_trap_extra"},
    },
    "animal": {
        "pgse": {"total": "RDapp_PGSE_2", "intra": "RDapp_PGSE_intra_2", "extra": "RDapp_PGSE_extra_2"},
        "ogse": {"total": "RDapp_OGSE_2", "intra": "RDapp_OGSE_intra_2", "extra": "RDapp_OGSE_extra_2"},
        "delta": {"total": "DeltaRDapp_2", "intra": "DeltaRDapp_intra_2", "extra": "DeltaRDapp_extra_2"},
    },
}
PROTO_TITLE = {"human": "Human (b=300, slew PGSE + trap OGSE)",
               "animal": "Animal 15.2T (b=800, slew PGSE + apod OGSE)"}
VF_ORDER = ["Healthy (VF 65–77%)", "Pathological (VF 29–34%)"]
VF_COLOR = {"Healthy (VF 65–77%)": "#2166ac", "Pathological (VF 29–34%)": "#b2182b"}


def _read_deff(substrate_dir, component):
    stats = os.path.join(substrate_dir, "figs", "substrate_stats")
    if not os.path.isdir(stats):
        return None
    cands = sorted(glob.glob(os.path.join(stats, f"{component}_effective_axon_diameter_stats_*.json")))
    if not cands:
        return None
    with open(cands[-1]) as f:
        return (json.load(f).get("bundle") or {}).get("d_eff_p3_q2_um")


def collect(data_dir):
    recs = []
    for sub in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(os.path.join(sub, "data")):
            continue
        pkl = os.path.join(sub, "sim", "RDapp", "rdapp_result.pkl")
        if not os.path.isfile(pkl):
            continue
        with open(pkl, "rb") as f:
            rd = pickle.load(f)
        name = os.path.basename(sub)
        arm = "VF70" if "VF70" in name else ("VF30" if "VF30" in name else None)
        rec = {
            "deff_inner": _read_deff(sub, "inner"),
            "deff_outer": _read_deff(sub, "outer"),
            "vf": VF_ORDER[0] if arm == "VF70" else (VF_ORDER[1] if arm == "VF30" else None),
        }
        if rec["deff_inner"] is None or rec["vf"] is None:
            continue
        for proto in ("human", "animal"):
            for seq in ("pgse", "ogse", "delta"):
                for comp in ("total", "intra", "extra"):
                    rec[f"{proto}_{seq}_{comp}"] = rd.get(KEYS[proto][seq][comp])
        recs.append(rec)
    return recs


def _polyfit_line(ax, x, y, deg=2, color="k"):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < deg + 1:
        return
    c = np.polyfit(x[m], y[m], deg)
    xs = np.linspace(x[m].min(), x[m].max(), 100)
    r = np.corrcoef(np.polyval(c, x[m]), y[m])[0, 1]
    ax.plot(xs, np.polyval(c, xs), "--", color=color, lw=1.3, alpha=0.8,
            label=f"quad fit r={r:.2f}")


def _by_vf(recs):
    g = {v: [r for r in recs if r["vf"] == v] for v in VF_ORDER}
    return {v: rs for v, rs in g.items() if rs}


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return p


# P1: compartmental D⊥ (PGSE hollow, OGSE filled) — rows intra/extra, cols protocol
def p1_compartmental_dperp(recs, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for col, proto in enumerate(("human", "animal")):
        for row, comp in enumerate(("intra", "extra")):
            ax = axes[row, col]
            for vf, rs in _by_vf(recs).items():
                x = [r["deff_inner"] for r in rs]
                ax.scatter(x, [r[f"{proto}_pgse_{comp}"] for r in rs],
                           facecolors="none", edgecolors=VF_COLOR[vf], s=34,
                           label=f"{vf} PGSE")
                ax.scatter(x, [r[f"{proto}_ogse_{comp}"] for r in rs],
                           color=VF_COLOR[vf], s=30, marker="o",
                           label=f"{vf} OGSE")
            ax.set_ylabel(f"$D_\\perp$ {comp} [µm²/ms]")
            ax.grid(True, ls="--", alpha=0.3)
            ax.set_ylim(0, 2.5)
            ax.set_xlim(1, 4)
            ax.set_box_aspect(0.5)
            if row == 0:
                ax.set_title(PROTO_TITLE[proto], fontsize=10)
            if row == 1:
                ax.set_xlabel("$\\langle d\\rangle_{eff}$ (inner, p3q2) [µm]")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, title="hollow=PGSE, filled=OGSE")
    fig.suptitle("Study 1 · P1 — Compartmental $D_\\perp$ vs $\\langle d\\rangle_{eff}$ (Set 2, K=200)", fontsize=12)
    fig.tight_layout()
    return [_save(fig, out_dir, "P1_compartmental_Dperp_vs_deff.png")]


# P2: ΔRDapp intra & extra (Harkins Fig 5) + extra vs outer Deff
def p2_delta_compartments(recs, out_dir):
    written = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, proto in zip(axes, ("human", "animal")):
        for vf, rs in _by_vf(recs).items():
            x = [r["deff_inner"] for r in rs]
            ax.scatter(x, [r[f"{proto}_delta_intra"] for r in rs], color=VF_COLOR[vf],
                       marker="o", s=32, label=f"{vf} intra")
            ax.scatter(x, [r[f"{proto}_delta_extra"] for r in rs], color=VF_COLOR[vf],
                       marker="^", s=34, facecolors="none", label=f"{vf} extra")
        ax.axhline(0, color="k", lw=0.7, alpha=0.5)
        ax.set_xlabel("$\\langle d\\rangle_{eff}$ (inner, p3q2) [µm]")
        ax.set_ylabel("$\\Delta RD_{app}$ [µm²/ms]")
        ax.set_title(PROTO_TITLE[proto], fontsize=10)
        ax.grid(True, ls="--", alpha=0.3)
        ax.set_ylim(0, 0.3)
        ax.set_xlim(1, 4)
        ax.set_box_aspect(0.5)
        ax.legend(fontsize=7, title="circle=intra, triangle=extra")
    fig.suptitle("Study 1 · P2 — ΔRDapp intra vs extra (Harkins Fig 5 analog)", fontsize=12)
    fig.tight_layout()
    written.append(_save(fig, out_dir, "P2_deltardapp_intra_extra_vs_deff.png"))

    # extra vs OUTER Deff (Harkins App. A1: extra reports outer diameter)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, proto in zip(axes, ("human", "animal")):
        for vf, rs in _by_vf(recs).items():
            rs2 = [r for r in rs if r["deff_outer"] is not None]
            ax.scatter([r["deff_outer"] for r in rs2],
                       [r[f"{proto}_delta_extra"] for r in rs2],
                       color=VF_COLOR[vf], marker="^", s=34, label=f"{vf} extra")
        ax.set_xlabel("$\\langle d\\rangle_{eff}$ (OUTER, p3q2) [µm]")
        ax.set_ylabel("$\\Delta RD_{app,extra}$ [µm²/ms]")
        ax.set_title(PROTO_TITLE[proto], fontsize=10)
        ax.grid(True, ls="--", alpha=0.3); ax.legend(fontsize=8)
        ax.set_ylim(0, 0.3)
        ax.set_xlim(1, 4)
        ax.set_box_aspect(0.5)
    fig.suptitle("Study 1 · P2b — ΔRDapp_extra vs OUTER Deff", fontsize=12)
    fig.tight_layout()
    written.append(_save(fig, out_dir, "P2b_deltardapp_extra_vs_outer_deff.png"))
    return written


# P3: RD_total PGSE (hollow) + OGSE (filled) — Harkins Fig 4 middle
def p3_total_dperp(recs, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, proto in zip(axes, ("human", "animal")):
        for vf, rs in _by_vf(recs).items():
            x = [r["deff_inner"] for r in rs]
            ax.scatter(x, [r[f"{proto}_pgse_total"] for r in rs],
                       facecolors="none", edgecolors=VF_COLOR[vf], s=36, label=f"{vf} PGSE")
            ax.scatter(x, [r[f"{proto}_ogse_total"] for r in rs],
                       color=VF_COLOR[vf], s=32, label=f"{vf} OGSE")
        ax.set_xlabel("$\\langle d\\rangle_{eff}$ (inner, p3q2) [µm]")
        ax.set_ylabel("$D_\\perp$ total [µm²/ms]")
        ax.set_title(PROTO_TITLE[proto], fontsize=10)
        ax.grid(True, ls="--", alpha=0.3)
        ax.set_ylim(0, 2.5)
        ax.set_xlim(1, 4)
        ax.set_box_aspect(0.5)
        ax.legend(fontsize=7, title="hollow=PGSE, filled=OGSE")
    fig.suptitle("Study 1 · P3 — Total $D_\\perp$ PGSE vs OGSE (Harkins Fig 4 mid)", fontsize=12)
    fig.tight_layout()
    return [_save(fig, out_dir, "P3_total_Dperp_pgse_ogse_vs_deff.png")]


# P4: ΔRDapp_total (Harkins Fig 4 bottom) — VF-independence test
def p4_delta_total(recs, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, proto in zip(axes, ("human", "animal")):
        allx, ally = [], []
        for vf, rs in _by_vf(recs).items():
            x = [r["deff_inner"] for r in rs]
            y = [r[f"{proto}_delta_total"] for r in rs]
            ax.scatter(x, y, color=VF_COLOR[vf], s=34, label=vf)
            allx += x; ally += y
        ax.set_xlabel("$\\langle d\\rangle_{eff}$ (inner, p3q2) [µm]")
        ax.set_ylabel("$\\Delta RD_{app}$ total [µm²/ms]")
        ax.set_title(PROTO_TITLE[proto], fontsize=10)
        ax.grid(True, ls="--", alpha=0.3); ax.legend(fontsize=8)
        ax.set_ylim(0, 0.3)
        ax.set_xlim(1, 4)
        ax.set_box_aspect(0.5)
    fig.suptitle("Study 1 · P4 — Total ΔRDapp vs Deff (VF-independence; Harkins Fig 4 bottom)", fontsize=12)
    fig.tight_layout()
    return [_save(fig, out_dir, "P4_deltardapp_total_vs_deff.png")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=_DEFAULT_DATA)
    ap.add_argument("--output-dir", "-o",
                    default=os.path.join(_HERE, "plots", "studies", "study1_vf", "harkins_style"))
    args = ap.parse_args()

    recs = collect(args.data_dir)
    print(f"Study 1: {len(recs)} substrates from {os.path.relpath(args.data_dir, _NOZOMI_ROOT)}"
          f"  (healthy {sum(r['vf']==VF_ORDER[0] for r in recs)}, "
          f"pathological {sum(r['vf']==VF_ORDER[1] for r in recs)})")
    if not recs:
        print("nothing to plot."); return
    out = os.path.abspath(args.output_dir)
    written = []
    written += p1_compartmental_dperp(recs, out)
    written += p2_delta_compartments(recs, out)
    written += p3_total_dperp(recs, out)
    written += p4_delta_total(recs, out)
    print(f"\nWrote {len(written)} figure(s):")
    for p in written:
        print(f"  {os.path.relpath(p, _NOZOMI_ROOT)}")


if __name__ == "__main__":
    main()
