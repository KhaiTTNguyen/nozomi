#!/usr/bin/env python3
"""
plot_studies.py
===============
Phase E of the Set 1/2/3 study: three studies of ΔRDapp vs effective axon
diameter (inner d_eff, p=3/q=2), for the hardware-consistent human (slew PGSE +
trapezoidal OGSE) and animal 15.2T (slew PGSE + apodized OGSE) protocols.

  Study 1 (Set 2, K=200) — volume-fraction effect (healthy VF70 vs pathological VF30).
  Study 2 (Set 1, bead 0.3) — orientation-dispersion effect (grouped by K/ODI).
  Study 3 (Set 3) — beading effect (grouped by bead_alpha), healthy vs mild-TBI.

Main figures use the TOTAL (voxel) signal -> plots/studies/{methods,study1_vf,
study2_dispersion,study3_beading,summary}/. Compartmental (intra/extra) versions
of each study's ΔRDapp-vs-Deff panel go to plots/studies/compartments/.

Usage:
    sim_venv/bin/python3 experiment/aim2-prep/plot_studies.py \\
        --output-dir experiment/aim2-prep/plots/studies
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_NOZOMI_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _NOZOMI_ROOT not in sys.path:
    sys.path.insert(0, _NOZOMI_ROOT)

_DEFAULT_ROOT = os.path.join(_HERE, "data_full")

# ΔRDapp keys per protocol (human = trapezoidal, animal = apodized), by component.
DELTA_KEYS = {
    "human":  {"total": "DeltaRDapp_trap", "intra": "DeltaRDapp_trap_intra", "extra": "DeltaRDapp_trap_extra"},
    "animal": {"total": "DeltaRDapp_2",    "intra": "DeltaRDapp_intra_2",    "extra": "DeltaRDapp_extra_2"},
}
# RDapp (D⊥) keys for the PGSE / OGSE sequences, by component.
RD_KEYS = {
    "human": {
        "pgse": {"total": "RDapp_PGSE", "intra": "RDapp_PGSE_intra", "extra": "RDapp_PGSE_extra"},
        "ogse": {"total": "RDapp_OGSE_trap", "intra": "RDapp_OGSE_trap_intra", "extra": "RDapp_OGSE_trap_extra"},
    },
    "animal": {
        "pgse": {"total": "RDapp_PGSE_2", "intra": "RDapp_PGSE_intra_2", "extra": "RDapp_PGSE_extra_2"},
        "ogse": {"total": "RDapp_OGSE_2", "intra": "RDapp_OGSE_intra_2", "extra": "RDapp_OGSE_extra_2"},
    },
}
PROTO_TITLE = {
    "human":  "Human (b=300 s/mm², slew PGSE + trapezoidal OGSE)",
    "animal": "Animal 15.2T (b=800 s/mm², slew PGSE + apodized OGSE)",
}
COMPONENTS = ("total", "intra", "extra")
DESIGN_DIAMS = [1.68, 2.58, 3.5, 4.5]


def _parse_folder(name):
    d = re.search(r'd([\d.]+)_', name)
    k = re.search(r'_K(\d+)_', name)
    be = re.search(r'bead([\d.]+|baseline)', name)
    avf = re.search(r'avf([\d.]+)', name)
    odi = re.search(r'ODI_([\d.]+)', name)
    arm = 'VF30' if 'VF30' in name else ('VF70' if 'VF70' in name else None)
    return {
        "d_design": float(d.group(1)) if d else None,
        "K": int(k.group(1)) if k else None,
        "bead": be.group(1) if be else None,
        "avf": float(avf.group(1)) if avf else None,
        "ODI": float(odi.group(1)) if odi else None,
        "arm": arm,
    }


def _read_inner_deff(substrate_dir):
    stats = os.path.join(substrate_dir, "figs", "substrate_stats")
    if not os.path.isdir(stats):
        return None
    cands = sorted(glob.glob(os.path.join(stats, "inner_effective_axon_diameter_stats_*.json")))
    if not cands:
        return None
    with open(cands[-1]) as f:
        return (json.load(f).get("bundle") or {}).get("d_eff_p3_q2_um")


def _read_rdapp(substrate_dir):
    pkl = os.path.join(substrate_dir, "sim", "RDapp", "rdapp_result.pkl")
    if not os.path.isfile(pkl):
        return None
    with open(pkl, "rb") as f:
        return pickle.load(f)


def collect(root):
    """Return list of per-substrate records with folder params, Deff, and ΔRDapp."""
    records = []
    for setname in ("set1", "set2", "set3"):
        for sub in sorted(glob.glob(os.path.join(root, setname, "*"))):
            if not os.path.isdir(os.path.join(sub, "data")):
                continue
            deff = _read_inner_deff(sub)
            rd = _read_rdapp(sub)
            if deff is None or rd is None:
                continue
            rec = _parse_folder(os.path.basename(sub))
            rec["set"] = setname
            rec["deff"] = float(deff)
            ok = True
            for proto in ("human", "animal"):
                for comp in COMPONENTS:
                    dv = rd.get(DELTA_KEYS[proto][comp])
                    if dv is None:
                        ok = False
                    rec[f"{proto}_delta_{comp}"] = None if dv is None else float(dv)
                    rec[f"{proto}_{comp}"] = rec[f"{proto}_delta_{comp}"]
                    for seq in ("pgse", "ogse"):
                        rv = rd.get(RD_KEYS[proto][seq][comp])
                        rec[f"{proto}_{seq}_{comp}"] = None if rv is None else float(rv)
            if ok:
                records.append(rec)
    return records


def _fit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return None
    a, b = np.polyfit(x[m], y[m], 1)
    r = np.corrcoef(x[m], y[m])[0, 1]
    return a, b, r, m.sum()


def _scatter_by_group(ax, recs, group_key, ycol, cmap_name="viridis", group_order=None,
                      group_label=lambda g: str(g), fit=True):
    groups = defaultdict(list)
    for r in recs:
        g = group_key(r)
        if g is None:
            continue
        groups[g].append(r)
    keys = group_order if group_order is not None else sorted(groups)
    keys = [g for g in keys if g in groups]
    colors = plt.get_cmap(cmap_name)(np.linspace(0.15, 0.9, max(len(keys), 1)))
    for g, c in zip(keys, colors):
        rs = groups[g]
        x = [r["deff"] for r in rs]
        y = [r[ycol] for r in rs]
        ax.scatter(x, y, color=c, s=26, alpha=0.85, edgecolors="none", label=group_label(g))
    if fit:
        allx = [r["deff"] for r in recs]
        ally = [r[ycol] for r in recs]
        f = _fit(allx, ally)
        if f:
            a, b, r_, n = f
            xs = np.linspace(min(allx), max(allx), 50)
            ax.plot(xs, a * xs + b, "k--", lw=1.2, alpha=0.7,
                    label=f"fit α={a:.3f}, r={r_:.2f}")
    ax.set_xlabel("Effective axon diameter $D_{eff}$ (p=3,q=2) [µm]")
    ax.set_ylabel("$\\Delta RD_{app}$ (OGSE − PGSE) [µm²/ms]")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(fontsize=7)


def _savefig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- Fig 1 -------
def fig1_methods(records, out_dir):
    written = []
    # (A/B) designed waveforms
    try:
        from simulation_toolkit.simulation_engine.waveform_design import design_waveform
        from simulation_toolkit.simulation_engine.helper.compute_rdapp_from_narrow_pulse import (
            _PGSE_SLEW_DESIGN, _OGSE_TRAP_DESIGN, _PGSE_SLEW_DESIGN_2, _OGSE_APOD_DESIGN_2,
        )
        specs = [
            ("Human slew PGSE (b=300)", _PGSE_SLEW_DESIGN),
            ("Human trapezoidal OGSE (b=300)", _OGSE_TRAP_DESIGN),
            ("Animal slew PGSE (b=800)", _PGSE_SLEW_DESIGN_2),
            ("Animal apodized OGSE (b=800)", _OGSE_APOD_DESIGN_2),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
        for ax, (title, cfg) in zip(axes.flatten(), specs):
            res = design_waveform(**cfg)
            w = res.waveform
            g = float(res.gmax_mT_per_m)
            ax.plot(w.t, w.wave * g, lw=1.5)
            ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5)
            ax.set_title(f"{title} | Gmax={g:.1f} mT/m", fontsize=10)
            ax.set_xlabel("Time (ms)"); ax.set_ylabel("G (mT/m)")
            ax.grid(True, ls="--", alpha=0.3)
        fig.suptitle("Fig 1A/B — Hardware-consistent gradient waveforms", fontsize=12)
        fig.tight_layout()
        written.append(_savefig(fig, out_dir, "fig1AB_waveforms.png"))
    except Exception as exc:
        print(f"  [warn] waveform panel skipped: {exc}")

    # (C) master ΔRDapp vs Deff pooled, human & animal
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, proto in zip(axes, ("human", "animal")):
        x = [r["deff"] for r in records]
        y = [r[f"{proto}_total"] for r in records]
        ax.scatter(x, y, s=18, alpha=0.5, color="#0072B2" if proto == "human" else "#D55E00")
        f = _fit(x, y)
        if f:
            a, b, r_, n = f
            xs = np.linspace(min(x), max(x), 50)
            ax.plot(xs, a * xs + b, "k--", lw=1.3, label=f"α={a:.3f}, r={r_:.2f}, n={n}")
            ax.legend(fontsize=9)
        ax.set_title(PROTO_TITLE[proto], fontsize=10)
        ax.set_xlabel("$D_{eff}$ (p=3,q=2) [µm]")
        ax.set_ylabel("$\\Delta RD_{app}$ [µm²/ms]")
        ax.grid(True, ls="--", alpha=0.3)
    fig.suptitle("Fig 1C — Master ΔRDapp vs Deff (all substrates)", fontsize=12)
    fig.tight_layout()
    written.append(_savefig(fig, out_dir, "fig1C_master_deltardapp_vs_deff.png"))
    return written


# ---------------------------------------------------------------- Fig 2 -------
def _vf_group(r):
    if r["avf"] is None:
        return None
    return "Healthy (VF 0.55–0.77)" if r["avf"] >= 0.5 else "Pathological (VF 0.29–0.34)"


def fig2_study1(records, out_dir, comp="total", subdir=None):
    recs = [r for r in records if r["set"] == "set2" and r["K"] == 200]
    if not recs:
        return []
    outd = out_dir if subdir is None else os.path.join(out_dir, subdir)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, proto in zip(axes, ("human", "animal")):
        _scatter_by_group(ax, recs, _vf_group, f"{proto}_{comp}", cmap_name="coolwarm",
                          group_order=["Healthy (VF 0.55–0.77)", "Pathological (VF 0.29–0.34)"])
        ax.set_title(PROTO_TITLE[proto], fontsize=10)
    fig.suptitle(f"Fig 2 — Study 1: volume-fraction effect ({comp})  [Set 2, K=200]", fontsize=12)
    fig.tight_layout()
    tag = "" if comp == "total" else f"_{comp}"
    written = [_savefig(fig, outd, f"fig2_study1_vf_vs_deff{tag}.png")]
    if comp == "total":
        # (C) ΔRDapp vs VF at matched design diameter
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, proto in zip(axes, ("human", "animal")):
            colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(DESIGN_DIAMS)))
            for dd, c in zip(DESIGN_DIAMS, colors):
                rs = [r for r in recs if abs((r["d_design"] or -9) - dd) < 0.05]
                ax.scatter([r["avf"] for r in rs], [r[f"{proto}_total"] for r in rs],
                           color=c, s=28, label=f"d={dd} µm")
            ax.set_xlabel("Axon volume fraction (avf)")
            ax.set_ylabel("$\\Delta RD_{app}$ [µm²/ms]")
            ax.set_title(PROTO_TITLE[proto], fontsize=10)
            ax.grid(True, ls="--", alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle("Fig 2C — Study 1: ΔRDapp vs volume fraction at matched diameter", fontsize=12)
        fig.tight_layout()
        written.append(_savefig(fig, outd, "fig2C_study1_deltardapp_vs_vf.png"))
    return written


# ---------------------------------------------------------------- Fig 3 -------
def fig3_study2(records, out_dir, comp="total", subdir=None):
    recs = [r for r in records if r["set"] == "set1"]
    if not recs:
        return []
    outd = out_dir if subdir is None else os.path.join(out_dir, subdir)
    korder = [200, 20, 10, 7, 4, 2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, proto in zip(axes, ("human", "animal")):
        _scatter_by_group(ax, recs, lambda r: r["K"], f"{proto}_{comp}", cmap_name="plasma",
                          group_order=korder, group_label=lambda k: f"K={k}")
        ax.set_title(PROTO_TITLE[proto], fontsize=10)
    fig.suptitle(f"Fig 3 — Study 2: orientation-dispersion effect ({comp})  [Set 1, bead 0.3]", fontsize=12)
    fig.tight_layout()
    tag = "" if comp == "total" else f"_{comp}"
    written = [_savefig(fig, outd, f"fig3_study2_dispersion_vs_deff{tag}.png")]
    if comp == "total":
        # (C) ΔRDapp vs ODI at matched design diameter
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, proto in zip(axes, ("human", "animal")):
            colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(DESIGN_DIAMS)))
            for dd, c in zip(DESIGN_DIAMS, colors):
                rs = [r for r in recs if abs((r["d_design"] or -9) - dd) < 0.05 and r["ODI"] is not None]
                rs.sort(key=lambda r: r["ODI"])
                ax.plot([r["ODI"] for r in rs], [r[f"{proto}_total"] for r in rs],
                        "o-", color=c, ms=5, lw=1, label=f"d={dd} µm")
            ax.set_xlabel("ODI")
            ax.set_ylabel("$\\Delta RD_{app}$ [µm²/ms]")
            ax.set_title(PROTO_TITLE[proto], fontsize=10)
            ax.grid(True, ls="--", alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle("Fig 3C — Study 2: ΔRDapp vs ODI at matched diameter", fontsize=12)
        fig.tight_layout()
        written.append(_savefig(fig, outd, "fig3C_study2_deltardapp_vs_odi.png"))
    return written


# ---------------------------------------------------------------- Fig 4 -------
def fig4_study3(records, out_dir, comp="total", subdir=None):
    recs = [r for r in records if r["set"] == "set3" and r["bead"] not in (None, "baseline")]
    if not recs:
        return []
    outd = out_dir if subdir is None else os.path.join(out_dir, subdir)
    beads = ["0.3", "0.5", "0.83", "1.24"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, proto in zip(axes, ("human", "animal")):
        _scatter_by_group(ax, recs, lambda r: r["bead"], f"{proto}_{comp}", cmap_name="autumn",
                          group_order=beads, group_label=lambda b: f"bead {b}")
        ax.set_title(PROTO_TITLE[proto], fontsize=10)
    fig.suptitle(f"Fig 4 — Study 3: beading effect ({comp})  [Set 3]", fontsize=12)
    fig.tight_layout()
    tag = "" if comp == "total" else f"_{comp}"
    written = [_savefig(fig, outd, f"fig4_study3_beading_vs_deff{tag}.png")]
    if comp != "total":
        return written
    # (C) ΔRDapp vs DESIGNED mean-d, colored by beading (isolates beading)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, proto in zip(axes, ("human", "animal")):
        colors = plt.get_cmap("autumn")(np.linspace(0.05, 0.85, len(beads)))
        for b, c in zip(beads, colors):
            rs = [r for r in recs if r["bead"] == b]
            ax.scatter([r["d_design"] for r in rs], [r[f"{proto}_total"] for r in rs],
                       color=c, s=26, label=f"bead {b}")
        ax.set_xlabel("Designed mean axon diameter [µm]")
        ax.set_ylabel("$\\Delta RD_{app}$ [µm²/ms]")
        ax.set_title(PROTO_TITLE[proto], fontsize=10)
        ax.grid(True, ls="--", alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Fig 4C — Study 3: ΔRDapp vs designed mean diameter (beading isolated)", fontsize=12)
    fig.tight_layout()
    written.append(_savefig(fig, outd, "fig4C_study3_deltardapp_vs_designd.png"))
    # (D) healthy(0.3) vs TBI(0.5-1.24) boxplots at matched design diameter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, proto in zip(axes, ("human", "animal")):
        data, labels, colors = [], [], []
        for dd in DESIGN_DIAMS:
            healthy = [r[f"{proto}_total"] for r in recs if r["bead"] == "0.3" and abs((r["d_design"] or -9) - dd) < 0.05]
            tbi = [r[f"{proto}_total"] for r in recs if r["bead"] in ("0.5", "0.83", "1.24") and abs((r["d_design"] or -9) - dd) < 0.05]
            if healthy:
                data.append(healthy); labels.append(f"d{dd}\nhealthy"); colors.append("#2c7fb8")
            if tbi:
                data.append(tbi); labels.append(f"d{dd}\nTBI"); colors.append("#d95f0e")
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_ylabel("$\\Delta RD_{app}$ [µm²/ms]")
        ax.set_title(PROTO_TITLE[proto], fontsize=10)
        ax.grid(True, axis="y", ls="--", alpha=0.3)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Fig 4D — Study 3: healthy (bead 0.3) vs mild-TBI (bead 0.5–1.24) at matched diameter", fontsize=12)
    fig.tight_layout()
    written.append(_savefig(fig, outd, "fig4D_study3_healthy_vs_tbi_boxplots.png"))
    return written


# ---------------------------------------------------------------- Fig 5 -------
def fig5_summary(records, out_dir):
    written = []
    groups = []  # (label, recs)
    s2 = [r for r in records if r["set"] == "set2" and r["K"] == 200]
    groups.append(("S1 healthy VF70", [r for r in s2 if r["avf"] and r["avf"] >= 0.5]))
    groups.append(("S1 patho VF30", [r for r in s2 if r["avf"] and r["avf"] < 0.5]))
    s1 = [r for r in records if r["set"] == "set1"]
    for k in (200, 20, 10, 7, 4, 2):
        groups.append((f"S2 K{k}", [r for r in s1 if r["K"] == k]))
    s3 = [r for r in records if r["set"] == "set3" and r["bead"] not in (None, "baseline")]
    for b in ("0.3", "0.5", "0.83", "1.24"):
        groups.append((f"S3 bead{b}", [r for r in s3 if r["bead"] == b]))

    # (A) slope of ΔRDapp vs Deff per group, human & animal
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [g[0] for g in groups]
    ypos = np.arange(len(labels))
    for proto, off, col in (("human", -0.2, "#0072B2"), ("animal", 0.2, "#D55E00")):
        slopes = []
        for _, rs in groups:
            f = _fit([r["deff"] for r in rs], [r[f"{proto}_total"] for r in rs]) if len(rs) >= 2 else None
            slopes.append(f[0] if f else np.nan)
        ax.barh(ypos + off, slopes, height=0.38, color=col, label=proto)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("ΔRDapp–Deff slope α [ (µm²/ms) / µm ]")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Fig 5A — ΔRDapp–Deff slope per study group (robustness vs sensitivity)", fontsize=11)
    ax.grid(True, axis="x", ls="--", alpha=0.3); ax.legend()
    fig.tight_layout()
    written.append(_savefig(fig, out_dir, "fig5A_slope_per_group.png"))

    # (B) human vs animal ΔRDapp correlation
    fig, ax = plt.subplots(figsize=(6.5, 6))
    x = [r["human_total"] for r in records]
    y = [r["animal_total"] for r in records]
    ax.scatter(x, y, s=16, alpha=0.5, color="#555")
    f = _fit(x, y)
    if f:
        a, b, r_, n = f
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, a * xs + b, "r--", lw=1.3, label=f"slope={a:.2f}, r={r_:.2f}, n={n}")
        ax.legend(fontsize=9)
    ax.set_xlabel("Human ΔRDapp [µm²/ms]")
    ax.set_ylabel("Animal ΔRDapp [µm²/ms]")
    ax.set_title("Fig 5B — Human vs animal ΔRDapp", fontsize=11)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    written.append(_savefig(fig, out_dir, "fig5B_human_vs_animal.png"))
    return written


# ============================ standardized a/b/c/d ============================
# Fixed, shared scales so human & animal panels (and studies) are comparable.
_XLABEL = "Effective axon diameter $D_{eff}$ (inner, p=3,q=2) [µm]"

STUDIES = [
    {
        "key": "study1_vf",
        "title": "Study 1 — volume fraction (Set 2, K=200)",
        "filter": lambda r: r["set"] == "set2" and r["K"] == 200,
        "group": _vf_group,
        "order": ["Healthy (VF 0.55–0.77)", "Pathological (VF 0.29–0.34)"],
        "label": lambda g: g,
        "cmap": "coolwarm",
        "xlim": (1.0, 4.0),
        "rd_ylim": (0.0, 2.0),
        "delta_ylim": (0.0, 0.3),
    },
    {
        "key": "study2_dispersion",
        "title": "Study 2 — orientation dispersion (Set 1, bead 0.3)",
        "filter": lambda r: r["set"] == "set1",
        "group": lambda r: r["K"],
        "order": [200, 20, 10, 7, 4, 2],
        "label": lambda k: f"K={k}",
        "cmap": "plasma",
        "xlim": (0.0, 7.0),
        "rd_ylim": (0.0, 2.0),
        "delta_ylim": (0.0, 1.0),
    },
    {
        "key": "study3_beading",
        "title": "Study 3 — beading (Set 3)",
        "filter": lambda r: r["set"] == "set3" and r["bead"] not in (None, "baseline"),
        "group": lambda r: r["bead"],
        "order": ["0.3", "0.5", "0.83", "1.24"],
        "label": lambda b: f"bead {b}",
        "cmap": "autumn",
        "xlim": (0.0, 14.0),
        "rd_ylim": (0.0, 2.0),
        "delta_ylim": (0.0, 1.0),
    },
]


def _grouped(recs, study):
    groupfn = study["group"]
    labelfn = study.get("label", lambda g: str(g))
    buckets = defaultdict(list)
    for r in recs:
        g = groupfn(r)
        if g is None:
            continue
        buckets[g].append(r)
    keys = [g for g in (study.get("order") or sorted(buckets)) if g in buckets]
    colors = plt.get_cmap(study["cmap"])(np.linspace(0.15, 0.9, max(len(keys), 1)))
    return [(labelfn(g), c, buckets[g]) for g, c in zip(keys, colors)]


def plot_a_compartmental_rd(study, recs, out_dir):
    written = []
    for comp in ("intra", "extra"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, proto in zip(axes, ("human", "animal")):
            for label, c, rs in _grouped(recs, study):
                x = [r["deff"] for r in rs]
                ax.scatter(x, [r[f"{proto}_pgse_{comp}"] for r in rs],
                           facecolors="none", edgecolors=c, s=32, label=f"{label} PGSE")
                ax.scatter(x, [r[f"{proto}_ogse_{comp}"] for r in rs],
                           color=c, s=28, label=f"{label} OGSE")
            ax.set_ylim(*study["rd_ylim"]); ax.set_xlim(*study["xlim"])
            ax.grid(True, ls="--", alpha=0.3)
            ax.set_xlabel(_XLABEL); ax.set_ylabel(f"$D_\\perp$ {comp} [µm²/ms]")
            ax.set_title(PROTO_TITLE[proto], fontsize=9)
        axes[0].legend(fontsize=6, title="hollow=PGSE, filled=OGSE")
        fig.suptitle(f"{study['title']} — (a) {comp} $D_\\perp$ (PGSE & OGSE)", fontsize=12)
        fig.tight_layout()
        written.append(_savefig(fig, out_dir, f"a_compartmental_RD_{comp}.png"))
    return written


def plot_b_compartmental_delta(study, recs, out_dir):
    written = []
    for comp in ("intra", "extra"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, proto in zip(axes, ("human", "animal")):
            for label, c, rs in _grouped(recs, study):
                x = [r["deff"] for r in rs]
                ax.scatter(x, [r[f"{proto}_delta_{comp}"] for r in rs],
                           color=c, s=30, label=label)
            ax.axhline(0, color="k", lw=0.7, alpha=0.5)
            ax.set_ylim(*study["delta_ylim"]); ax.set_xlim(*study["xlim"])
            ax.set_xlabel(_XLABEL); ax.set_ylabel(f"$\\Delta RD_{{app}}$ {comp} [µm²/ms]")
            ax.set_title(PROTO_TITLE[proto], fontsize=9)
            ax.grid(True, ls="--", alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{study['title']} — (b) {comp} $\\Delta RD_{{app}}$", fontsize=12)
        fig.tight_layout()
        written.append(_savefig(fig, out_dir, f"b_compartmental_dRDapp_{comp}.png"))
    return written


def plot_c_total_rd(study, recs, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, proto in zip(axes, ("human", "animal")):
        for label, c, rs in _grouped(recs, study):
            x = [r["deff"] for r in rs]
            ax.scatter(x, [r[f"{proto}_pgse_total"] for r in rs],
                       facecolors="none", edgecolors=c, s=32, label=f"{label} PGSE")
            ax.scatter(x, [r[f"{proto}_ogse_total"] for r in rs],
                       color=c, s=28, label=f"{label} OGSE")
        ax.set_ylim(*study["rd_ylim"]); ax.set_xlim(*study["xlim"])
        ax.set_xlabel(_XLABEL); ax.set_ylabel("$D_\\perp$ total [µm²/ms]")
        ax.set_title(PROTO_TITLE[proto], fontsize=9)
        ax.grid(True, ls="--", alpha=0.3)
    axes[0].legend(fontsize=6, title="hollow=PGSE, filled=OGSE")
    fig.suptitle(f"{study['title']} — (c) total $D_\\perp$ (PGSE & OGSE)", fontsize=12)
    fig.tight_layout()
    return [_savefig(fig, out_dir, "c_total_RD.png")]


def plot_d_total_delta(study, recs, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, proto in zip(axes, ("human", "animal")):
        for label, c, rs in _grouped(recs, study):
            x = [r["deff"] for r in rs]
            ax.scatter(x, [r[f"{proto}_delta_total"] for r in rs], color=c, s=30, label=label)
        ax.set_ylim(*study["delta_ylim"]); ax.set_xlim(*study["xlim"])
        ax.set_xlabel(_XLABEL); ax.set_ylabel("$\\Delta RD_{app}$ total [µm²/ms]")
        ax.set_title(PROTO_TITLE[proto], fontsize=9)
        ax.grid(True, ls="--", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{study['title']} — (d) total $\\Delta RD_{{app}}$", fontsize=12)
    fig.tight_layout()
    return [_savefig(fig, out_dir, "d_total_dRDapp.png")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=_DEFAULT_ROOT, help="data_full root")
    ap.add_argument("--output-dir", "-o", default=os.path.join(_HERE, "plots", "studies"))
    args = ap.parse_args()

    out = os.path.abspath(args.output_dir)
    print(f"Collecting from {args.root} ...")
    records = collect(args.root)
    print(f"  {len(records)} substrates with Deff + RDapp")
    if not records:
        print("nothing to plot."); return

    written = []
    written += fig1_methods(records, os.path.join(out, "methods"))
    for study in STUDIES:
        recs = [r for r in records if study["filter"](r)]
        if not recs:
            print(f"  [warn] no records for {study['key']}"); continue
        sdir = os.path.join(out, study["key"])
        written += plot_a_compartmental_rd(study, recs, sdir)
        written += plot_b_compartmental_delta(study, recs, sdir)
        written += plot_c_total_rd(study, recs, sdir)
        written += plot_d_total_delta(study, recs, sdir)

    print(f"\nWrote {len(written)} figure(s):")
    for p in written:
        print(f"  {os.path.relpath(p, _NOZOMI_ROOT)}")


if __name__ == "__main__":
    main()
