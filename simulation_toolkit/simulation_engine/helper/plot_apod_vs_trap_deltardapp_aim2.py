"""
plot_apod_vs_trap_deltardapp_aim2.py
====================================
Compare human-protocol ΔD⊥ computed with the apodized-cosine OGSE
(``DeltaRDapp``) versus the slew-limited trapezoidal-cosine OGSE
(``DeltaRDapp_trap``) across the Aim 2 dataset, both against the
area-weighted effective axon diameter d_eff (p=3, q=2).

Reads each substrate's ``sim/RDapp/rdapp_result.pkl`` (must have been produced by
the updated ``compute_rdapp_from_narrow_pulse`` that stores the trapezoidal keys)
and the effective diameter from ``figs/substrate_stats/outer_effective_axon_diameter_stats*.json``.

Usage:
    python plot_apod_vs_trap_deltardapp_aim2.py experiment/aim2-prep/data \\
        --output experiment/aim2-prep/plots/apod_vs_trap/human_apod_vs_trap.png
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation_toolkit.simulation_engine.helper.plot_deltardapp_vs_diameter import (
    _extract_effective_diameter_metrics,
)


def collect(data_root: str) -> np.ndarray:
    """Return array of rows [d_eff, DeltaRDapp_apod, DeltaRDapp_trap]."""
    rows = []
    for batch in sorted(os.listdir(data_root)):
        bp = os.path.join(data_root, batch)
        if not os.path.isdir(bp):
            continue
        for pg in sorted(os.listdir(bp)):
            pgp = os.path.join(bp, pg)
            if not os.path.isdir(pgp):
                continue
            for sub in sorted(os.listdir(pgp)):
                sp = os.path.join(pgp, sub)
                rp = os.path.join(sp, "sim", "RDapp", "rdapp_result.pkl")
                if not os.path.isfile(rp):
                    continue
                try:
                    d_eff = float(_extract_effective_diameter_metrics(sp)["d_eff_p3_q2"])
                except Exception:
                    continue
                with open(rp, "rb") as fh:
                    r = pickle.load(fh)
                if "DeltaRDapp_trap" not in r or "DeltaRDapp" not in r:
                    continue
                rows.append((d_eff, float(r["DeltaRDapp"]), float(r["DeltaRDapp_trap"])))
    return np.array(rows, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Apodized vs trapezoidal human ΔD⊥ comparison.")
    parser.add_argument("data_root")
    parser.add_argument(
        "--output",
        default="experiment/aim2-prep/plots/apod_vs_trap/human_apod_vs_trap.png",
    )
    args = parser.parse_args()

    data = collect(args.data_root)
    if data.size == 0:
        raise SystemExit(f"No rdapp_result.pkl with trapezoidal keys found under {args.data_root}")

    d_eff, apod, trap = data[:, 0], data[:, 1], data[:, 2]
    rel = (trap - apod) / apod
    print(f"n = {len(d_eff)} substrates")
    print(f"apodized  ΔD⊥ : mean={apod.mean():.4f}  range=[{apod.min():.4f}, {apod.max():.4f}] µm²/ms")
    print(f"trapezoid ΔD⊥ : mean={trap.mean():.4f}  range=[{trap.min():.4f}, {trap.max():.4f}] µm²/ms")
    print(f"relative diff (trap-apod)/apod : mean={rel.mean()*100:.2f}%  "
          f"RMS={np.sqrt(np.mean(rel**2))*100:.2f}%  max|.|={np.max(np.abs(rel))*100:.2f}%")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(d_eff, apod, s=18, alpha=0.6, label="apodized OGSE")
    ax.scatter(d_eff, trap, s=18, alpha=0.6, marker="x", label="trapezoidal OGSE")
    ax.set_xlabel(r"$\langle d \rangle_\mathrm{eff}$ (µm)")
    ax.set_ylabel(r"$\Delta D_\perp$ (µm²/ms)")
    ax.set_title("Human ΔD⊥ vs effective diameter\n(b=300, TE=78; PGSE slew baseline)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    ax = axes[1]
    ax.scatter(apod, trap, s=18, alpha=0.6, color="tab:purple")
    lim = [min(apod.min(), trap.min()), max(apod.max(), trap.max())]
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.7, label="1:1")
    ax.set_xlabel(r"$\Delta D_\perp$ apodized (µm²/ms)")
    ax.set_ylabel(r"$\Delta D_\perp$ trapezoidal (µm²/ms)")
    ax.set_title(f"Trapezoidal vs apodized (n={len(d_eff)})\n"
                 f"RMS rel. diff = {np.sqrt(np.mean(rel**2))*100:.2f}%", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(args.output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot: {args.output}")


if __name__ == "__main__":
    main()
