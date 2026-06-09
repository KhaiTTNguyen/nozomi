"""
MOCK scalability figures (synthetic data) — for choosing a presentation style.

Generates realistic-looking but FAKE numbers so we can compare plot layouts
before the real sweep is run. Nothing here touches the GPU or substrate_main.

Produces three candidate layouts into tests/scalability/results/mock/:
  1. mock_dualaxis.png   — single panel, time + GPU mem on twin y-axes
  2. mock_stacked.png    — 2-panel stacked (time / GPU mem) vs L, shared x
  3. mock_table.png      — the supporting per-size table as an image

Run:
    python tests/scalability/mock_plots.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results" / "mock"
OUT.mkdir(parents=True, exist_ok=True)

# --- Fixed substrate (mirrors d168-K200-beading0.3, the 4.2 reference) --------
MEAN_D_UM = 1.68
ICVF = 0.57
REF_L_UM = 69.0          # manuscript Section 4.2 reference cube
REF_N_FIBERS = 500
GPU_TOTAL_GB = 24.0      # RTX A5000
BUDGET_FRAC = 0.80       # leave >=20% free on the shared card
BUDGET_GB = GPU_TOTAL_GB * BUDGET_FRAC

# --- MOCK sweep ---------------------------------------------------------------
# num_fibers swept; box side L ~ sqrt(N) at fixed ICVF & diameter.
N_FIBERS = np.array([100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000])
# Calibrate L so that N=500 -> ~69 um, then L ∝ sqrt(N).
L_UM = REF_L_UM * np.sqrt(N_FIBERS / REF_N_FIBERS)

# Spheres ~ proportional to N_fibers (fixed length / spacing per fiber).
N_SPHERES = (N_FIBERS * 52).astype(int)

# Generation time: dominated by O(L^2)=O(N) collision/opt work, with a mild
# super-linear factor from LBFGS line searches at high density -> ~L^2.1.
rng = np.random.default_rng(0)
time_s = 9.0 * (L_UM / 10.0) ** 2.1
time_s *= rng.normal(1.0, 0.04, size=time_s.shape)   # small per-run jitter

# Peak GPU memory: torch tensors scale ~linearly with N_spheres (≈ L^2),
# plus a fixed CUDA context overhead (~0.6 GB). Coefficient calibrated so the
# top of the sweep approaches the 80% budget (makes the feasibility line bite).
gpu_gb = 0.6 + 1.20e-5 * N_SPHERES
gpu_gb *= rng.normal(1.0, 0.02, size=gpu_gb.shape)

# Peak host RSS: numpy buffers + python, also ~linear in N_spheres but smaller.
host_gb = 1.1 + 3.0e-6 * N_SPHERES

# Output pickle size (MB): ~20 bytes/sphere.
out_mb = N_SPHERES * 20 / 1e6

# Which sizes are feasible under the 80% budget?
feasible = gpu_gb <= BUDGET_GB
last_feasible_idx = np.where(feasible)[0].max()

# --- Fit scaling exponents (log-log slope) ------------------------------------
def _fit_exponent(x, y):
    b, a = np.polyfit(np.log(x), np.log(y), 1)   # slope, intercept
    return b, np.exp(a)

p_time, k_time = _fit_exponent(L_UM, time_s)
p_gpu, k_gpu = _fit_exponent(L_UM, gpu_gb)

# Extrapolate: largest L whose fitted GPU mem == budget.
L_ceiling = (BUDGET_GB / k_gpu) ** (1.0 / p_gpu)
L_full_card = (GPU_TOTAL_GB / k_gpu) ** (1.0 / p_gpu)


def _annotate_ref(ax, y, txt_y_frac=0.05):
    ax.axvline(REF_L_UM, color="0.5", ls=":", lw=1)
    ax.annotate(f"§4.2 ref\n{REF_L_UM:.0f} μm (N={REF_N_FIBERS})",
                xy=(REF_L_UM, y), xytext=(REF_L_UM * 1.05, y),
                fontsize=8, color="0.3", va="center")


# ============================================================ layout 1: dual axis
def plot_dualaxis():
    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(L_UM, time_s, "o-", color="#1f77b4",
                  label="gen time")
    l2 = ax2.plot(L_UM, gpu_gb, "s--", color="#d62728",
                  label="peak GPU mem")

    # budget + ceiling
    ax2.axhline(BUDGET_GB, color="#d62728", ls=":", lw=1, alpha=0.7)
    ax2.text(L_UM[0], BUDGET_GB * 1.02,
             f"80% budget = {BUDGET_GB:.1f} GB", color="#d62728", fontsize=8)
    ax1.axvline(L_ceiling, color="green", ls="-.", lw=1.2)
    ax1.text(L_ceiling * 0.7, time_s.max(),
             f"feasible ≤ {L_ceiling:.0f} μm", color="green",
             fontsize=9, rotation=90, va="top")

    ax1.set_xscale("log"); ax1.set_yscale("log"); ax2.set_yscale("log")
    ax1.set_xlabel("substrate side length L (μm)")
    ax1.set_ylabel("generation wall-clock time (s)", color="#1f77b4")
    ax2.set_ylabel("peak GPU memory (GB)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("MOCK — substrate generation scalability (dual-axis)")
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=8)
    fig.tight_layout()
    p = OUT / "mock_dualaxis.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ========================================================= layout 2: stacked 2-panel
def plot_stacked():
    fig, (axt, axm) = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)

    # --- time panel ---
    axt.plot(L_UM[feasible], time_s[feasible], "o-", color="#1f77b4",
             label="gen time (within budget)")
    axt.plot(L_UM[~feasible], time_s[~feasible], "o", mfc="none",
             color="#1f77b4", alpha=0.5, label="exceeds budget")
    axt.set_yscale("log"); axt.set_xscale("log")
    axt.set_ylabel("generation time (s)")
    axt.set_title("MOCK — substrate generation scalability (stacked)")
    axt.grid(True, which="both", ls="--", alpha=0.3)
    axt.legend(fontsize=8, loc="upper left")

    # --- memory panel ---
    axm.plot(L_UM[feasible], gpu_gb[feasible], "s-", color="#d62728",
             label="peak GPU mem")
    axm.plot(L_UM[~feasible], gpu_gb[~feasible], "s", mfc="none",
             color="#d62728", alpha=0.5, label="exceeds budget")
    axm.axhline(BUDGET_GB, color="green", ls=":", lw=1.2,
                label=f"80% budget ({BUDGET_GB:.1f} GB)")
    axm.axhline(GPU_TOTAL_GB, color="0.5", ls=":", lw=1,
                label=f"card total ({GPU_TOTAL_GB:.0f} GB)")
    axm.axvline(L_ceiling, color="green", ls="-.", lw=1.0)
    axm.text(L_ceiling * 1.02, gpu_gb.min(),
             f"feasible ≤ {L_ceiling:.0f} μm\n(extrap. {L_full_card:.0f} μm @ full card)",
             color="green", fontsize=8, va="bottom")
    axm.set_yscale("log"); axm.set_xscale("log")
    axm.set_xlabel("substrate side length L (μm)")
    axm.set_ylabel("peak GPU memory (GB)")
    axm.grid(True, which="both", ls="--", alpha=0.3)
    axm.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    p = OUT / "mock_stacked.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ============================================================= layout 3: table image
def plot_table():
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    cols = ["N_fibers", "L (μm)", "N_spheres", "gen time",
            "peak GPU (GB)", "peak host (GB)", "output (MB)", "status"]
    rows = []
    for i in range(len(N_FIBERS)):
        t = time_s[i]
        t_str = f"{t:.0f} s" if t < 90 else f"{t/60:.1f} min" if t < 5400 else f"{t/3600:.2f} h"
        status = "ok" if feasible[i] else "over budget"
        rows.append([f"{N_FIBERS[i]:,}", f"{L_UM[i]:.0f}", f"{N_SPHERES[i]:,}",
                     t_str, f"{gpu_gb[i]:.2f}", f"{host_gb[i]:.2f}",
                     f"{out_mb[i]:.1f}", status])
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    # highlight reference row (N=500) and shade over-budget rows
    for i in range(len(N_FIBERS)):
        if N_FIBERS[i] == REF_N_FIBERS:
            for j in range(len(cols)):
                tbl[i + 1, j].set_facecolor("#fff3cd")
        if not feasible[i]:
            for j in range(len(cols)):
                tbl[i + 1, j].set_facecolor("#f8d7da")
    ax.set_title("MOCK — per-size scalability table "
                 "(yellow = §4.2 ref, red = exceeds 80% budget)", fontsize=10)
    fig.tight_layout()
    p = OUT / "mock_table.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ============================================================= layout 4: time only
def plot_time_only():
    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.plot(L_UM, time_s, "o-", color="#1f77b4")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Substrate side length ($\mu$m)")
    ax.set_ylabel("Wall-clock time (sec)")
    ax.set_title("Substrate generation time vs Substrate size")
    ax.grid(True, which="both", ls="--", alpha=0.3)

    fig.tight_layout()
    p = OUT / "mock_time_only.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def main():
    p0 = plot_time_only()
    p1 = plot_dualaxis()
    p2 = plot_stacked()
    p3 = plot_table()
    print("MOCK scaling fits:")
    print(f"  time   ∝ L^{p_time:.2f}")
    print(f"  GPUmem ∝ L^{p_gpu:.2f}")
    print(f"  largest feasible L (80% budget) ≈ {L_ceiling:.0f} μm")
    print(f"  extrapolated L at full {GPU_TOTAL_GB:.0f} GB ≈ {L_full_card:.0f} μm")
    print("wrote:")
    for p in (p0, p1, p2, p3):
        print(f"  {p}")


if __name__ == "__main__":
    main()
