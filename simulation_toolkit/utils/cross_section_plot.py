"""
cross_section_plot.py
---------------------
Utility for plotting a transverse (xy-plane) cross-section of an axon substrate
at the mid-point along the z-axis (z = 0).

One circle is drawn per axon per membrane component.  The circle is computed by
linearly interpolating along the sphere-chain to the exact z-plane crossing,
yielding a single clean circle instead of the set of all spheres that happen to
straddle the plane.

For myelinated substrates the outer circle (myelin boundary) is drawn as a
thin-bordered semi-transparent ring and the inner circle (axon lumen) as a
solid filled circle, making the myelin sheath visible as the annular gap.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import simulation_toolkit.toolkit_params as config_params


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float32)


def _split_chains(spheres: np.ndarray, z_extent: float):
    """Split a sphere array into contiguous fiber chains.

    A chain boundary is detected when z drops by more than z_extent/2, which
    only happens at a PBC wrap (z: +Lz/2 → -Lz/2) and never within a single
    chain (z is monotonically increasing along a fiber).  The threshold is
    keyed to the z-height Lz -- NOT the in-plane box -- so thin-z anisotropic
    substrates (Lz < Lx) still split correctly.
    All chains are returned, including PBC image chains shifted by ±L in x/y.
    """
    if spheres.shape[0] == 0:
        return []
    half = z_extent / 2.0
    chains = []
    start = 0
    for i in range(1, spheres.shape[0]):
        if spheres[i, 2] < spheres[i - 1, 2] - half:
            seg = spheres[start:i]
            if seg.shape[0] >= 2:
                chains.append(seg)
            start = i
    seg = spheres[start:]
    if seg.shape[0] >= 2:
        chains.append(seg)
    return chains


def _chain_z_crossing(chain: np.ndarray, z_plane: float = 0.0):
    """Linearly interpolate along *chain* to find the (x, y, r) at *z_plane*.

    Returns a tuple (cx, cy, r) or None if the chain does not cross z_plane.
    """
    z = chain[:, 2]
    for i in range(len(z) - 1):
        z0, z1 = float(z[i]), float(z[i + 1])
        if (z0 <= z_plane <= z1) or (z1 <= z_plane <= z0):
            dz = z1 - z0
            t = 0.5 if abs(dz) < 1e-10 else (z_plane - z0) / dz
            cx = float(chain[i, 0]) * (1 - t) + float(chain[i + 1, 0]) * t
            cy = float(chain[i, 1]) * (1 - t) + float(chain[i + 1, 1]) * t
            r  = float(chain[i, 3]) * (1 - t) + float(chain[i + 1, 3]) * t
            return cx, cy, r
    return None


def _compute_crossings(spheres: np.ndarray, z_extent: float, z_plane: float):
    """Return a list of (cx, cy, r, fiber_id) — one entry per chain crossing."""
    chains = _split_chains(spheres, z_extent)
    results = []
    for chain in chains:
        pt = _chain_z_crossing(chain, z_plane)
        if pt is not None:
            fid = int(chain[0, 4])
            results.append((pt[0], pt[1], pt[2], fid))
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_cross_section_z_mid(
    outer_fibers,
    inner_fibers=None,
    box_length: float | None = None,
    box_length_z: float | None = None,
    component_label: str | None = None,
    color=None,
    z_plane: float = 0.0,
):
    """Plot the transverse cross-section of a substrate at *z_plane*.

    Exactly one circle is drawn per axon for each membrane component.
    For myelinated substrates the annular gap between the outer and inner
    circles represents the myelin sheath.

    Parameters
    ----------
    outer_fibers:
        Sphere array (N, 5+) – outer membrane (or unmyelinated axon).
    inner_fibers:
        Optional sphere array (M, 5+) – inner axon membrane (myelinated case).
    box_length:
        Box side length (µm). Falls back to ``config_params.BOX_LENGTH``.
    component_label:
        Optional suffix appended to the saved filename.
    color:
        Per-fiber RGBA colour array (indexed by ascending chain order).
        Rainbow palette is used when *None*.
    z_plane:
        Z-coordinate of the cutting plane (µm). Default 0.0 (mid-box).
    """
    outer_np = _to_numpy(outer_fibers)
    inner_np = _to_numpy(inner_fibers) if inner_fibers is not None else None

    L = float(box_length) if box_length is not None else float(config_params.BOX_LENGTH)
    Lz = float(box_length_z) if box_length_z is not None else float(config_params.BOX_LENGTH_Z)
    is_myelinated = inner_np is not None

    outer_crossings = _compute_crossings(outer_np, Lz, z_plane)
    if not outer_crossings:
        print("[cross_section_plot] No outer fibers cross z_plane; skipping plot.")
        return

    # Group inner crossings by fid so each inner chain is matched to the outer
    # chain with the same fiber_id.  For PBC image chains (multiple entries per
    # fid) we pair positionally within the fid group, preserving the same order
    # the substrate writer stored them.
    from collections import defaultdict
    inner_by_fid: dict[int, list] = defaultdict(list)
    if is_myelinated:
        for cx, cy, r, fid in _compute_crossings(inner_np, Lz, z_plane):
            inner_by_fid[fid].append((cx, cy, r))

    # ---- count unique axons for title/filename ----
    unique_fids = sorted({fid for _, _, _, fid in outer_crossings})
    n_axons = len(unique_fids)

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    fid_draw_counter: dict[int, int] = defaultdict(int)
    for cx, cy, r_outer, fid in outer_crossings:
        inner_list = inner_by_fid.get(fid, [])
        idx = fid_draw_counter[fid]
        fid_draw_counter[fid] += 1

        if is_myelinated and idx < len(inner_list):
            ix, iy, r_inner = inner_list[idx]
            # Outer (myelin boundary): solid black disk — myelin = black annular ring
            ax.add_patch(plt.Circle(
                (cx, cy), r_outer,
                facecolor=(0.0, 0.0, 0.0, 1.0),
                edgecolor="none",
                zorder=2,
                clip_on=True,
            ))
            # Inner (axon lumen): grey disk on top — intra-axonal space = grey
            ax.add_patch(plt.Circle(
                (ix, iy), r_inner,
                facecolor=(0.65, 0.65, 0.65, 1.0),
                edgecolor="none",
                zorder=3,
                clip_on=True,
            ))
        else:
            # Unmyelinated: solid black disk
            ax.add_patch(plt.Circle(
                (cx, cy), r_outer,
                facecolor=(0.0, 0.0, 0.0, 1.0),
                edgecolor="none",
                zorder=2,
                clip_on=True,
            ))

    ax.set_xlim(-L / 2, L / 2)
    ax.set_ylim(-L / 2, L / 2)
    ax.set_xlabel("x (µm)", fontsize=13)
    ax.set_ylabel("y (µm)", fontsize=13)
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#222222")

    title = f"Cross-section at z={z_plane:.1f} µm  |  {n_axons} axons"
    if is_myelinated:
        title += "  |  outer+inner membrane"
    ax.set_title(title, fontsize=13, pad=10)

    # ------------------------------------------------------------------ save
    folder_path = os.path.join(config_params.SUBSTRATE_OUTPUT_FOLDER_PATH, "figs", "visual")
    os.makedirs(folder_path, exist_ok=True)

    label_suffix = "" if component_label is None else f"_{component_label}"
    fname = (
        f"cross_section_z{z_plane:.1f}{label_suffix}"
        f"_{n_axons}axons"
        f"_{config_params.EXP_DATE_TIME}"
        f"_VF{config_params.VOLUME_FRACTION}"
        ".png"
    )
    save_path = os.path.join(folder_path, fname)
    fig.tight_layout()
    plt.savefig(save_path, dpi=500, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"[cross_section_plot] Saved → {save_path}")
