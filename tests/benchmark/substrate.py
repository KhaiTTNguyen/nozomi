"""
Build a canonical parallel-cylinder substrate and export it in formats needed
by each framework.

Geometry: N x N square-packed cylinders, radius r, separation s, axis = z.
Box:      [-L/2, L/2] x [-L/2, L/2] x [-Lz/2, Lz/2]   (L = N*s)
Periodic in xy; cylinders are built longer than Lz so z is effectively periodic
for our purposes (spins never leave the simulated duration in z).
"""

from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np

from .config import CFG, SUBSTRATE_DIR


def _payload_matches_cfg(payload: dict, cfg=CFG) -> bool:
    """Return True when cached substrate metadata matches the active config."""
    config = payload.get("config", {})
    keys = (
        "cylinder_radius_m", "cylinder_sep_m", "n_cyl_per_side", "box_z_m",
        "D0_m2_s", "delta_s", "Delta_s", "te_s", "gradient_axis",
        "bvals_s_mm2", "n_walkers", "time_step_s", "seed_mode",
        "n_repeats", "base_seed",
    )
    for key in keys:
        expected = getattr(cfg, key)
        actual = config.get(key)
        if isinstance(expected, tuple):
            expected = list(expected)
        if actual != expected:
            return False
    return True


# -------------------------------------------------------------------------
def build_cylinder_list_m(cfg=CFG):
    """Return (centers_xy [N,2], radii [N], L (m), Lz (m)) in METERS.
    Centers lie in [-L/2, L/2]^2 on a square lattice of period `cylinder_sep_m`.
    """
    N = cfg.n_cyl_per_side
    s = cfg.cylinder_sep_m
    L = N * s
    Lz = cfg.box_z_m

    # Put centers on the half-integer grid so they lie strictly inside [-L/2, L/2].
    ix = np.arange(N) - (N - 1) / 2.0
    xx, yy = np.meshgrid(ix * s, ix * s, indexing="xy")
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    radii = np.full(centers.shape[0], cfg.cylinder_radius_m, dtype=float)
    return centers, radii, L, Lz


# -------------------------------------------------------------------------
def export_json(path: Path, cfg=CFG) -> dict:
    """Human-readable dump of the substrate + config."""
    centers, radii, L, Lz = build_cylinder_list_m(cfg)
    # Exclude user-/machine-specific external-binary paths so the cache file
    # does not leak absolute filesystem paths.
    cfg_dict = {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in asdict(cfg).items()
                if k not in ("camino_bin", "mcdc_bin")}
    payload = {
        "config": cfg_dict,
        "box_x_m": L, "box_y_m": L, "box_z_m": Lz,
        "n_cylinders": int(centers.shape[0]),
        "icvf": cfg.icvf,
        "centers_m": centers.tolist(),
        "radii_m": radii.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


# -------------------------------------------------------------------------
def build_cylinder_mesh(cfg=CFG, n_segments: int = 24,
                        cylinder_length_factor: float = 2.0):
    """
    Build triangulated lateral surfaces of all cylinders.
    Returns
    -------
    vertices : (Nv, 3) float ndarray in meters
    faces    : (Nf, 3) int32 ndarray of vertex indices
    bbox_half : tuple(L/2, L/2, half_len) half-extents of the xy box and z half-length.

    Only the lateral (side) surface is meshed (no caps); cylinders extend
    ``cylinder_length_factor * Lz`` along z so walkers stay inside during TE.
    """
    centers, radii, L, Lz = build_cylinder_list_m(cfg)
    half_len = 0.5 * cylinder_length_factor * Lz

    theta = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    verts = []
    faces = []
    vidx = 0
    for (cx, cy), r in zip(centers, radii):
        ring_low = np.column_stack((cx + r * cos_t, cy + r * sin_t,
                                    np.full_like(cos_t, -half_len)))
        ring_high = np.column_stack((cx + r * cos_t, cy + r * sin_t,
                                     np.full_like(cos_t, +half_len)))
        verts.append(ring_low)
        verts.append(ring_high)
        for i in range(n_segments):
            i2 = (i + 1) % n_segments
            v0 = vidx + i
            v1 = vidx + i2
            v2 = vidx + n_segments + i2
            v3 = vidx + n_segments + i
            faces.append((v0, v1, v2))
            faces.append((v0, v2, v3))
        vidx += 2 * n_segments

    vertices = np.vstack(verts)
    faces = np.asarray(faces, dtype=np.int32)
    return vertices, faces, (L / 2.0, L / 2.0, half_len)


# -------------------------------------------------------------------------
def export_ply_mesh(path: Path, cfg=CFG, n_segments: int = 24,
                    cylinder_length_factor: float = 2.0) -> Path:
    """Write the cylinder mesh as ASCII PLY (lateral surface only)."""
    vertices, faces, _ = build_cylinder_mesh(
        cfg, n_segments=n_segments,
        cylinder_length_factor=cylinder_length_factor,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
    return path


# -------------------------------------------------------------------------
def plot_cross_section(path: Path, cfg=CFG) -> Path:
    """Write a PNG showing the xy cross section of the cylinder substrate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    centers, radii, L, Lz = build_cylinder_list_m(cfg)
    L_um = L * 1e6
    Lz_um = Lz * 1e6

    fig, ax = plt.subplots(figsize=(7, 7))
    # Periodic-image ghosts (lighter) to visualize the tiling.
    for dx in (-L, 0.0, L):
        for dy in (-L, 0.0, L):
            is_ghost = not (dx == 0.0 and dy == 0.0)
            for (cx, cy), r in zip(centers, radii):
                ax.add_patch(Circle(
                    ((cx + dx) * 1e6, (cy + dy) * 1e6), r * 1e6,
                    facecolor=("#cfd8dc" if is_ghost else "#1f77b4"),
                    edgecolor=("#90a4ae" if is_ghost else "#0b3d91"),
                    linewidth=0.5, alpha=(0.35 if is_ghost else 0.85),
                ))
    # Simulated voxel box.
    ax.add_patch(Rectangle((-L_um / 2, -L_um / 2), L_um, L_um,
                           fill=False, edgecolor="k", linewidth=1.5,
                           linestyle="--", label="simulated voxel"))

    pad = 0.15 * L_um
    ax.set_xlim(-L_um / 2 - pad, L_um / 2 + pad)
    ax.set_ylim(-L_um / 2 - pad, L_um / 2 + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.set_title(
        f"Substrate cross section (z-oriented cylinders)\n"
        f"N = {centers.shape[0]} cylinders, r = {cfg.cylinder_radius_m*1e6:.2f} μm, "
        f"sep = {cfg.cylinder_sep_m*1e6:.2f} μm, ICVF = {cfg.icvf:.3f}\n"
        f"voxel = {L_um:.1f} × {L_um:.1f} × {Lz_um:.1f} μm   "
        f"(gradient axis = {cfg.gradient_axis})"
    )
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -------------------------------------------------------------------------
def cached_substrate(cfg=CFG) -> dict:
    """Create substrate JSON + PLY + cross-section PNG once, return payload."""
    json_path = SUBSTRATE_DIR / "substrate.json"
    ply_path = SUBSTRATE_DIR / "substrate.ply"
    png_path = SUBSTRATE_DIR / "substrate_xy.png"

    refresh = True
    if json_path.exists():
        with open(json_path) as f:
            refresh = not _payload_matches_cfg(json.load(f), cfg)

    if refresh:
        export_json(json_path, cfg)
        export_ply_mesh(ply_path, cfg)
        plot_cross_section(png_path, cfg)
    else:
        if not ply_path.exists():
            export_ply_mesh(ply_path, cfg)
        if not png_path.exists():
            plot_cross_section(png_path, cfg)

    with open(json_path) as f:
        return json.load(f)


if __name__ == "__main__":
    from .config import ensure_dirs
    ensure_dirs()
    payload = cached_substrate()
    print(f"Cylinders: {payload['n_cylinders']}, ICVF = {payload['icvf']:.3f}")
    print(f"Box: {payload['box_x_m']*1e6:.2f} x {payload['box_y_m']*1e6:.2f} "
          f"x {payload['box_z_m']*1e6:.2f} um")
    print(f"Substrate cached under: {SUBSTRATE_DIR}")
    print(f"Cross-section plot: {SUBSTRATE_DIR / 'substrate_xy.png'}")
