"""Re-plot the optimized-fibers 3D substrate figure for an existing substrate pkl
using the fixed anisotropic box, overwriting the old (isotropic-cube) figure.

Writes into the substrate's own figs/visual/ folder, reproducing the original
filename (EXP_DATE_TIME + VF parsed from the existing plot) so the stale figure
is replaced in place.

Usage:
    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=6 MPLBACKEND=Agg sim_venv/bin/python \
        experiment/aim2-prep/_replot_box_check.py <path-to-substrate.pkl>
"""
import os
import re
import sys

import simulation_toolkit.toolkit_params as config_params
import simulation_toolkit.utils.common_utils as util
import simulation_toolkit.utils.fiber_3D_plot as fiber_3D_plot
from simulation_toolkit.utils import cross_section_plot

_OPT_RE = re.compile(
    r"^optimized_(?P<label>.+)_horizontal_90_POV_\d+_fibers_"
    r"(?P<dt>.+)_VF_(?P<vf>[0-9.]+)\.png$"
)


def _find_old_plot(visual_dir):
    """Return (path, exp_date_time, vf, label) for the existing optimized plot."""
    if not os.path.isdir(visual_dir):
        return None
    for name in os.listdir(visual_dir):
        m = _OPT_RE.match(name)
        if m and m.group("dt") != "boxcheck":
            return (os.path.join(visual_dir, name),
                    m.group("dt"), float(m.group("vf")), m.group("label"))
    return None


def main(pkl_path):
    geom = util.load_substrate_geometry(pkl_path)
    run_dir = os.path.dirname(os.path.dirname(pkl_path))
    visual_dir = os.path.join(run_dir, "figs", "visual")

    old = _find_old_plot(visual_dir)
    if old is None:
        raise SystemExit(f"No existing optimized plot found under {visual_dir}")
    old_path, exp_dt, vf, label = old
    print(f"Loaded: lx={geom.lx}, ly={geom.ly}, lz={geom.lz}, "
          f"num_fibers={geom.num_fibers}")
    print(f"Replacing: {old_path}  (dt={exp_dt}, VF={vf}, label={label})")

    config_params.BOX_LENGTH = geom.lx
    config_params.BOX_LENGTH_Z = geom.lz
    config_params.SUBSTRATE_OUTPUT_FOLDER_PATH = run_dir
    config_params.EXP_DATE_TIME = exp_dt
    config_params.VOLUME_FRACTION = vf

    os.remove(old_path)
    fiber_3D_plot.plot_fibers(
        geom.outer_fibers,
        optimized=True,
        POV="horizontal_90",
        component_label=label,
    )

    # ---- cross-section at z=0 (delete stale, regenerate with thin-z split fix) ----
    for name in os.listdir(visual_dir):
        if name.startswith("cross_section_z"):
            os.remove(os.path.join(visual_dir, name))
    cross_section_plot.plot_cross_section_z_mid(
        outer_fibers=geom.outer_fibers,
        inner_fibers=None,
        box_length=geom.lx,
        box_length_z=geom.lz,
    )
    print(f"Wrote updated figure under: {visual_dir}/")


if __name__ == "__main__":
    main(sys.argv[1])
