#!/usr/bin/env python
"""Re-generate the global OD / FOD plots and fit a Watson distribution on
previously generated substrates.

Usage
-----
    python -m simulation_toolkit.cli.reprocess_substrate_orientation \\
        --root ./experiment/result/2026-03-22_bead_1.24
    # or run directly:
    python simulation_toolkit/cli/reprocess_substrate_orientation.py \\
        --root ./experiment/result/2026-03-22_bead_1.24

The ROOT folder should contain one or more substrate sub-folders (each with
a ``data/*.pkl`` file produced by substrate_main). The script:

    * loads each substrate pickle,
    * sets ``config_params.BOX_LENGTH`` and ``config_params.SUBSTRATE_OUTPUT_FOLDER_PATH``
      so existing plot utilities work,
    * parses the prescribed kappa from the folder name (``_K<int>_``) when
      available so it is embedded in the plot titles / filenames,
    * calls ``plot_global_axon_OD`` -> writes the global OD histogram and the
      analytic Watson FOD glyph (both tagged with the fitted kappa / ODI).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from glob import glob
from pathlib import Path

# Make the repository importable regardless of the cwd.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent  # .../nozomi
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch  # noqa: F401  (pickles were produced with torch tensors)

import simulation_toolkit.toolkit_params as config_params
import simulation_toolkit.utils.common_utils as common_util
from simulation_toolkit.utils import orientation_plot


_K_PATTERN = re.compile(r"_K(\d+)_")
_D_PATTERN = re.compile(r"_d([\d.]+)")
_BEAD_PATTERN = re.compile(r"_bead_([\d.]+)")
_NFIB_PATTERN = re.compile(r"_(\d+)fibers")


def _parse_kappa_from_name(name: str):
    m = _K_PATTERN.search(name)
    return float(m.group(1)) if m else None


def _find_substrate_folders(root: Path):
    """Yield substrate sub-folders under ``root`` that contain a data/*.pkl."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Root folder does not exist: {root}")

    # Depth-first: any folder with a non-empty "data" dir holding .pkl files.
    for data_dir in root.rglob("data"):
        if not data_dir.is_dir():
            continue
        pkls = sorted(glob(str(data_dir / "*.pkl")))
        # Skip init2d.pkl-only folders (no optimized-fiber pickle).
        pkls = [p for p in pkls if not os.path.basename(p).startswith("init2d")]
        if not pkls:
            continue
        yield data_dir.parent, pkls[0]


def _reprocess_one(substrate_folder: Path, pkl_path: str):
    print(f"\n=== Reprocessing: {substrate_folder}")
    print(f"    pickle: {os.path.basename(pkl_path)}")

    optimized_fibers, L = common_util.load_data_pickle(pkl_path)
    # optimized_fibers may be a torch tensor; the new plot function handles both.

    # Configure module globals used by the plotting utilities.
    config_params.BOX_LENGTH = float(L)
    config_params.SUBSTRATE_OUTPUT_FOLDER_PATH = str(substrate_folder)

    k_prescribed = _parse_kappa_from_name(substrate_folder.name)
    if k_prescribed is not None:
        config_params.ORIENTATION_SHAPE_PARAM = k_prescribed

    # --- arc-length OD/FOD fit kept for reference but not emitted in the
    #     production pipeline (the global Watson glyph is the chosen output).
    # fit_arclength = orientation_plot.plot_along_axon_OD_arclength(
    #     optimized_fibers, optimized=True)

    fit_global = orientation_plot.plot_global_axon_OD(
        optimized_fibers, optimized=True)

    return {
        'substrate_folder': str(substrate_folder),
        'kappa_prescribed': k_prescribed,
        'kappa_fit_global': float(fit_global.get('kappa', np.nan)),
        'ODI_fit_global': float(fit_global.get('ODI', np.nan)),
        'n_global_samples': int(fit_global.get('n_samples', 0)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Recompute the global OD/FOD and Watson-kappa fit for "
                    "previously generated substrates.")
    parser.add_argument("--root", required=True,
                        help="Folder containing substrate sub-folders "
                             "(e.g. ./experiment/result/2026-03-22_bead_1.24).")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"Scanning: {root}")
    rows = []
    failures = []
    for substrate_folder, pkl_path in _find_substrate_folders(root):
        try:
            rows.append(_reprocess_one(substrate_folder, pkl_path))
        except Exception as exc:
            print(f"    FAILED: {exc}")
            traceback.print_exc()
            failures.append((str(substrate_folder), str(exc)))

    if rows:
        # Short console summary (no files written).
        print("\n{:<12} {:<12} {:<12} {:<10}".format(
            "K_designed", "K_global", "ODI_global", "n_vec"))
        for r in rows:
            print("{:<12} {:<12.3f} {:<12.5f} {:<10}".format(
                str(r['kappa_prescribed']),
                r['kappa_fit_global'], r['ODI_fit_global'],
                r['n_global_samples']))
    else:
        print("No substrate pickles found.")

    if failures:
        print(f"\n{len(failures)} folder(s) failed:")
        for f, err in failures:
            print(f"  - {f}: {err}")


if __name__ == "__main__":
    main()
