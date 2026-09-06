#!/usr/bin/env python3
"""
add_inner_compartment.py
========================
Post-hoc myelin generation for the Set 1/2/3 study.

Some substrates in ``experiment/aim2-prep/data_full/set{1,2,3}`` were generated
WITHOUT an inner (axonal / myelin) compartment. This script adds one in place:
for every un-myelinated substrate it derives inner-membrane spheres from the
existing outer fibers (g_ratio scaling), then re-saves the same ``data/*.pkl``
as a myelinated substrate. The outer geometry is never modified.

Stale-intra handling
---------------------
For an un-myelinated substrate, a previously-run ``intra`` simulation actually
used the OUTER geometry (``simulation_main.py`` falls back to outer when
``is_myelinated`` is False). Once we add the true inner compartment that intra
result is invalid, so any ``sim/ADCdata/diffcoeff_intra_*.pkl`` (and matching
intra PNGs) and the ``sim/RDapp/rdapp_result.pkl`` are moved into
``sim/ADCdata/_pre_myelin_backup/``. The valid EXTRA (outer) result is kept, so
``batch_simulation build-scan`` will re-run intra only.

Idempotent: already-myelinated substrates are skipped and their sims untouched.

Layout expected (one or more set roots)::

    <set_root>/<substrate>/data/array*.pkl
                          /sim/ADCdata/...

Usage
-----
    cd nozomi
    sim_venv/bin/python3 experiment/aim2-prep/add_inner_compartment.py            # all three sets
    sim_venv/bin/python3 experiment/aim2-prep/add_inner_compartment.py \\
        experiment/aim2-prep/data_full/set2 --dry-run
"""
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_NOZOMI_ROOT = _HERE.parents[2]   # nozomi/
if str(_NOZOMI_ROOT) not in sys.path:
    sys.path.insert(0, str(_NOZOMI_ROOT))

from simulation_toolkit.substrate_generator.myelin import generate_inner_fibers
from simulation_toolkit.utils.common_utils import (
    load_substrate_geometry,
    save_myelinated_substrate_to_pickle,
)

_DEFAULT_SET_ROOTS = [
    str(_HERE.parent / "data_full" / "set1"),
    str(_HERE.parent / "data_full" / "set2"),
    str(_HERE.parent / "data_full" / "set3"),
]


def _iter_substrate_pkls(set_root: str):
    """Yield (substrate_dir, pkl_path) for ``<set_root>/<substrate>/data/array*.pkl``."""
    set_root = os.path.abspath(set_root)
    if not os.path.isdir(set_root):
        print(f"  !! missing set root: {set_root}", file=sys.stderr)
        return
    for substrate_id in sorted(os.listdir(set_root)):
        substrate_dir = os.path.join(set_root, substrate_id)
        data_dir = os.path.join(substrate_dir, "data")
        if not os.path.isdir(data_dir):
            continue
        pkls = sorted(glob.glob(os.path.join(data_dir, "array*.pkl")))
        if not pkls:
            continue
        yield substrate_dir, pkls[-1]


def _backup_stale_intra(substrate_dir: str, dry_run: bool):
    """Move outer-fallback intra outputs + stale RDapp into a backup folder so
    the true inner intra is re-simulated. Returns the number of items moved."""
    adc_dir = os.path.join(substrate_dir, "sim", "ADCdata")
    rdapp_dir = os.path.join(substrate_dir, "sim", "RDapp")
    backup_dir = os.path.join(adc_dir, "_pre_myelin_backup")

    stale = []
    if os.path.isdir(adc_dir):
        for fn in os.listdir(adc_dir):
            lower = fn.lower()
            if "intra" in lower and (fn.endswith(".pkl") or fn.endswith(".png")):
                stale.append(os.path.join(adc_dir, fn))
    rdapp_pkl = os.path.join(rdapp_dir, "rdapp_result.pkl")
    if os.path.isfile(rdapp_pkl):
        stale.append(rdapp_pkl)

    if not stale:
        return 0
    if dry_run:
        print(f"    [dry-run] would back up {len(stale)} stale intra/RDapp file(s)")
        return len(stale)

    os.makedirs(backup_dir, exist_ok=True)
    for src in stale:
        dst = os.path.join(backup_dir, os.path.basename(src))
        shutil.move(src, dst)
    print(f"    backed up {len(stale)} stale intra/RDapp file(s) -> {os.path.relpath(backup_dir, _NOZOMI_ROOT)}")
    return len(stale)


def process_substrate(substrate_dir, pkl_path, g_ratio, spacing_ratio,
                      dry_run=False, force=False):
    substrate = load_substrate_geometry(pkl_path)
    if substrate.is_myelinated and not force:
        return "skip_myelinated"

    if dry_run:
        print(f"    [dry-run] would add inner compartment "
              f"(g_ratio={g_ratio}, spacing_ratio={spacing_ratio})")
        _backup_stale_intra(substrate_dir, dry_run=True)
        return "would_myelinate"

    inner_fibers = generate_inner_fibers(
        substrate.outer_fibers,
        g_ratio=g_ratio,
        inner_sphere_spacing_ratio=spacing_ratio,
        box_length=substrate.box_length,
        box_length_z=substrate.lz,
    )
    save_myelinated_substrate_to_pickle(
        pkl_path,
        substrate.outer_fibers,
        inner_fibers,
        substrate.box_length,
        g_ratio,
        spacing_ratio,
        lz=substrate.lz,
        num_fibers=substrate.num_fibers,
    )
    _backup_stale_intra(substrate_dir, dry_run=False)
    return "myelinated"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set_roots", nargs="*", default=_DEFAULT_SET_ROOTS,
                    help="One or more set roots (default: data_full/set1,set2,set3).")
    ap.add_argument("--g-ratio", type=float, default=0.7,
                    help="Inner/outer radius ratio (default: 0.7).")
    ap.add_argument("--spacing-ratio", type=float, default=0.5,
                    help="inner_sphere_spacing_ratio (default: 0.5).")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate inner compartment even if already myelinated.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report actions without modifying any files.")
    args = ap.parse_args()

    totals = {"myelinated": 0, "would_myelinate": 0, "skip_myelinated": 0, "error": 0}
    for set_root in args.set_roots:
        print(f"\n{'='*70}\nSet root: {os.path.abspath(set_root)}\n{'='*70}")
        for substrate_dir, pkl_path in _iter_substrate_pkls(set_root):
            rel = os.path.relpath(substrate_dir, _NOZOMI_ROOT)
            try:
                result = process_substrate(
                    substrate_dir, pkl_path,
                    g_ratio=args.g_ratio, spacing_ratio=args.spacing_ratio,
                    dry_run=args.dry_run, force=args.force,
                )
            except Exception as exc:  # keep the queue going on isolated failures
                print(f"  [ERROR] {rel}: {exc}")
                totals["error"] += 1
                continue
            totals[result] += 1
            tag = {"myelinated": "myelinated",
                   "would_myelinate": "would-myelinate",
                   "skip_myelinated": "already-myelinated (skip)"}[result]
            print(f"  [{tag}] {rel}")

    print(f"\n{'='*70}")
    print(f"Done.  myelinated={totals['myelinated']}  "
          f"would_myelinate={totals['would_myelinate']}  "
          f"already_myelinated={totals['skip_myelinated']}  "
          f"errors={totals['error']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
