"""
reprocess_diameter_stats_aim2.py
================================
Aim 2 variant of ``reprocess_diameter_stats.py``.

The Aim 2 data layout (``experiment/aim2-prep/data``) has one extra nesting
level (a batch folder) compared to the Aim 1 layout. The original walker
only descends two levels and therefore finds 0 substrates in Aim 2 data:

    data_root/
        <batch>/                         e.g. 2026-03-18_bead_083_simulated
            <parameter_group>/           e.g. bead0.83_d1.68_OD10_initVF0.29_500axons
                <substrate_replicate>/   e.g. 2026-03-13_17-50_d1.68_..._500fibers
                    data/array*.pkl

This script reuses ``process_data_root``'s per-substrate processing logic by
swapping in a 3-level finder. The original script is left untouched so the
Aim 1 (visualization) workflow keeps working.

Usage
-----
    cd nozomi
    source sim_venv/bin/activate

    python simulation_toolkit/simulation_engine/helper/reprocess_diameter_stats_aim2.py \\
        experiment/aim2-prep/data

    # skip substrates that already have a JSON
    python simulation_toolkit/simulation_engine/helper/reprocess_diameter_stats_aim2.py \\
        experiment/aim2-prep/data --skip-existing
"""

import argparse
import glob
import os
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Make the nozomi package importable when run as a standalone script
# ------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_NOZOMI_ROOT = _HERE.parent.parent.parent.parent   # nozomi/
if str(_NOZOMI_ROOT) not in sys.path:
    sys.path.insert(0, str(_NOZOMI_ROOT))

import simulation_toolkit.toolkit_params as config_params
from simulation_toolkit.utils.along_fiber_plot import (
    save_effective_axon_diameter_stats_from_pickle,
)
from simulation_toolkit.simulation_engine.helper.reprocess_diameter_stats import (
    _already_processed,
)


def _find_substrate_pkls_aim2(data_root: str):
    """
    Yield (substrate_dir, pkl_path) pairs for the Aim 2 three-level layout::

        data_root/
            <batch>/
                <parameter_group>/
                    <substrate_replicate>/
                        data/array*.pkl
    """
    data_root = os.path.abspath(data_root)
    for batch in sorted(os.listdir(data_root)):
        batch_path = os.path.join(data_root, batch)
        if not os.path.isdir(batch_path):
            continue
        for param_group in sorted(os.listdir(batch_path)):
            param_group_path = os.path.join(batch_path, param_group)
            if not os.path.isdir(param_group_path):
                continue
            for substrate_id in sorted(os.listdir(param_group_path)):
                substrate_path = os.path.join(param_group_path, substrate_id)
                data_dir = os.path.join(substrate_path, 'data')
                if not os.path.isdir(data_dir):
                    continue
                pkls = glob.glob(os.path.join(data_dir, 'array*.pkl'))
                if not pkls:
                    continue
                # Use the most-recently-named pkl if multiple exist
                yield substrate_path, sorted(pkls)[-1]


def process_data_root_aim2(data_root: str, skip_existing: bool = False, slice_step_radius_fraction=None):
    processed, skipped, failed = 0, 0, 0

    for substrate_path, pkl_path in _find_substrate_pkls_aim2(data_root):
        substrate_id = os.path.basename(substrate_path)
        parent_label = os.path.basename(os.path.dirname(substrate_path))
        batch_label = os.path.basename(os.path.dirname(os.path.dirname(substrate_path)))

        if skip_existing and _already_processed(substrate_path):
            print(f"  [skip-existing] {batch_label}/{parent_label}/{substrate_id}")
            skipped += 1
            continue

        output_folder = os.path.join(substrate_path, 'figs', 'substrate_stats')
        # Derive a timestamp label from the substrate folder name (first 16 chars)
        config_params.EXP_DATE_TIME = substrate_id[:16] if len(substrate_id) >= 16 else substrate_id

        print(f"  Processing {batch_label}/{parent_label}/{substrate_id}")
        print(f"    pkl: {os.path.relpath(pkl_path, _NOZOMI_ROOT)}")
        try:
            saved = save_effective_axon_diameter_stats_from_pickle(
                pkl_path,
                output_folder=output_folder,
                slice_step_radius_fraction=slice_step_radius_fraction,
            )
            for label, path in saved.items():
                print(f"    [{label}] → {os.path.relpath(path, _NOZOMI_ROOT)}")
            processed += 1
        except Exception as exc:
            print(f"    [ERROR] {exc}")
            failed += 1

    return processed, skipped, failed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-compute effective axon diameter statistics for all substrates "
            "in one or more Aim 2 data-root folders (3-level batch/param-group/substrate layout)."
        )
    )
    parser.add_argument(
        'data_roots',
        nargs='+',
        help='One or more root folders, e.g. experiment/aim2-prep/data',
    )
    parser.add_argument(
        '--slice-step-radius-fraction', '-f',
        type=float,
        default=0.5,
        help=(
            'Dynamic slice step as a fraction of each segment\'s mean sphere radius '
            '(default: 0.5).  Set to 0 to use the fixed --slice-step-um value instead.'
        ),
    )
    parser.add_argument(
        '--slice-step-um',
        type=float,
        default=0.05,
        help='Fixed slice step in µm (only used when --slice-step-radius-fraction=0).',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip substrates that already have a JSON stats file.',
    )
    args = parser.parse_args()

    fraction = args.slice_step_radius_fraction if args.slice_step_radius_fraction > 0 else None
    total_processed = total_skipped = total_failed = 0
    for root in args.data_roots:
        root = os.path.abspath(root)
        print(f"\n{'='*70}")
        print(f"Aim 2 data root: {root}")
        if fraction is not None:
            print(f"Slice step: {fraction*100:.0f}% of mean sphere radius (dynamic)")
        else:
            print(f"Slice step: {args.slice_step_um} µm (fixed)")
        print(f"{'='*70}")
        p, s, f = process_data_root_aim2(
            root,
            skip_existing=args.skip_existing,
            slice_step_radius_fraction=fraction,
        )
        total_processed += p
        total_skipped += s
        total_failed += f

    print(f"\n{'='*70}")
    print(f"Done.  processed={total_processed}  skipped={total_skipped}  failed={total_failed}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
