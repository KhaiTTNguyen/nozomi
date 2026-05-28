"""
compute_rdapp_from_narrow_pulse_aim2.py
=======================================
Aim 2 variant of ``compute_rdapp_from_narrow_pulse.py``.

The Aim 2 data layout has an *extra* nesting level (a batch folder grouping
multiple parameter groups), so ``experiment/aim2-prep/data`` has the layout:

    data_root/
        <batch>/                                  e.g. 2026-03-18_bead_083_simulated
            <parameter_group>/                    e.g. bead0.83_d1.68_OD10_initVF0.29_500axons
                <substrate_replicate>/            e.g. 2026-03-13_17-50_d1.68_K10_..._500fibers
                    sim/ADCdata/*.pkl

The original script walks only two levels (``data_root/<exp_group>/<substrate>/sim/ADCdata``)
and therefore never reaches the ADCdata folders in the Aim 2 layout.

This script reuses ``compute_rdapp_for_substrate`` and ``save_rdapp_result``
from the original module, but replaces the directory walker with one that
handles the three-level Aim 2 layout. The original script remains untouched
so the Aim 1 (``experiment/visualization/data``) workflow keeps working.

Usage:
    python compute_rdapp_from_narrow_pulse_aim2.py experiment/aim2-prep/data
"""

import argparse
import os
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Make sure the nozomi package is importable regardless of cwd
# ------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_NOZOMI_ROOT = _HERE.parent.parent.parent.parent   # nozomi/
if str(_NOZOMI_ROOT) not in sys.path:
    sys.path.insert(0, str(_NOZOMI_ROOT))

from simulation_toolkit.simulation_engine.helper.compute_rdapp_from_narrow_pulse import (
    compute_rdapp_for_substrate,
    save_rdapp_result,
)


def process_data_root_aim2(data_root: str):
    """
    Walk an Aim 2 ``data`` folder and process every substrate replicate
    that contains ``sim/ADCdata/*.pkl`` files.

    Expected structure:
        data_root/
            <batch>/
                <parameter_group>/
                    <substrate_replicate>/
                        sim/ADCdata/*.pkl
    """
    data_root = os.path.abspath(data_root)
    if not os.path.isdir(data_root):
        raise NotADirectoryError(f"Not a directory: {data_root}")

    print(f"\n{'='*70}")
    print(f"Processing Aim 2 data root: {data_root}")
    print(f"{'='*70}\n")

    processed, skipped, failed = 0, 0, 0

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
                adc_dir = os.path.join(substrate_path, 'sim', 'ADCdata')

                if not os.path.isdir(adc_dir):
                    continue  # not a substrate replicate folder

                print(f"[{batch}/{param_group}]  {substrate_id}")

                try:
                    result = compute_rdapp_for_substrate(substrate_path)
                    out_path = save_rdapp_result(result, substrate_path)
                    print(f"  saved at {out_path}\n")
                    processed += 1
                except FileNotFoundError as e:
                    print(f"  [skip] {e}\n")
                    skipped += 1
                except Exception as e:
                    print(f"  [ERROR] {e}\n")
                    failed += 1

    print(f"{'='*70}")
    print(f"Done.  processed={processed}  skipped={skipped}  failed={failed}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute wide-pulse apparent RD from pre-simulated narrow-pulse "
            "ADCdata across an Aim 2 experiment data folder "
            "(3-level batch/param-group/substrate layout)."
        )
    )
    parser.add_argument(
        'data_root',
        help=(
            "Path to root folder containing batch subfolders, "
            "e.g. experiment/aim2-prep/data"
        ),
    )
    args = parser.parse_args()
    process_data_root_aim2(args.data_root)


if __name__ == '__main__':
    main()
