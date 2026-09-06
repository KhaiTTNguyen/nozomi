"""
reprocess_diameter_stats.py
============================
Batch-compute effective axon diameter statistics for previously generated
substrates that have not yet been processed.

Walks one or more data-root folders and for each substrate subfolder that
contains a ``data/array*.pkl`` file, calls
``save_effective_axon_diameter_stats_from_pickle``.

Output JSON files are written to
``<substrate>/figs/substrate_stats/outer_effective_axon_diameter_stats_<ts>.json``

Usage
-----
    cd nozomi
    source sim_venv/bin/activate

    # single root
    python simulation_toolkit/simulation_engine/helper/reprocess_diameter_stats.py \\
        experiment/visualization/data

    # multiple roots
    python simulation_toolkit/simulation_engine/helper/reprocess_diameter_stats.py \\
        experiment/visualization/data experiment/visualization/dataVF03

    # skip already-processed substrates (default: re-run all)
    python simulation_toolkit/simulation_engine/helper/reprocess_diameter_stats.py \\
        experiment/visualization/data --skip-existing
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


def _find_substrate_pkls(data_root: str):
    """
    Yield (substrate_dir, pkl_path) pairs.

    Expected structure::

        data_root/
            <exp_group>/
                <substrate_id>/
                    data/
                        array*.pkl
    """
    data_root = os.path.abspath(data_root)
    for exp_group in sorted(os.listdir(data_root)):
        exp_path = os.path.join(data_root, exp_group)
        if not os.path.isdir(exp_path):
            continue
        for substrate_id in sorted(os.listdir(exp_path)):
            substrate_path = os.path.join(exp_path, substrate_id)
            data_dir = os.path.join(substrate_path, 'data')
            if not os.path.isdir(data_dir):
                continue
            pkls = glob.glob(os.path.join(data_dir, 'array*.pkl'))
            if not pkls:
                continue
            # Use the most-recently-named pkl if multiple exist
            yield substrate_path, sorted(pkls)[-1]


def _already_processed(substrate_path: str) -> bool:
    """Return True if an outer_effective_axon_diameter_stats JSON already exists."""
    stats_dir = os.path.join(substrate_path, 'figs', 'substrate_stats')
    if not os.path.isdir(stats_dir):
        return False
    return any(
        fn.startswith('outer_effective_axon_diameter_stats') and fn.endswith('.json')
        for fn in os.listdir(stats_dir)
    )


def _process_one_substrate(task):
    """Worker entry: process a single substrate. Returns (status, id, info)."""
    substrate_path, pkl_path, skip_existing, slice_step_radius_fraction = task
    substrate_id = os.path.basename(substrate_path)
    if skip_existing and _already_processed(substrate_path):
        return ("skipped", substrate_id, None)
    output_folder = os.path.join(substrate_path, 'figs', 'substrate_stats')
    # Timestamp label derived from the substrate folder name (first 16 chars).
    config_params.EXP_DATE_TIME = substrate_id[:16] if len(substrate_id) >= 16 else substrate_id
    try:
        saved = save_effective_axon_diameter_stats_from_pickle(
            pkl_path,
            output_folder=output_folder,
            slice_step_radius_fraction=slice_step_radius_fraction,
        )
        return ("processed", substrate_id, list(saved.keys()))
    except Exception as exc:
        return ("failed", substrate_id, str(exc))


def process_data_root(data_root: str, skip_existing: bool = False, slice_step_radius_fraction=None,
                      jobs: int = 1, shard=None):
    tasks = [
        (substrate_path, pkl_path, skip_existing, slice_step_radius_fraction)
        for substrate_path, pkl_path in _find_substrate_pkls(data_root)
    ]
    # shard=(i, n): keep only tasks whose position satisfies idx % n == i, so
    # several instances can split disjoint substrates without overlapping work.
    if shard is not None:
        i, n = shard
        tasks = [t for idx, t in enumerate(tasks) if idx % n == i]
    processed, skipped, failed = 0, 0, 0

    def _tally(result):
        nonlocal processed, skipped, failed
        status, sid, info = result
        if status == "skipped":
            print(f"  [skip-existing] {sid}")
            skipped += 1
        elif status == "processed":
            print(f"  [done] {sid}  ({', '.join(info)})")
            processed += 1
        else:
            print(f"  [ERROR] {sid}: {info}")
            failed += 1

    if jobs and jobs > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=jobs) as pool:
            for result in pool.imap_unordered(_process_one_substrate, tasks):
                _tally(result)
    else:
        for task in tasks:
            _tally(_process_one_substrate(task))

    return processed, skipped, failed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-compute effective axon diameter statistics for all substrates "
            "in one or more data-root folders."
        )
    )
    parser.add_argument(
        'data_roots',
        nargs='+',
        help='One or more root folders, e.g. experiment/visualization/data',
    )
    parser.add_argument(
        '--slice-step-radius-fraction', '-f',
        type=float,
        default=0.5,
        help=(
            'Dynamic slice step as a fraction of each segment\'s mean sphere radius '
            '(default: 0.5, i.e. 50%% of mean radius ≈ 25%% of mean diameter).  '
            'Set to 0 to use the fixed --slice-step-um value instead.'
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
    parser.add_argument(
        '--jobs', '-j',
        type=int,
        default=1,
        help='Number of parallel worker processes (default: 1). Each substrate '
             'is independent; use a value up to available cores.',
    )
    parser.add_argument(
        '--shard',
        type=str,
        default=None,
        help='Process only a disjoint subset "I/N" (e.g. 0/3) so multiple '
             'instances can split the work without overlap.',
    )
    args = parser.parse_args()

    shard = None
    if args.shard:
        i_str, n_str = args.shard.split('/')
        shard = (int(i_str), int(n_str))
        if not (0 <= shard[0] < shard[1]):
            parser.error(f"--shard I/N requires 0 <= I < N (got {args.shard})")

    fraction = args.slice_step_radius_fraction if args.slice_step_radius_fraction > 0 else None
    total_processed = total_skipped = total_failed = 0
    for root in args.data_roots:
        root = os.path.abspath(root)
        print(f"\n{'='*70}")
        print(f"Data root: {root}")
        if fraction is not None:
            print(f"Slice step: {fraction*100:.0f}% of mean sphere radius (dynamic)")
        else:
            print(f"Slice step: {args.slice_step_um} µm (fixed)")
        print(f"Parallel workers: {args.jobs}" + (f"   shard: {args.shard}" if shard else ""))
        print(f"{'='*70}")
        p, s, f = process_data_root(root, skip_existing=args.skip_existing,
                                    slice_step_radius_fraction=fraction, jobs=args.jobs,
                                    shard=shard)
        total_processed += p
        total_skipped += s
        total_failed += f

    print(f"\n{'='*70}")
    print(f"Done.  processed={total_processed}  skipped={total_skipped}  failed={total_failed}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
