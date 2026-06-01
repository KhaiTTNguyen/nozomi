"""MC/DC thread-count scaling sweep.

Runs the *canonical* benchmark workload (full n_walkers / num_steps on the
100-cylinder substrate) at several ``num_process`` values to find the knee of
the speed-up curve. The selected default then goes into ``BenchConfig.n_processes``.

MC/DC walker-loop parallelism uses std::thread (NOT OpenMP), and ``num_process 0``
is coerced to single-threaded, so the count must be set explicitly.

Usage:
    cd nozomi
    PYTHONPATH=. python -m tests.benchmark.sweep_mcdc_threads
    PYTHONPATH=. python -m tests.benchmark.sweep_mcdc_threads --threads 8 16 32 64 128
"""
from __future__ import annotations

import argparse
import json

from tests.benchmark.config import CFG
from tests.benchmark.run_mcdc import ensure_dirs, run_once


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+",
                    default=[8, 16, 32, 64])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ensure_dirs()
    print(f"MC/DC scaling sweep | N={CFG.n_walkers} walkers | "
          f"T={int(round(CFG.te_s / CFG.time_step_s))} steps | "
          f"100-cylinder substrate\n")

    rows: list[dict] = []
    baseline = None
    for nt in args.threads:
        res = run_once(repeat=0, n_processes=nt)
        elapsed = res["elapsed_s"]
        if baseline is None:
            baseline = elapsed
        speedup = baseline / elapsed
        # quick signal sanity: normalized signal at first/last b-value
        sig = res["signal"]
        rows.append({
            "n_processes": nt,
            "elapsed_s": elapsed,
            "speedup_vs_first": speedup,
            "signal_first": sig[0],
            "signal_last": sig[-1],
        })
        print(f"  num_process={nt:>3} : {elapsed:8.2f} s  "
              f"(x{speedup:5.2f} vs {args.threads[0]} threads)  "
              f"S[-1]/S0={sig[-1]:.4f}")

    print("\nSummary (look for the knee where added threads stop helping):")
    print(json.dumps(rows, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
