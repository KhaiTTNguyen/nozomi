import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_ROOT = (
    Path(__file__).resolve().parent
    / "plots"
    / "Kperp_across_diameter"
    / "computed_Kperp_total"
)
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parent
    / "plots"
    / "Kperp_across_diameter"
    / "compact_row_7x5"
)

COLORS = {
    1.68: "red",
    2.58: "blue",
    3.5: "green",
    4.5: "purple",
}

METRICS = {
    "extra": {
        "key": "Kperp_extra",
        "ylabel": r"$K_{\mathrm{e},\!\perp}$",
        "time_limit": 20.0,
        "ylim": (0.0, 0.8),
        "output_stem": "Kperp_across_diameter_{kappa}_extra_compact",
    },
    "intra": {
        "key": "Kperp_intra",
        "ylabel": r"$K_{\mathrm{i},\!\perp}$",
        "time_limit": 100.0,
        "ylim": (0.0, 12.0),
        "output_stem": "Kperp_across_diameter_{kappa}_intra_compact",
    },
    "total": {
        "key": "Kperp_total",
        "ylabel": r"$K_{\mathrm{total},\!\perp}$",
        "time_limit": 100.0,
        "ylim": (0.0, 6.5),
        "output_stem": "Kperp_total_across_diameter_{kappa}_compact",
    },
}


def load_grouped_runs(input_root):
    grouped = defaultdict(lambda: defaultdict(list))
    for path in sorted(input_root.glob("Kperp_total_K*_d*.pkl")):
        with path.open("rb") as file:
            run = pickle.load(file)
        grouped[int(run["kappa"])][float(run["diameter"])].append(run)
    return grouped


def average_runs(runs, metric_key):
    min_len = min(len(run["time"]) for run in runs)
    time_stack = np.asarray([run["time"][:min_len] for run in runs])
    value_stack = np.asarray([run[metric_key][:min_len] for run in runs])
    return {
        "time": np.nanmean(time_stack, axis=0),
        "mean": np.nanmean(value_stack, axis=0),
        "std": np.nanstd(value_stack, axis=0, ddof=1)
        if len(runs) > 1
        else np.zeros(min_len),
    }


def plot_metric(grouped, metric_name, output_root, figsize, dpi):
    metric = METRICS[metric_name]
    output_root.mkdir(parents=True, exist_ok=True)

    for kappa, runs_by_diameter in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=figsize)
        for diameter, runs in sorted(runs_by_diameter.items()):
            averaged = average_runs(runs, metric["key"])
            mask = averaged["time"] <= metric["time_limit"]
            time = averaged["time"][mask]
            mean = averaged["mean"][mask]
            std = averaged["std"][mask]
            color = COLORS.get(diameter)

            ax.plot(time, mean, color=color, label=f"{diameter:g} um", linewidth=3.0)
            ax.fill_between(time, mean - std, mean + std, color=color, alpha=0.25)

        ax.set_xlabel(r"$t\;(\mathrm{ms})$", fontsize=27)
        ax.set_ylabel(metric["ylabel"], fontsize=27)
        ax.set_title(f"$\\kappa$={kappa}", fontsize=27)
        ax.set_xlim([0.002, metric["time_limit"]])
        ax.set_ylim(metric["ylim"])
        # ax.legend(fontsize=18, framealpha=0.85, handlelength=2.0)
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        output_path = output_root / f"{metric['output_stem'].format(kappa=f'K{kappa}')}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create compact Kperp extra, intra, and total figures for three-per-row layouts."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--width", type=float, default=7.0)
    parser.add_argument("--height", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=500)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(METRICS),
        default=sorted(METRICS),
    )
    args = parser.parse_args()

    grouped = load_grouped_runs(args.input_root)
    if not grouped:
        raise FileNotFoundError(f"No computed Kperp_total pickle files found in {args.input_root}")

    figsize = (args.width, args.height)
    for metric_name in args.metrics:
        plot_metric(grouped, metric_name, args.output_root, figsize, args.dpi)


if __name__ == "__main__":
    main()