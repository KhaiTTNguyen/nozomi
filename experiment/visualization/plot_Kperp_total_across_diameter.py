import argparse
import csv
import pickle
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path(__file__).resolve().parent / "data"
OUTPUT_ROOT = Path(__file__).resolve().parent / "plots" / "Kperp_across_diameter"
KTOTAL_FORMULA = (
    "Ktotal = (fi*Ki*Di^2 + fe*Ke*De^2)/Dtotal^2 "
    "+ 3*(fi*Di^2 + fe*De^2)/Dtotal^2 - 3"
)


def extract_float(pattern, text):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def extract_int(pattern, text):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def load_array(path):
    with path.open("rb") as file:
        data = pickle.load(file)
    return np.asarray(data, dtype=float)


def pick_one(paths, label):
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"Missing {label}")
    return paths[0]


def find_run_inputs(run_dir):
    sim_dir = run_dir / "sim"
    adc_dir = sim_dir / "ADCdata"
    kurtosis_dir = sim_dir / "kurtosis_data"

    if not adc_dir.is_dir() or not kurtosis_dir.is_dir():
        return None

    try:
        return {
            "adc_intra": pick_one(adc_dir.glob("*intra*.pkl"), "intra ADC"),
            "adc_extra": pick_one(adc_dir.glob("*extra*.pkl"), "extra ADC"),
            "kurtosis_intra": pick_one(
                kurtosis_dir.glob("FINAL_kurtosis_intra*data.pkl"),
                "FINAL intra kurtosis",
            ),
            "kurtosis_extra": pick_one(
                kurtosis_dir.glob("FINAL_kurtosis_extra*data.pkl"),
                "FINAL extra kurtosis",
            ),
        }
    except FileNotFoundError as error:
        print(f"Skipping {run_dir}: {error}")
        return None


def radial_signal(array):
    return (array[:, 0] + array[:, 1]) / 2.0, array[:, 3]


def interpolate_to(time_target, values, time_source):
    order = np.argsort(time_source)
    return np.interp(time_target, time_source[order], values[order])


def compute_kperp_total(inputs):
    adc_intra = load_array(inputs["adc_intra"])
    adc_extra = load_array(inputs["adc_extra"])
    kurtosis_intra = load_array(inputs["kurtosis_intra"])
    kurtosis_extra = load_array(inputs["kurtosis_extra"])

    di, t_di = radial_signal(adc_intra)
    de, t_de = radial_signal(adc_extra)
    ki, t_ki = radial_signal(kurtosis_intra)
    ke, t_ke = radial_signal(kurtosis_extra)

    time = t_ki.copy()
    di = interpolate_to(time, di, t_di)
    de = interpolate_to(time, de, t_de)
    ke = interpolate_to(time, ke, t_ke)

    filename = inputs["kurtosis_intra"].name
    fi = extract_float(r"_avf_([0-9.]+)_", filename)
    if fi is None:
        fi = extract_float(r"_avf_([0-9.]+)", str(inputs["kurtosis_intra"]))
    if fi is None:
        raise ValueError(f"Could not extract intra-axonal volume fraction from {filename}")

    fe = 1.0 - fi
    dtotal = fi * di + fe * de

    with np.errstate(divide="ignore", invalid="ignore"):
        ktotal = (
            (fi * ki * di**2 + fe * ke * de**2) / dtotal**2
            + 3.0 * (fi * di**2 + fe * de**2) / dtotal**2
            - 3.0
        )

    invalid = ~np.isfinite(ktotal)
    if np.any(invalid):
        ktotal[invalid] = np.nan

    return {
        "time": time,
        "Kperp_total": ktotal,
        "Dperp_total": dtotal,
        "Dperp_intra": di,
        "Dperp_extra": de,
        "Kperp_intra": ki,
        "Kperp_extra": ke,
        "fi": fi,
        "fe": fe,
        "formula": KTOTAL_FORMULA,
    }


def iter_run_dirs(data_root):
    for experiment_dir in sorted(data_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        for run_dir in sorted(experiment_dir.iterdir()):
            if run_dir.is_dir() and (run_dir / "sim").is_dir():
                yield experiment_dir, run_dir


def collect_kperp_total(data_root, output_root):
    computed_dir = output_root / "computed_Kperp_total"
    computed_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(lambda: defaultdict(list))
    summary_rows = []

    for experiment_dir, run_dir in iter_run_dirs(data_root):
        inputs = find_run_inputs(run_dir)
        if inputs is None:
            continue

        metadata_text = " ".join(
            [experiment_dir.name, run_dir.name]
            + [path.name for path in inputs.values()]
        )
        diameter = extract_float(r"_d([0-9.]+)_", metadata_text)
        kappa = extract_int(r"_K(\d+)_", metadata_text)
        odi = extract_float(r"_ODI_([0-9.]+)", metadata_text)

        if diameter is None or kappa is None:
            print(f"Skipping {run_dir}: could not extract diameter/K metadata")
            continue

        print(f"Computing Kperp_total: K{kappa}, d={diameter}, {run_dir.name}")
        result = compute_kperp_total(inputs)
        result.update({
            "diameter": diameter,
            "kappa": kappa,
            "odi": odi,
            "run_dir": str(run_dir),
            "input_files": {name: str(path) for name, path in inputs.items()},
        })

        output_name = f"Kperp_total_K{kappa}_d{diameter}_{run_dir.name}.pkl"
        output_path = computed_dir / output_name
        with output_path.open("wb") as file:
            pickle.dump(result, file)

        grouped[f"K{kappa}"][diameter].append(result)
        summary_rows.append({
            "kappa": kappa,
            "diameter": diameter,
            "odi": odi,
            "fi": result["fi"],
            "fe": result["fe"],
            "formula": result["formula"],
            "run_dir": str(run_dir),
            "output_file": str(output_path),
        })

    summary_path = computed_dir / "Kperp_total_summary.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "kappa", "diameter", "odi", "fi", "fe", "formula",
                "run_dir", "output_file",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved computed Kperp_total data to: {computed_dir}")
    print(f"Saved summary CSV to: {summary_path}")
    return grouped


def average_runs_by_diameter(runs_by_diameter):
    averaged = {}
    for diameter, runs in runs_by_diameter.items():
        min_len = min(len(run["time"]) for run in runs)
        time_stack = np.asarray([run["time"][:min_len] for run in runs])
        k_stack = np.asarray([run["Kperp_total"][:min_len] for run in runs])

        averaged[diameter] = {
            "time": np.nanmean(time_stack, axis=0),
            "mean": np.nanmean(k_stack, axis=0),
            "std": np.nanstd(k_stack, axis=0, ddof=1) if len(runs) > 1 else np.zeros(min_len),
            "num_runs": len(runs),
            "fi_mean": float(np.mean([run["fi"] for run in runs])),
        }
    return averaged


def plot_kperp_total(grouped, output_root, diff_time_limit=100.0):
    colors = {
        1.68: "red",
        2.58: "blue",
        3.5: "green",
        4.5: "purple",
    }

    for k_value, runs_by_diameter in sorted(grouped.items(), key=lambda item: int(item[0][1:])):
        averaged = average_runs_by_diameter(runs_by_diameter)
        if not averaged:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        for diameter, data in sorted(averaged.items()):
            mask = data["time"] <= diff_time_limit
            time = data["time"][mask]
            mean = data["mean"][mask]
            std = data["std"][mask]
            color = colors.get(diameter, None)

            ax.plot(
                time,
                mean,
                color=color,
                label=f"{diameter:g} um",
                linewidth=3,
            )
            ax.fill_between(time, mean - std, mean + std, color=color, alpha=0.25)

        ax.set_xlabel(r"$t\;(\mathrm{ms})$", fontsize=27)
        ax.set_ylabel(r"$K_{\mathrm{total},\!\perp}$", fontsize=27)
        ax.set_title(f"$\\kappa$={k_value[1:]}", fontsize=27)
        ax.set_xlim([0.002, diff_time_limit])
        ax.set_ylim([0, 6.5])
        ax.legend(fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = output_root / f"Kperp_total_across_diameter_{k_value}.png"
        fig.savefig(plot_path, dpi=500, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot total perpendicular kurtosis from intra/extra compartments."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--diff-time-limit", type=float, default=100.0)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    grouped = collect_kperp_total(args.data_root, args.output_root)
    plot_kperp_total(grouped, args.output_root, args.diff_time_limit)


if __name__ == "__main__":
    main()