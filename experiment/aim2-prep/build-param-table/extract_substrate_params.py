#!/usr/bin/env python3
"""
extract_substrate_params.py
===========================
Step 1 of the aim2-prep parameter-table build.

Walk the Aim 2 substrate data tree

    <data_root>/<param_group>/<substrate_replicate>/

and extract, for every substrate replicate, the *designed* and *achieved*
parameters needed to fill the achieved-parameter table. One CSV row is written
per substrate.

Sources for each value
----------------------
designed / achieved-avf  -> the substrate ``.pkl`` filename in ``<substrate>/data/``
    e.g. array500_fibers_boxL_80.0_2026-03-22_23-44_avf_0.69_d1.68_sig0.45wo10
         _wc3_wl3_K200_ODI_0.0032_17145.83_sec.pkl
      - achieved_avf  : _avf_<v>_        -> 0.69
      - designed_diam : _d<v>_           -> 1.68
      - designed_diam_sd : sig<v>wo      -> 0.45
      - designed_K    : _K<n>_           -> 200
      - designed_ODI  : _ODI_<v>_        -> 0.0032

achieved diameter mean/std -> ``figs/substrate_stats/Diameter_distribution_mean<m>_std<s>_<date>.png``

achieved global ODI    -> ``figs/substrate_stats/ODI/FOD_3D_glyph_global_watson_Kdes_<K>_Kfit_<x>_ODIfit_<v>.png``

designed_initVF        -> param-group folder name ``bead<b>_d<d>_OD<K>_initVF<vf>_<n>axons``

Substrates missing the global ODI ``ODIfit`` PNG are skipped (the ODI computation
is still under development). When more than one matching ODIfit PNG exists (e.g.
left over from an earlier run with a different Kfit), the most recently modified
file is used.

Usage
-----
    python extract_substrate_params.py
    python extract_substrate_params.py --data-root /abs/path/to/2026-03-22_bead_05 \
        --output substrate_params.csv
"""

import argparse
import csv
import os
import re
from pathlib import Path

# ------------------------------------------------------------------
# Default locations (resolved relative to this file so the script is
# runnable from any working directory and re-runnable as-is).
# ------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_BUILD_DIR = _HERE.parent                       # build-param-table/
_AIM2_PREP = _BUILD_DIR.parent                  # aim2-prep/
_DEFAULT_DATA_ROOT = _AIM2_PREP / "data" / "2026-03-22_bead_05"
_DEFAULT_OUTPUT = _BUILD_DIR / "substrate_params.csv"

# ------------------------------------------------------------------
# Regexes
# ------------------------------------------------------------------
# Param-group folder, e.g. bead0.5_d1.68_OD200_initVF0.465_500axons
_PARAM_GROUP_RE = re.compile(
    r"^bead(?P<bead>[\d.]+)_d(?P<diam>[\d.]+)_OD(?P<od>\d+)"
    r"_initVF(?P<initvf>[\d.]+)_\d+axons$"
)

# Designed params + achieved avf, all from the substrate .pkl filename.
_PKL_AVF_RE = re.compile(r"_avf_(?P<avf>[\d.]+)_")
_PKL_DIAM_RE = re.compile(r"_d(?P<diam>[\d.]+)_sig")
_PKL_SIG_RE = re.compile(r"_sig(?P<sig>[\d.]+)wo")
_PKL_K_RE = re.compile(r"_K(?P<k>\d+)_")
_PKL_ODI_RE = re.compile(r"_ODI_(?P<odi>[\d.]+)_")

# Achieved diameter histogram PNG.
_DIAM_PNG_RE = re.compile(
    r"^Diameter_distribution_mean(?P<mean>[\d.]+)_std(?P<std>[\d.]+)_.*\.png$"
)

# Achieved global ODI — 3D glyph PNG (must contain an ODIfit token to be usable).
_ODI_GLOBAL_RE = re.compile(
    r"^FOD_3D_glyph_global_watson_Kdes_\d+_Kfit_(?P<kfit>[\d.]+)_ODIfit_(?P<odi>[\d.]+)\.png$"
)

CSV_FIELDS = [
    "param_group",
    "substrate",
    "designed_diam",
    "designed_diam_sd",
    "designed_K",
    "designed_ODI",
    "designed_initVF",
    "achieved_avf",
    "achieved_diam_mean",
    "achieved_diam_std",
    "Kfit_global",
    "achieved_global_ODI",
]


def _find_pkl(substrate_path: Path):
    """Return the substrate .pkl Path inside <substrate>/data/, or None."""
    data_dir = substrate_path / "data"
    if not data_dir.is_dir():
        return None
    pkls = sorted(data_dir.glob("*.pkl"))
    if not pkls:
        return None
    # In normal cases there is exactly one; if several, prefer the newest.
    return max(pkls, key=lambda p: p.stat().st_mtime)


def _parse_pkl_name(pkl_name: str):
    """Parse designed params + achieved avf from the .pkl filename."""
    out = {}
    for key, rx, grp, cast in (
        ("achieved_avf", _PKL_AVF_RE, "avf", float),
        ("designed_diam", _PKL_DIAM_RE, "diam", float),
        ("designed_diam_sd", _PKL_SIG_RE, "sig", float),
        ("designed_K", _PKL_K_RE, "k", int),
        ("designed_ODI", _PKL_ODI_RE, "odi", float),
    ):
        m = rx.search(pkl_name)
        out[key] = cast(m.group(grp)) if m else None
    return out


def _parse_diameter_png(stats_dir: Path):
    """Return (mean, std) achieved diameter from the newest matching PNG, or (None, None)."""
    if not stats_dir.is_dir():
        return None, None
    matches = []
    for entry in stats_dir.iterdir():
        m = _DIAM_PNG_RE.match(entry.name)
        if m:
            matches.append((entry.stat().st_mtime, float(m.group("mean")), float(m.group("std"))))
    if not matches:
        return None, None
    matches.sort(key=lambda t: t[0])
    _, mean, std = matches[-1]
    return mean, std


def _parse_odi_png(odi_dir: Path, regex: re.Pattern):
    """Return (kfit, odi) from the newest PNG matching regex, or (None, None)."""
    if not odi_dir.is_dir():
        return None, None
    matches = []
    for entry in odi_dir.iterdir():
        m = regex.match(entry.name)
        if m:
            matches.append((entry.stat().st_mtime, float(m.group("kfit")), float(m.group("odi"))))
    if not matches:
        return None, None
    matches.sort(key=lambda t: t[0])
    _, kfit, odi = matches[-1]
    return kfit, odi


def collect_rows(data_root: Path):
    """Walk the data tree and return (rows, skipped) for all substrate replicates."""
    rows = []
    skipped = []

    for group_entry in sorted(data_root.iterdir()):
        if not group_entry.is_dir():
            continue
        gm = _PARAM_GROUP_RE.match(group_entry.name)
        if gm is None:
            continue
        designed_initvf = float(gm.group("initvf"))

        for sub_entry in sorted(group_entry.iterdir()):
            if not sub_entry.is_dir():
                continue

            pkl = _find_pkl(sub_entry)
            if pkl is None:
                skipped.append((group_entry.name, sub_entry.name, "no .pkl"))
                continue
            pkl_vals = _parse_pkl_name(pkl.name)

            stats_dir = sub_entry / "figs" / "substrate_stats"
            diam_mean, diam_std = _parse_diameter_png(stats_dir)

            odi_dir = stats_dir / "ODI"
            kfit_g, odi_g = _parse_odi_png(odi_dir, _ODI_GLOBAL_RE)

            if odi_g is None:
                skipped.append((group_entry.name, sub_entry.name, "missing global ODIfit"))
                continue

            rows.append({
                "param_group": group_entry.name,
                "substrate": sub_entry.name,
                "designed_diam": pkl_vals["designed_diam"],
                "designed_diam_sd": pkl_vals["designed_diam_sd"],
                "designed_K": pkl_vals["designed_K"],
                "designed_ODI": pkl_vals["designed_ODI"],
                "designed_initVF": designed_initvf,
                "achieved_avf": pkl_vals["achieved_avf"],
                "achieved_diam_mean": diam_mean,
                "achieved_diam_std": diam_std,
                "Kfit_global": kfit_g,
                "achieved_global_ODI": odi_g,
            })

    return rows, skipped


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT,
                        help=f"Batch folder to walk (default: {_DEFAULT_DATA_ROOT})")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT,
                        help=f"Output CSV path (default: {_DEFAULT_OUTPUT})")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        parser.error(f"data-root does not exist: {data_root}")

    rows, skipped = collect_rows(data_root)
    write_csv(rows, args.output)

    print(f"Data root : {data_root}")
    print(f"Wrote     : {args.output}  ({len(rows)} substrate rows)")
    if skipped:
        print(f"Skipped   : {len(skipped)} substrate(s)")
        for group, sub, reason in skipped:
            print(f"    - {group}/{sub}  [{reason}]")


if __name__ == "__main__":
    main()
