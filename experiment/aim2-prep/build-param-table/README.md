# build-param-table

Builds the achieved-parameter table from the `2026-03-22_bead_05` substrate data.

## Re-run (2 steps)

```bash
cd nozomi/experiment/aim2-prep/build-param-table

# 1) Extract one row per substrate -> substrate_params.csv
python extract_substrate_params.py

# 2) Aggregate into per-group mean +/- SD -> param_table_summary.csv + param_table.md
python summarize_param_table.py
```

Re-run both steps any time the ODI computation is updated. Both scripts overwrite
their outputs, so they are safe to run repeatedly.

## What gets extracted (per substrate)

- **achieved Vin / designed diam, diam SD, K, ODI** — from the `.pkl` filename in `<substrate>/data/`
- **achieved axon diameter (mean, std)** — from `figs/substrate_stats/Diameter_distribution_mean*_std*.png`
- **achieved global ODI** — from `figs/substrate_stats/ODI/FOD_3D_glyph_global_watson_*_ODIfit_*.png`

Substrates missing the global ODI `ODIfit` PNG are skipped (printed to the console).
When duplicate ODI/diameter PNGs exist, the most recently modified one is used.

## Outputs

- `substrate_params.csv` — one row per substrate (intermediate, editable).
- `param_table_summary.csv` — per group `(designed_diam, designed_K)`, mean & SD columns.
- `param_table.md` — Markdown table matching the target figure.
