# Monte-Carlo Diffusion Simulator Benchmark

Benchmarks **Nozomi** against **Camino**, **Disimpy**, and **MC/DC** on a
canonical parallel-cylinder white-matter substrate using the same PGSE
sequence, same b-values, same walker count, same time step, and the *same
random seed per repeat*. All four engines run Monte-Carlo simulation
on the same 100-cylinder substrate, and the simulated signal is compared
against a closed-form analytic reference (van Gelderen 1994).

**Intra-axonal accuracy comparison** (`run_all.sh` + `aggregate.py`) —
signal vs b-value and wall-clock runtime across the four engines on one
fixed substrate, benchmarked against the van Gelderen analytic signal.

---

## Purpose

For the intra-axonal compartment of a parallel-cylinder substrate, the
perpendicular PGSE signal has a **closed-form analytic solution**
(van Gelderen et al. 1994) `S_analytic(b)`.

This experiment measures two quantities per framework:

- **Accuracy**: the perpendicular-gradient signal $S(b)$ at five b-values,
  compared to the analytic reference.
- **Runtime**: total wall-clock time from process start (substrate load +
  mesh I/O + simulation + signal write) to finish, averaged over repeats.

### Substrate (identical for all four engines)

Square-packed parallel cylinders along axis = $z$. The radius is chosen to model a
**large-caliber spinal-cord axon** that sits just above the resolution floor
of a clinical scanner — small enough to be physiologically realistic, large
enough that perpendicular restricted attenuation is detectable above MC shot
noise at $G_{\max} = 80$ mT/m.

| property             | value                          |
| -------------------- | ------------------------------ |
| cylinder radius      | 4.0 μm (diameter = 8 μm)       |
| cylinder separation  | 9.997 μm                       |
| lattice              | 10 × 10 = **100 cylinders**    |
| voxel (xy × z)       | 99.97 × 99.97 × 20 μm          |
| ICVF                 | 0.503                          |
| $D_0$                | 2 × 10⁻⁹ m²/s                  |
| cylinder walls       | impermeable                    |
| walker seeding       | **intra-axonal only**          |

The radius / separation defaults live in [`config.py`](config.py); to change
the substrate, edit `BenchConfig.cylinder_radius_m` and `cylinder_sep_m`. Each
timestamped benchmark run writes its own substrate cache under
`results/<timestamp>/substrate_cache/`; if that cache is reused with a changed
config, [`substrate.py`](substrate.py) refreshes the exported files.

Exported formats (written once by [`substrate.py`](substrate.py)):

- `results/<timestamp>/substrate_cache/substrate.json` — metadata + centers/radii
- `results/<timestamp>/substrate_cache/substrate.ply`  — ASCII PLY lateral-surface mesh
- `results/<timestamp>/substrate_cache/substrate_xy.png` — cross-section plot
- `results/<timestamp>/substrate_cache/pgse.scheme`    — Camino-style STEJSKALTANNER scheme
- `results/<timestamp>/substrate_cache/mcdc_cylinders.txt`, `mcdc_pgse.scheme` (written by the
  MC/DC runner on first invocation)

### PGSE sequence

**Clinical floor protocol** (shortest realisable PGSE at $G_{\max} = 80$ mT/m
for $b_{\max} = 2500$ s/mm²): the gradient pulses are placed back-to-back
($\Delta = \delta$, no inter-pulse gap) so the diffusion time hits its
hardware-limited minimum. Single x-axis gradient (perpendicular to fibers).

| parameter                 | value                              |
| ------------------------- | ---------------------------------- |
| δ (pulse width)           | 20.3 ms                            |
| Δ (pulse separation)      | 20.3 ms (back-to-back, Δ = δ)      |
| diffusion time $t_d$      | $\Delta - \delta/3$ = 13.5 ms      |
| TE (= Δ + δ, minimum)     | 40.6 ms                            |
| gradient axis             | +x (perpendicular to cylinders)    |
| b-values                  | 0, 500, 1000, 1500, 2000, 2500 s/mm²|
| Gmax                      | 0, 35.4, 50.1, 70.8, 79.1 mT/m     |
| time step (default)       | 2 μs → 20,300 steps / TE           |
| $\sqrt{6 D \, dt}$        | 0.155 μm                           |
| γ (all engines)           | 2.6751525 × 10⁸ rad·s⁻¹·T⁻¹        |

The protocol saturates the 80 mT/m hardware at the highest b-value. Reducing
$t_d$ further is not possible without exceeding $G_{\max}$ — this is the
Nilsson/Fieremans axon-diameter resolution limit; for $G_{\max} = 80$ mT/m
at b = 2500 it sets a minimum resolvable diameter of $\sim 5$ μm, which is
why R = 4 μm is the smallest interesting radius for this hardware.

Monte-Carlo settings are tunable in [`config.py`](config.py) (`n_walkers`,
`n_repeats`, `base_seed`, `time_step_s`). Every run with `base_seed = 12345`
and `repeat r` uses effective seed `12345 + r`, shared across all four
engines, so cross-framework differences reflect algorithmic / discretization
differences — not different MC draws.

### Analytic reference

[`analytic.py`](analytic.py) implements the van Gelderen 1994 perpendicular
PGSE signal for a solid cylinder of radius $R$ and free diffusivity $D$ under
rectangular Stejskal-Tanner pulses:

$$-\ln \frac{S}{S_0} = 2 \gamma^2 G^2 \sum_{m=1}^{\infty} \frac{1}{\alpha_m^6 \, D^2 \, (\alpha_m^2 R^2 - 1)} \left[ 2\alpha_m^2 D \delta - 2 + 2 e^{-\alpha_m^2 D \delta} + 2 e^{-\alpha_m^2 D \Delta} - e^{-\alpha_m^2 D (\Delta - \delta)} - e^{-\alpha_m^2 D (\Delta + \delta)} \right]$$

where $\alpha_m$ are the positive roots of $J_1'(\alpha R) = 0$. The first
20 terms are summed using the current benchmark radius and PGSE settings, and
the resulting analytic signal is written into each benchmark log by
[`run_all.sh`](run_all.sh).

### Per-framework notes

| framework | geometry input                                                    | intra seeding           | RNG control                                                     | implementation     | output                                      |
| --------- | ----------------------------------------------------------------- | ----------------------- | --------------------------------------------------------------- | ------------------ | ------------------------------------------- |
| Nozomi    | `Structure3D` sphere stacks (200 spheres/cyl, edge ghosts)        | `init_location=intra`   | stdlib `random`, `np.random`, `sim.setup(initstates_int=seed)`  | PyCUDA             | $\langle \cos \phi_{\rm end} \rangle$ per b |
| Camino    | native `-geometry cylinder -packing SQUARE` (100-cyl lattice, PBC)| `-initial intra`        | `-seed <int>`                                                   | Java (CPU)         | big-endian float stream                     |
| Disimpy   | triangle mesh (`substrates.mesh`, lateral-only)                   | `init_pos="intra"`      | `simulation(seed=<int>)`                                        | Numba CUDA         | complex signal per measurement              |
| MC/DC     | `cylinders_list` (z-oriented, scale = 1000 m→mm)                  | `ini_walkers_pos intra` | `seed <int>` line in `sim.conf`                                 | C++ std::thread (CPU, 8 proc) | text DWI file                               |

All four frameworks use native intra-compartment seeding keywords (no custom
rejection sampling), so the seeded walker distributions match each engine's
own internal geometry.

MC/DC parallelizes its walker loop across `num_process` worker threads
(`std::thread`). The benchmark sets `num_process = 8` (`BenchConfig.n_processes`),
a typical laptop / workstation core count, so the reported MC/DC runtime is
machine-representative and reproducible across machines. Override with
`BENCH_N_PROCESS`. (Thread count affects runtime only, not the signal.)

### Outputs

The aggregator ([`aggregate.py`](aggregate.py)) writes:

- `summary.json` — per-framework mean ± std runtime, mean ± std signal at
  each b-value, and per-b-value nRMSE against Camino.
- `signal_vs_b.png` — signal-vs-b curves for all four engines plus the
  van Gelderen analytic reference.
- `runtime.png` — wall-clock runtime bar chart with error bars across
  repeats.

---

## Results directory layout

Every invocation of `run_all.sh` writes into a **timestamped subfolder** so no
sweep is ever overwritten:

```
tests/benchmark/results/
├── 2026-04-23_152301/              ← one full sweep
│   ├── camino/run_{00,01,02}.json
│   ├── disimpy/run_{00,01,02}.json
│   ├── mcdc/run_{00,01,02}/...
│   ├── nozomi/run_{00,01,02}.json
│   ├── summary.json
│   ├── signal_vs_b.png
│   └── runtime.png
├── 2026-04-24_090142/              ← next sweep, untouched
└── latest  →  2026-04-23_152301/   ← symlink to most recent
```

Override the timestamp with `BENCH_RESULTS_DIR=path/to/dir bash run_all.sh`
if you want a named folder.

---

## How to run

From the repo root with the Nozomi venv activated:

```bash
cd /path/to/nozomi
source sim_venv/bin/activate

# --- Intra-axonal accuracy experiment (all 4 frameworks × cfg.n_repeats) ---
bash tests/benchmark/run_all.sh

# Subset (e.g. only Nozomi and Camino)
bash tests/benchmark/run_all.sh nozomi camino

# Larger sweep / different GPU (env overrides take precedence over config.py)
BENCH_N_WALKERS=50000 BENCH_N_REPEATS=3 GPU=2 bash tests/benchmark/run_all.sh

# Different time step (to study convergence)
BENCH_TIME_STEP_S=1e-6 bash tests/benchmark/run_all.sh

# Use the full (intra + extra) compartment instead of intra-only
BENCH_COMPARTMENT=all bash tests/benchmark/run_all.sh

# --- Re-aggregate an older sweep ---
python -m tests.benchmark.aggregate \
    --run-dir tests/benchmark/results/2026-04-23_152301
```

Omitting `--run-dir` defaults to `results/latest/` (the most recent sweep).

### Environment-variable overrides (honored by `config.py` / `run_all.sh`)

| variable               | effect                                                        |
| ---------------------- | ------------------------------------------------------------- |
| `BENCH_N_WALKERS`      | override `BenchConfig.n_walkers`                              |
| `BENCH_N_REPEATS`      | override `BenchConfig.n_repeats`                              |
| `BENCH_TIME_STEP_S`    | override `BenchConfig.time_step_s`                            |
| `BENCH_GRADIENT_AXIS`  | `gx,gy,gz` override of `BenchConfig.gradient_axis`            |
| `BENCH_COMPARTMENT`    | `intra` (default) \| `extra` \| `all` — walker seeding region |
| `BENCH_N_PROCESS`      | MC/DC worker threads (default 8)                              |
| `BENCH_RESULTS_DIR`    | explicit output folder (otherwise timestamped)                |
| `CUDA_VISIBLE_DEVICES` / `GPU` | GPU index for Nozomi and Disimpy                      |

---

## Hardware

AMD EPYC 7513 CPU, 503 GB RAM, 8 × RTX A5000 24 GB (Nozomi / Disimpy share any
one GPU via `CUDA_VISIBLE_DEVICES`). MC/DC runs with 8 worker processes, a
representative typical-machine core count.

## Installation notes

- **Camino**: expects a Camino install with `datasynth` on disk. Set
  `CAMINO_BIN=<dir>` to the directory containing `datasynth` (default is
  `<repo_parent>/camino/bin`).
- **Disimpy**: `pip install disimpy` inside `sim_venv` (already installed).
- **MC/DC**: cloned and built at `tests/benchmark/external/MCDC_Simulator_public/`.
  Binary path: `…/MCDC_Simulator_public/MC-DC_Simulator`. Override with
  `MCDC_BIN=<path>`. The binary only runs when invoked with `--conf <file>`
  (plain `<file>` exits without running — upstream quirk).

## Files

```
tests/benchmark/
├── README.md                   ← this file
├── config.py                   ← BenchConfig (single source of truth), env overrides
├── substrate.py                ← cylinder lattice + PLY / JSON / mesh exporters
├── analytic.py                 ← van Gelderen 1994 analytic reference signal
├── run_nozomi.py               ← Nozomi runner (DwiSim3d + Structure3D)
├── run_camino.py               ← Camino runner (datasynth; SQUARE or PLY mode)
├── run_disimpy.py              ← Disimpy runner (substrates.mesh + simulations.simulation)
├── run_mcdc.py                 ← MC/DC runner (MC-DC_Simulator --conf)
├── run_all.sh                  ← accuracy-sweep orchestrator (timestamps results)
├── aggregate.py                ← accuracy summary + signal_vs_b.png + runtime.png
└── results/
    ├── <YYYY-MM-DD_HHMMSS>/    ← one per sweep
  │   ├── substrate_cache/     ← substrate used by this sweep
    └── latest  →  most recent  ← symlink
```
