## Installation Guide
```bash
cd nozomi

# To create virtual env
`python3 -m venv sim_venv`

# To activate
`source sim_venv/bin/activate`

# Install 
pip install -r requirements.txt

```

## User Guide
```bash
# Activate virtual environment
`source sim_venv/bin/activate`
```

## Validation Experiments Guide
These experiments validate the Monte Carlo diffusion engine for cases of:
* Free diffision in empty arena
* Intra and Extra axonal compartmentation
* Periodic boundary condition handling 

How to run:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/validation/narrow_pulse/test_03_periodic_boundary_xyz.py
```

Expected output: All 7 tests passed. 

Figures of spin positions pre- and post-simulations will be saved in 
```bash
/nozomi/tests/validation/narrow_pulse/test_figures
```

## Calibration Experiments Guide
### Number of Molecule and Time step Calibration
This experiment seek the optimal number of molecule and time step for optimal accuracy - reproducibility - computational efficiency trade-offs

```bash
# Format
CUDA_VISIBLE_DEVICES=<gpu_number> python3 ./tests/calibration/num_molecules_and_time_step_calibration/validation_for_num_molecules_and_time_step_search.py

# Example
CUDA_VISIBLE_DEVICES=0 python3 ./tests/calibration/num_molecules_and_time_step_calibration/validation_for_num_molecules_and_time_step_search.py
 ```

An example result is...

In code: rational for what 

## Substrate Generation Guide:
### Input: 
Substrate generation can be setup in the `/nozomi/experiment/experiment_setup/single_substrate/<your-susbtrate-name>.json`

TODO: what parameters goes into a JSON
they're scalars arrays

Run default
```bash
./bin/run-geometry-gen.sh --gpu=0
```

Run with substrate config (JSON) file
```bash
./bin/run-geometry-gen.sh --gpu=1 --config=./experiment/setup/substrate/single_substrate/2025-11-20_13-20-18_default-substrate.json
```

### Output: 
Generated substrates will be outputted to `/nozomi/experiment/result/`

Expected output:
Folder with experiment name
`/nozomi/experiment/result/VF0.3_experiment_d2.58_sig0.69_10axons_reproducibility_OD7/2026-01-28_19-36-58_d2.58_K7_ODI_0.0903_10fibers` which contain 2 folders 
```bash
2026-01-28_19-36-58_d2.58_K7_ODI_0.0903_10fibers
├── data # (.pkl file containing the coordinates of spheres making up the axons in the substrate)
└── figs
    ├── init2D (# 2D initialization of substrate)
    ├── substrate_stats (# figures for orientation dispersion glyphs, radius variation along axon, CV of diameter, diameter distribution)
    ├── visual (# 3D visual of the substrate)
```

## Monte Carlo Diffusion Simulation guide:

### Input: 
MCDS can be setup in the `/nozomi/experiment/setup/simulation/default-sim.json`. Setup requires:
* time_step : 0.002,
* num_spins : 500000,
* D0_intra: 2.25,
* D0_extra: 2.0

```bash
./bin/run-simulation.sh --substrates=<substrates-folder> --gpu=<gpu-number> --sim_time=<total-diffusion-time> --compartment=<axonal-compartment>

./bin/run-simulation.sh --substrates=./experiment/result/experiment_VF0.5_d2.58_sig0.69_50axons_OD20 --gpu=1 --sim_time=100  --compartment=intra

./bin/run-simulation.sh --substrates=./experiment/result/experiment_VF0.5_d2.58_sig0.69_50axons_OD20 --gpu=1 --sim_time=100  --compartment=extra
```

### Output: 
MCDS results will be outputted to 
```bash
`/nozomi/experiment/experiment_result/<substrates-folder>/<single-susbtrate-folder>/sim`. 
```

The `sim` folder include:
* A `ADCdata` folder that stores the `.pkl` file storing the time-dependent diffusion coefficient.

* A figure visualizing the time-dependent diffusion coefficient.
