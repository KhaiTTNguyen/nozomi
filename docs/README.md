# Project Structure
## Directory Descriptions

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
# To activate
`source sim_venv/bin/activate`

# geometry-gen mean_diameter=2.0 sigma_diameter=0.5 gpu=1
# geometry-gen --config examples/configs/basic_substrate.json
# geometry-gen --folder examples/configs/batch_configs
```

## Example
## For substrates:
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



## For simulations:
TODO: Where to setup, where will endup, what will be inside.
Config file are configurable?
  "time_step" : 0.002,
  "num_spins" : 500000,
  "D0_intra": 2.25,
  "D0_extra": 2.0

### Input: 
MCDS can be setup in the `/nozomi/experiment/setup/simulation/default-sim.json`. Setup requires:
    * A list of substrates
    * Simulation time (ms)
    * Time step (ms)
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


## Experiment Visualization
What each script does
Where it outputs results
Example usage

plot_ADC_across_diameter.py experiment_result

2 parts
* Collect all experiments that you want to visualize into `/nozomi/experiment/experiment_visualization/results_for_visualization/`

* Run


## Nozomi Validation Experiments Guide

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/validation/narrow_pulse/test_03_periodic_boundary_xyz.py
```
## Nozomi Calibration Experiments Guide
### Number of Molecule and Time step Calibration
What it does
An example result is...

In code: rational for what 

```bash
# Format
CUDA_VISIBLE_DEVICES=<gpu_number> python3 ./tests/calibration/num_molecules_and_time_step_calibration/validation_for_num_molecules_and_time_step_search.py

# Example
CUDA_VISIBLE_DEVICES=0 python3 ./tests/calibration/num_molecules_and_time_step_calibration/validation_for_num_molecules_and_time_step_search.py
 ```

### Simulation Domain Segmentation Calibration
This is what...

An example is on figure

For showing run time with respect to number of segments
```bash
python3 ./tests/calibration/sim_domain_segment_calibration/plot_segment_calibration.py
```

For shpwing reproducibility of radial and axial diffusivity metrics regardless of number of segments
```bash
python3 ./tests/calibration/sim_domain_segment_calibration/plot_RD_AD_across_segment_choices.py
```


