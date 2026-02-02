# Project Structure
## Directory Descriptions
```bash
geometry-simulation-toolkit/
├── README.md
├── requirements.txt                   # Dependencies
├── .gitignore                         
│
├── bin/                              # Entry point scripts
│   ├── run-geometry-gen.sh
│   └── run-simulation.sh
│
├── simulation_toolkit/                 # Main package
│   ├── __init__.py
│   ├── config.py                     # Global configuration
│   ├── utils/                        # Shared utilities
│   │   ├── __init__.py
│   │   ├── gpu_manager.py
│   │   ├── adjust_geometry.py
│   │   ├── along_fiber_plot.py
│   │   ├── fiber_3D_plot.py
│   │   ├── orientation_plot.py
│   │   └── logging_utils.py
│   │
│   ├── substrate_generator/          # Component A
│   │   ├── __init__.py
│   │   ├── helper/                   
│   │   ├── geometric_optimization.py  # Step 3             
│   │   ├── initialization_2d.py       # Step 1
│   │   └── meshing.py                 # Step 2
│   │
│   ├── simulation_engine/            # Component B
│   │   ├── __init__.py
│   │   ├── diffsim3d.py                   # interface between user and cuda kernel
│   │   ├── geometry.py                # Geometry handling
│   │   ├── monte_carlo_sim_opt_D.py                # run script for simulation
│   │   └── sim3d_kernel_addition.cu                 # cuda kernel for simulation
│   │
│   └── cli/                          # Command line interface
│       ├── __init__.py
│       ├── main.py                   # Main CLI entry point
│       ├── substrate_cli.py          # Substrate generation CLI
│       └── simulation_cli.py         # Simulation CLI
│
├── experiment/                         # User examples
│   ├── README.md
│   ├── experiment_result/                      # Example configurations
│   │   ├── experiment1
│   │   └── experiment2
│   │
│   └── experiment_setup/                      # Example scripts
│       ├── multi_substrates
│       └── single_substrate 
│
├── validation_calibration_tests/                            # Unit tests
│   ├── __init__.py
│   ├── narrow_pulse/                      # Example configurations
│   │   ├── experiment1
│   │   └── experiment2
│   ├── num_molecules_and_time_step_calibration/                      # Example configurations
│   │   ├── experiment1
│   │   └── experiment2
│   ├── sim_domain_segment_calibration/                      # Example configurations
│   │   ├── experiment1
│   │   └── experiment2
│
└── docs/                            # Documentation
    ├── installation.md
    ├── user_guide.md
    └── api_reference.md
```

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

geometry-gen mean_diameter=2.0 sigma_diameter=0.5 gpu=1
geometry-gen --config examples/configs/basic_substrate.json
geometry-gen --folder examples/configs/batch_configs
```

## Example
## For substrates:
### Input: 
Substrate generation can be setup in the `/nozomi/experiment/experiment_setup/single_substrate/<your-susbtrate-name>.json`

Run default
```bash
./bin/run-geometry-gen.sh --gpu=0
```

Run with substrate config (JSON) file
```bash
./bin/run-geometry-gen.sh --gpu=1 --config=./experiment/experiment_setup/single_substrate/2025-11-20_13-20-18_default-substrate.json
```

<!-- Run with batched config folder
```bash
./bin/run-geometry-gen.sh --gpu=6 --config_folder=./experiment/experiment_setup/multi_substrates
``` -->


### Output: 
Generated substrates will be outputted to `/nozomi/experiment/experiment_result/`

Expected output:
Folder with experiment name
`/nozomi/experiment/experiment_result/VF0.3_experiment_d2.58_sig0.69_10axons_reproducibility_OD7/2026-01-28_19-36-58_d2.58_K7_ODI_0.0903_10fibers` which contain 2 folders 
```bash
data # (.pkl file containing the coordinates of spheres making up the axons in the substrate)
figs
├── init2D (# 2D initialization of substrate)
├── substrate_stats (# figures for orientation dispersion glyphs, radius variation along axon, CV of diameter, diameter distribution)
├── visual (# 3D visual of the substrate)
```



## For simulations:
### Input: 
MCDS can be setup in the `simulation_setup/simulation_params.json`. Setup requires:
    * A list of substrates
    * Simulation time (ms)
    * Time step (ms)
```bash
./bin/run-simulation.sh --substrates=<substrates-folder> --gpu=<gpu-number> --compartment=<axonal-compartment>

./bin/run-simulation.sh --substrates=./experiment/experiment_result/VF0.3_experiment_d2.58_sig0.69_50axons_reproducibility_OD20 --gpu=0 --compartment=intra

./bin/run-simulation.sh --substrates=./experiment/experiment_result/VF0.3_experiment_d2.58_sig0.69_50axons_reproducibility_OD20 --gpu=1 --compartment=extra
```

### Output: 
MCDS results will be outputted to 
```bash
`/nozomi/experiment/experiment_result/<substrates-folder>/<single-susbtrate-folder>/sim`. 
```

The `sim` folder include:
* A `.pkl` file storing the time-dependent diffusion coefficient.

* A figure visualizing the time-dependent diffusion coefficient.


## Experiment Visualization
What each script does
Where it outputs results
Example usage

## Nozomi Validation Experiments Guide

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/validation/narrow_pulse/test_03_periodic_boundary_xyz.py
```
## Nozomi Calibration Experiments Guide
### Number of Molecule and Time step Calibration
```bash
# Format
CUDA_VISIBLE_DEVICES=<gpu_number> python3 ./tests/calibration/num_molecules_and_time_step_calibration/validation_for_num_molecules_and_time_step_search.py

# Example
CUDA_VISIBLE_DEVICES=0 python3 ./tests/calibration/num_molecules_and_time_step_calibration/validation_for_num_molecules_and_time_step_search.py
 ```

### Simulation Domain Segmentation Calibration
```bash
# Format
CUDA_VISIBLE_DEVICES=<gpu_number> python3 ./tests/calibration/num_molecules_and_time_step_calibration/validation_for_num_molecules_and_time_step_search.py

# Example
CUDA_VISIBLE_DEVICES=0 python3 ./tests/calibration/num_molecules_and_time_step_calibration/validation_for_num_molecules_and_time_step_search.py
 ```
