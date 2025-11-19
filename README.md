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
├── geometry_toolkit/                 # Main package
│   ├── __init__.py
│   ├── config.py                     # Global configuration
│   ├── utils/                        # Shared utilities
│   │   ├── __init__.py
│   │   ├── gpu_manager.py
│   │   ├── file_utils.py
│   │   └── logging_utils.py
│   │
│   ├── substrate_generator/          # Component A
│   │   ├── __init__.py
│   │   ├── core.py                   # Main substrate generation logic
│   │   ├── geometry.py               # Geometry calculations
│   │   ├── fiber_models.py           # Fiber modeling
│   │   └── validators.py             # Input validation
│   │
│   ├── simulation_engine/            # Component B
│   │   ├── __init__.py
│   │   ├── core.py                   # Main simulation logic
│   │   ├── physics.py                # Physics calculations
│   │   ├── solvers.py                # Numerical solvers
│   │   └── output.py                 # Result processing
│   │
│   └── cli/                          # Command line interface
│       ├── __init__.py
│       ├── main.py                   # Main CLI entry point
│       ├── substrate_cli.py          # Substrate generation CLI
│       └── simulation_cli.py         # Simulation CLI
│
├── examples/                         # User examples
│   ├── README.md
│   ├── configs/                      # Example configurations
│   │   ├── basic_substrate.json
│   │   ├── complex_substrate.json
│   │   └── batch_configs/
│   │       ├── config1.json
│   │       ├── config2.json
│   │       └── config3.json
│   │
│   ├── notebooks/                    # Jupyter notebooks
│   │   ├── 01_basic_substrate_generation.ipynb
│   │   ├── 02_running_simulations.ipynb
│   │   └── 03_batch_processing.ipynb
│   │
│   └── scripts/                      # Example scripts
│       ├── generate_basic_substrate.py
│       ├── run_parameter_sweep.py
│       └── visualize_results.py
│
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_substrate_generator.py
│   ├── test_simulation_engine.py
│   └── test_cli.py
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

For substrates:
* Input: Substrate generation can be setup in the `substrate_setup/susbtrate_params.json`

* Output: Generated substrates will be outputted to `HIPASimExperiment/hipasim/data`

For simulations:
* Input: MCDS can be setup in the `simulation_setup/simulation_params.json`. Setup requires:
    * A list of substrates
    * Simulation time (ms)
    * Time step (ms)
    
* Output: MCDS results will be outputted to `HIPASimExperiment/hipasim/simulations/<output_folder>`. `<output_folder>` can be specified in `simulation_params.json` file

This includes a `.txt` specifying paths of the substrate data files used.