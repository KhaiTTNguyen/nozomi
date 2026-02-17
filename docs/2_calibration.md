# Calibration Experiments
## 1) Number of Molecule and Time step Calibration
This experiment determines the optimal number of molecules and time step size for achieving the best accuracy-reproducibility-computational efficiency trade-offs. It validates Monte Carlo numerically-simulated results against analytical solutions for intra-axonal diffusion in a single cylinder.

### What it does:
The calibration tests different parameter combinations:
```python
molecules_values = [int(1e6), int(5e5), int(2e5), int(1e5), int(5e4), int(2e4), int(1e4)]
time_step_values = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]  # in ms
n_repeats = 5                                                           # number of repetitions for each combination
```

**For each combination, the experiment:**
1. Runs Monte Carlo diffusion simulation in a single cylinder geometry
2. Calculates time-dependent radial diffusion coefficient $D_{\perp}(t)$ 
3. Compares simulation results against analytical solutions from Burcaw et al. (2015)
4. Calculates Mean Absolute Error (MAE) and computation time
5. Repeats 5 times per configuration for reproducibility assessement.

#### How to run:
This will take ~11hr to run all the simulations for all number of molecule/time step combinations and their repetitions. (Using a `tmux` session is recommended).

```bash
cd nozomi
# ALWAYS activate virtual environment when opening a new `tmux` window / session.
source sim_venv/bin/activate

# Format
CUDA_VISIBLE_DEVICES=<gpu_number> python3  ./tests/calibration/num_molecules_and_time_step_calibration/calibration_for_num_molecules_and_time_step_search.py

# Example, change your <gpu_number> as needed. Below, <gpu_number>=0
CUDA_VISIBLE_DEVICES=0 python3 ./tests/calibration/num_molecules_and_time_step_calibration/calibration_for_num_molecules_and_time_step_search.py
 ```

#### Expected output:
Output will be saved to:
```bash
/nozomi/tests/calibration/num_molecules_and_time_step_calibration/figs/cylinder_validation_<experiment_date_time>/
```

**Key output files:**
- `cylinder_validation_summary_a0.5_D02.0_threshold0.05.csv`: Summary statistics for all parameter combinations
- `analytical_validation_analysis_a0.5_D02.0_threshold0.05.png`: Analysis plots showing:
  - **Left panel**: MAE vs time step for different molecule numbers
  - **Right panel**: Computation time vs number of molecules for different time steps
- Individual comparison plots: `molecules_<N>_timestep_<dt>/`: Detailed $D_{\perp}(t)$ vs time comparisons for each combination.

**Typical results guidance:**
- Lower time steps (0.0001-0.002 ms) generally provide better accuracy
- Higher molecule numbers (>100,000) reduce statistical noise
- Time step 0.002 ms with 500,000 molecules provides good accuracy-computational efficiency balance

## 2) Simulation Domain Segmentation Calibration
This calibration tests how the spatial segmentation of the simulation domain affects computational time and result reproducibility. The simulation domain is divided into segments for efficient collision detection.

### What it tests:
The experiment analyzes segment numbers: `[5, 10, 15, 20, 25, 30, 35]` across 3D spatial dimensions (x, y, z axes).

**Two key evaluations:**
1. **Performance**: Runtime vs. number of segments
2. **Reproducibility**: Radial Diffusivity and Axial Diffusivity variability across different segmentation choices

### For runtime vs number of segments:
Create a substrate config file for substrate generation, in `nozomi/experiment/setup/substrate/single_substrate/`

An example config file can be `d168-K200-single-substrate-for-segment-calibration.json`:
```json
{
  "experiment_name": "auto_generated",  
  "parameters": {
      "orientation_shape_parameter": 200,
      "box_length_init": 0,
      "target_volume_fraction": 0.65,
      "num_fibers": 500,
      "mean_diameter":  1.68,
      "sigma_diameter":  0.45,
      "dist_shape": 0.1,
      "space_buffer_starts_ends": 0.23,
      "spheres_spacing": 0.5,
      "space_buffer_repulse": 0.001,
      "w_overlap":10,
      "w_curve": 3, 
      "w_length": 3,
      "bead_spacing_mean": 5.70,
      "bead_spacing_stdv": 2.88,
      "bead_amplitude_mean": 1.0,
      "bead_amplitude_stdv": 0.8,
      "repeats": 1
  }
}
```
Then generate the substrate. `--gpu` number can be changed if multiple GPUs are available. 
To make the output specific for this calibration experiment, specify the `--output_folder_path` to `./tests/calibration/sim_domain_segment_calibration/data`
```bash
cd nozomi
# ALWAYS activate virtual environment when opening a new `tmux` window / session.
source sim_venv/bin/activate

./run-scripts/run-geometry-gen.sh --gpu=0 --config=./experiment/setup/substrate/single_substrate/d168-K200-single-substrate-for-segment-calibration.json --output_folder_path=./tests/calibration/sim_domain_segment_calibration/data
```
Then run Monte-Carlo diffusion simulation in the generated substrate. These can be run in separate `tmux` windows, each on a different `--gpu` for faster computation.
```bash
# 5 segments - intra axonal simulation
./run-scripts/run-simulation.sh --substrates=./tests/calibration/sim_domain_segment_calibration/data/VF0.65_d1.68_OD200_bead1.0_500axons --gpu=0 --sim_time=100  --nseg=5 --compartment=intra

# 5 segments - extra axonal simulation
./run-scripts/run-simulation.sh --substrates=./tests/calibration/sim_domain_segment_calibration/data/VF0.65_d1.68_OD200_bead1.0_500axons --gpu=0 --sim_time=100  --nseg=5 --compartment=extra

# 10 segments - intra axonal simulation
./run-scripts/run-simulation.sh --substrates=./tests/calibration/sim_domain_segment_calibration/data/VF0.65_d1.68_OD200_bead1.0_500axons --gpu=0 --sim_time=100  --nseg=10 --compartment=intra

# 10 segments - extra axonal simulation
./run-scripts/run-simulation.sh --substrates=./tests/calibration/sim_domain_segment_calibration/data/VF0.65_d1.68_OD200_bead1.0_500axons --gpu=0 --sim_time=100  --nseg=10 --compartment=extra

# 15 segments - intra axonal simulation
./run-scripts/run-simulation.sh --substrates=./tests/calibration/sim_domain_segment_calibration/data/VF0.65_d1.68_OD200_bead1.0_500axons --gpu=0 --sim_time=100  --nseg=15 --compartment=intra

# 15 segments - extra axonal simulation
./run-scripts/run-simulation.sh --substrates=./tests/calibration/sim_domain_segment_calibration/data/VF0.65_d1.68_OD200_bead1.0_500axons --gpu=0 --sim_time=100  --nseg=15 --compartment=extra

# 20 segments - intra axonal simulation
./run-scripts/run-simulation.sh --substrates=./tests/calibration/sim_domain_segment_calibration/data/VF0.65_d1.68_OD200_bead1.0_500axons --gpu=0 --sim_time=100  --nseg=20 --compartment=intra

# 20 segments - extra axonal simulation
./run-scripts/run-simulation.sh --substrates=./tests/calibration/sim_domain_segment_calibration/data/VF0.65_d1.68_OD200_bead1.0_500axons --gpu=0 --sim_time=100  --nseg=20 --compartment=extra

# 25 segments - intra axonal simulation
./run-scripts/run-simulation.sh --substrates=./tests/calibration/sim_domain_segment_calibration/data/VF0.65_d1.68_OD200_bead1.0_500axons --gpu=0 --sim_time=100  --nseg=25 --compartment=intra

# 25 segments - extra axonal simulation
./run-scripts/run-simulation.sh --substrates=./tests/calibration/sim_domain_segment_calibration/data/VF0.65_d1.68_OD200_bead1.0_500axons --gpu=0 --sim_time=100  --nseg=25 --compartment=extra

# 30 segments - intra axonal simulation
./run-scripts/run-simulation.sh --substrates=./tests/calibration/sim_domain_segment_calibration/data/VF0.65_d1.68_OD200_bead1.0_500axons --gpu=0 --sim_time=100  --nseg=30 --compartment=intra

# 35 segments - extra axonal simulation
./run-scripts/run-simulation.sh --substrates=./tests/calibration/sim_domain_segment_calibration/data/VF0.65_d1.68_OD200_bead1.0_500axons --gpu=0 --sim_time=100  --nseg=30 --compartment=extra

# 35 segments - intra axonal simulation
./run-scripts/run-simulation.sh --substrates=./tests/calibration/sim_domain_segment_calibration/data/VF0.65_d1.68_OD200_bead1.0_500axons --gpu=0 --sim_time=100  --nseg=35 --compartment=intra

# 35 segments - extra axonal simulation
./run-scripts/run-simulation.sh --substrates=./tests/calibration/sim_domain_segment_calibration/data/VF0.65_d1.68_OD200_bead1.0_500axons --gpu=0 --sim_time=100  --nseg=35 --compartment=extra
```
The simulation data is stored in `VF0.65_d1.68_OD200_bead1.0_500axons/2026-02-13_17-37_d1.68_K200_ODI_0.0032_bead_1.0_500fibers/sim/ADCdata`.
To plot computation time vs. number of segments:
```bash
python3 tests/calibration/sim_domain_segment_calibration/plot_segment_calibration_from_files.py "tests/calibration/sim_domain_segment_calibration/data/VF0.65_d1.68_OD200_bead1.0_500axons/2026-02-13_17-37_d1.68_K200_ODI_0.0032_bead_1.0_500fibers/sim/ADCdata"
```

#### Expected output:
Output will be saved to:
```bash
/nozomi/tests/calibration/sim_domain_segment_calibration/calibrate_segments_vs_runtime/figs/segment_calibration_plot.png
```
### For showing reproducibility of radial and axial diffusivity metrics regardless of number of segments
```bash
cd nozomi
# Basic usage with specified input folder:
python3 tests/calibration/sim_domain_segment_calibration/plot_RD_AD_across_segment_choices.py tests/calibration/sim_domain_segment_calibration/data/VF0.65_d1.68_OD200_bead1.0_500axons/2026-02-13_17-37_d1.68_K200_ODI_0.0032_bead_1.0_500fibers/sim/ADCdata

# Usage with specified input folder & output folder:
python3 tests/calibration/sim_domain_segment_calibration/plot_RD_AD_across_segment_choices.py <path/to/ADCdata> -o <custom/output/folder>

# Full use
python3 tests/calibration/sim_domain_segment_calibration/plot_RD_AD_across_segment_choices.py <input_folder_path> --rd_time_limit <max_diffusion_time_in_ms_for_plotting_radial_diffusivity> --ad_time_limit <max_diffusion_time_in_ms_for_plotting_axial_diffusivity> -o <output_folder_path>
```
#### Expected output:
Validation plots showing that diffusion metrics remain consistent regardless of segment choice:
```bash
/nozomi/tests/calibration/sim_domain_segment_calibration/validate_same_results_with_same_substrate_different_segments_choices/figs/
```
