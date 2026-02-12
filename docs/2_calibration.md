# Calibration Experiments
## Number of Molecule and Time step Calibration
This experiment determines the optimal number of molecules and time step size for achieving the best accuracy-reproducibility-computational efficiency trade-offs. It validates Monte Carlo numerically-simulated results against analytical solutions for intra-axonal diffusion in a single cylinder.

### What it does:
The calibration systematically tests different parameter combinations:
```python
molecules_values = [int(1e6), int(5e5), int(2e5), int(1e5), int(5e4), int(2e4), int(1e4)]
time_step_values = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]  # in ms
```

**For each combination, the experiment:**
1. Runs Monte Carlo diffusion simulation in a single cylinder geometry
2. Calculates time-dependent radial diffusion coefficient $D_{\perp}(t)$ 
3. Compares simulation results against analytical solutions from Burcaw et al. (2015)
4. Calculates Mean Absolute Error (MAE) and computation time
5. Repeats 5 times per configuration for reproducibility assessement.

#### How to run:
```bash
# Format
CUDA_VISIBLE_DEVICES=<gpu_number> python3  ./tests/calibration/num_molecules_and_time_step_calibration/calibration_for_num_molecules_and_time_step_search.py

# Example
CUDA_VISIBLE_DEVICES=0 python3 ./tests/calibration/num_molecules_and_time_step_calibration/calibration_for_num_molecules_and_time_step_search.py
 ```
This will take a while to run all the simulations for number of molecule/time step combinations and their repetitions.

#### Expected output:
Output will be saved to:
```bash
/nozomi/tests/calibration/num_molecules_and_time_step_calibration/figs/cylinder_validation_<experiment_date_time>/
```

**Key output files:**
- `single_cylinder_validation_summary_a0.5_D02.0_threshold0.02.csv`: Summary statistics for all parameter combinations
- `analytical_validation_analysis_a0.5_D02.0_threshold0.02.png`: Analysis plots showing:
  - **Left panel**: MAE vs time step for different molecule numbers
  - **Right panel**: Computation time vs number of molecules for different time steps
- Individual comparison plots: `molecules_<N>_timestep_<dt>/`: Detailed D(t) vs time comparisons for each configuration

**Typical results guidance:**
- Lower time steps (0.0001-0.002 ms) generally provide better accuracy
- Higher molecule numbers (>100,000) reduce statistical noise
- Time step 0.002 ms with 500,000 molecules provides good accuracy-efficiency balance

## Simulation Domain Segmentation Calibration
This calibration tests how the spatial segmentation of the simulation domain affects both computational time and result reproducibility. The simulation domain is divided into segments for efficient collision detection.

### What it tests:
The experiment analyzes segment numbers: `[5, 10, 15, 20, 25, 30, 35]` across 3D spatial dimensions (nsegx, nsegy, nsegz).

**Two key analyses:**
1. **Performance**: Runtime vs. number of segments
2. **Reproducibility**: Radial Diffusivity (RD) and Axial Diffusivity (AD) variability across different segmentation choices

### For runtime vs number of segments:
```bash
python3 ./tests/calibration/sim_domain_segment_calibration/plot_segment_calibration.py
```
#### Expected output:
Output will be saved to:
```bash
/nozomi/tests/calibration/sim_domain_segment_calibration/calibrate_segments_vs_runtime/figs/segment_calibration_plot.png
```
### For showing reproducibility of radial and axial diffusivity metrics regardless of number of segments
```bash
python3 ./tests/calibration/sim_domain_segment_calibration/plot_RD_AD_across_segment_choices.py
```
#### Expected output:
Validation plots showing that diffusion metrics remain consistent regardless of segmentation choice:
```bash
/nozomi/tests/calibration/sim_domain_segment_calibration/validate_same_results_with_same_substrate_different_segments_choices/figs/
```
