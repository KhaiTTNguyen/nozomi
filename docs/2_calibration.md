## Calibration Experiments
### Number of Molecule and Time step Calibration
This experiment seek the optimal number of molecule and time step for optimal accuracy - reproducibility - computational efficiency trade-offs

What it does
    molecules_values = [int(1e6), int(5e5), int(2e5), int(1e5), int(5e4), int(2e4), int(1e4)]
    time_step_values = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    n_repeats = 10
    
    Calculate analytical solutions for cylinder
An example result is...

In code: rational for what 

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
/nozomi/tests/calibration/num_molecules_and_time_step_calibration/figs/cylinder_validation_<experiement_date_time>/
```

### Simulation Domain Segmentation Calibration
This is what...


#### For showing run time with respect to number of segments
```bash
python3 ./tests/calibration/sim_domain_segment_calibration/plot_segment_calibration.py
```
#### Expected output:
Output will be saved to:
```bash
/nozomi/tests/calibration/sim_domain_segment_calibration/calibrate_segments_vs_runtime/figs/segment_calibration_plot.png
```
#### For showing reproducibility of radial and axial diffusivity metrics regardless of number of segments
```bash
python3 ./tests/calibration/sim_domain_segment_calibration/plot_RD_AD_across_segment_choices.py
```
#### Expected output:
Output will be saved to:
```bash
/home/nguyt16@ds.vanderbilt.edu/nozomi/tests/calibration/sim_domain_segment_calibration/validate_same_results_with_same_substrate_different_segments_choices/figs
```
