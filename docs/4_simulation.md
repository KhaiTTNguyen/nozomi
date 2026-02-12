## Monte Carlo Diffusion Simulation (MCDS) guide:

### Input: 
MCDS can be setup in the `/nozomi/experiment/setup/simulation/default-sim.json`. The default settings are:
```json
{  
  "time_step" : 0.002,
  "num_spins" : 500000,
  "D0_intra": 2.25,
  "D0_extra": 2.0
}
```
To run simulation experiments:
```bash
./bin/run-simulation.sh --substrates=<substrates-folder> --gpu=<gpu-number> --sim_time=<total-diffusion-time> --compartment=<axonal-compartment>

# Example:
# NOTE: Substrates must be generated before simulation. 
# If no substrates, generate them via:
./bin/run-geometry-gen.sh --gpu=0

# Then run
./bin/run-simulation.sh --substrates=./experiment/result/VF0.5_d2.58_OD200_bead1.0_100axons --gpu=1 --sim_time=100  --compartment=intra

./bin/run-simulation.sh --substrates=./experiment/result/VF0.5_d2.58_OD200_bead1.0_100axons --gpu=1 --sim_time=100  --compartment=extra
```

### Output: 
MCDS results are expected to be outputted to 
```bash
/nozomi/experiment/experiment_result/<substrates-folder>/<single-susbtrate-folder>/sim 
```

The `sim` folder include:
* A `ADCdata` folder that stores the `.pkl` file storing the time-dependent diffusion coefficient.

* A figure visualizing the diffusion coefficient ( $\mu\text{m}^2/\text{ms}$ ) with respect to diffusion time ($\text{ms}$)

