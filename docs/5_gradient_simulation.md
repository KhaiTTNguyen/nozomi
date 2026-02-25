## Monte Carlo Diffusion Simulation (MCDS) guide:

### Input: 
Simulations incorporating wide pulse gradient can be setup in the `/nozomi/experiment/setup/simulation/`. Sample PGSE and OGSE gradients settings are provided:
```json
// pgse_sim_params.json
{  
  "sim_time" : 78, 
  "time_step" : 0.002,
  "num_spins" : 500000,
  "D0_intra": 2.25,
  "D0_extra": 2.0,
  "nseg": 20,
  "gradient" : {
    "type": "PGSE",
    "direction": [1, 0, 0],
    "big_delta": 40.0,
    "little_delta": 10.0
  }
}

// ogse_sim_params.json
{  
  "sim_time" : 78, 
  "time_step" : 0.002,
  "num_spins" : 500000,
  "D0_intra": 2.25,
  "D0_extra": 2.0,
  "nseg": 20,
  "gradient" : {
    "type": "OGSE",
    "direction": [1, 0, 0],
    "num_oscil": 1,
    "duration": 38.4
  }
}
```
To run simulation experiments:
```bash
cd nozomi

./run-scripts/run-gradient-simulation.sh --substrates=<substrates-folder> --gpu=<gpu-number> --sim_time=<total-diffusion-time> --nseg=<number-of-segments-in-each-3Daxis> --compartment=<axonal-compartment> --config=<path-to-simulation-config-file>

# Example:
# NOTE: Substrates must be generated before simulation. 
# If no substrates, generate the default substrate, configured at `nozomi/experiment/setup/substrate/default/default-substrate.json:
./run-scripts/run-geometry-gen.sh --gpu=0

# Then run
./run-scripts/run-gradient-simulation.sh --substrates=./experiment/result/VF0.5_d2.58_OD200_bead1.0_100axons --gpu=0 --sim_time=78 --nseg=20 --compartment=intra --config=./experiment/setup/simulation/pgse_sim_params.json

./run-scripts/run-gradient-simulation.sh --substrates=./experiment/result/VF0.5_d2.58_OD200_bead1.0_100axons --gpu=0 --sim_time=78 --nseg=20 --compartment=extra --config=./experiment/setup/simulation/ogse_sim_params.json
```

### Output: 
Results are expected to be outputted to 
```bash
/nozomi/experiment/result/<substrates-folder>/<single-susbtrate-folder>/gradient_sim 
```

The `gradient_sim` folder include:
<!-- * A `ADCdata` folder that stores the `.pkl` file storing the time-dependent diffusion coefficient.

* A figure visualizing the diffusion coefficient ( $\mu\text{m}^2/\text{ms}$ ) with respect to diffusion time ($\text{ms}$) -->


