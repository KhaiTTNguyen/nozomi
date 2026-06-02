## Monte Carlo Diffusion Simulation (MCDS) guide:

### Input: 
MCDS can be setup in the `/nozomi/experiment/setup/simulation/default-sim.json`. The default settings are:
```json
{  
  "time_step" : 0.002,
  "num_spins" : 500000,
  "D0_intra": 2.25,
  "D0_extra": 2.0,
  "nseg": 20
}
```
To run simulation experiments:
```bash
cd nozomi

./run-scripts/run-simulation.sh --substrates=<substrates-folder> --gpu=<gpu-number> --sim_time=<total-diffusion-time> --nseg=<number-of-segments-in-each-3Daxis> --compartment=<axonal-compartment>

# Example:
# NOTE: Substrates must be generated before simulation. 
# If no substrates, generate the default substrate, configured at `nozomi/experiment/setup/substrate/default/default-substrate.json:
./run-scripts/run-geometry-gen.sh --gpu=0

# Then run
./run-scripts/run-simulation.sh --substrates=./experiment/result/VF0.5_d2.58_OD200_bead1.0_100axons --gpu=0 --sim_time=100  --nseg=20 --compartment=intra


./run-scripts/run-simulation.sh --substrates=./experiment/result/2026-03-22_bead_1.24/bead1.24_d4.5_OD10_initVF0.145_500axons --gpu=4 --sim_time=100  --nseg=20 --compartment=intra


./run-scripts/run-simulation.sh --substrates=./experiment/result/VF0.5_d2.58_OD200_bead1.0_100axons --gpu=0 --sim_time=100  --nseg=20 --compartment=extra 


./run-scripts/run-simulation.sh --substrates=./experiment/result/myelin-mock/myelin_small_200_verification --gpu=2 --sim_time=100  --nseg=20 --compartment=intra

./run-scripts/run-simulation.sh --substrates=./experiment/result/set1_healthy_20260506_195745/bead0.5_d3.5_OD200_initVF0.27_500axons/d3.5_K200_ODI_0.0032_bead_0.5_VF_0.27 --gpu=6 --sim_time=100  --nseg=20 --compartment=extra
# NEEDS INTRA 2026-06-01 3:47pm
```

### Output: 
MCDS results are expected to be outputted to 
```bash
/nozomi/experiment/result/<substrates-folder>/<single-susbtrate-folder>/sim 
```

The `sim` folder include:
* A `ADCdata` folder that stores the `.pkl` file storing the time-dependent diffusion coefficient.

* A figure visualizing the diffusion coefficient ( $\mu\text{m}^2/\text{ms}$ ) with respect to diffusion time ($\text{ms}$)


### Appendix:
Mathematical details included [here](https://github.com/KhaiTTNguyen/nozomi/blob/master/docs/simulation_references.md) address:
* Modeling of the simulation mechanism
* Numerically-stable method for solving quadratic equation of collision interaction water molecules and axon membranes.
* Tolerance handling numerical precision around sphere boundaries.
