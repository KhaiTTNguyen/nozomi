# NOZOMI Documentation

This repo hosts the NOZOMI toolbox for generating 3D white matter numerical susbtrates with periodic boundaries and running Monte Carlo simulation of water diffusion in such substrates.

# Installation
* An NVIDIA GPU is needed for fast computation of the framework. 

The framework was developed and tested on NVIDIA RTX A5000 GPU with an AMD EPYC 7513 32-core CPU.

Check GPU availability:
```bash
nvidia-smi
```
Clone NOZOMI repo:
```bash 
git clone git@github.com:KhaiTTNguyen/nozomi.git
```
Since of of the simulations can take a long time to run, it is recommended to use `tmux` to keep the processes running when you turn off your terminal
```bash
# To start a session
tmux new -s <session-name>          

# To exit a session
Press and hold Ctrl, then press 'b', then release both keys. This is the tmux prefix.
Immediately after, press the d key. 

# To attach to a session
tmux a -t  <session-name> 

# To delete a session
Press and hold Ctrl, then press 'b', then release both keys.
Immediately after, press the 'x' key. 

```

Create virtual env
```bash
cd nozomi
python3 -m venv sim_venv
```

Activate virtual environment
```bash
source sim_venv/bin/activate
```
All software dependencies are included in `requirements.txt` file and can be installed by running:
```bash
pip install -r requirements.txt
```

# User Guide
Always activate virtual environment before any runs
```bash
source sim_venv/bin/activate
```

## Validation Experiments
- **[Validation Experiments Guide](https://github.com/KhaiTTNguyen/nozomi/blob/master/docs/1_validation.md)** - Validation experiments for Monte Carlo simulations

## Calibration Experiments
- **[Calibration Experiments Guide](https://github.com/KhaiTTNguyen/nozomi/blob/master/docs/2_calibration.md)** - Calibration experiments for 

## Substrate Generation Guide:
- **[Substrate Generation Guide](https://github.com/KhaiTTNguyen/nozomi/blob/master/docs/3_substrate_generation.md)** - Detailed substrate generation guide

## Monte Carlo Diffusion Simulation (MCDS) guide:
- **[Monte Carlo Diffusion Simulation (MCDS) guide](https://github.com/KhaiTTNguyen/nozomi/blob/master/docs/4_simulation.md)** -  Details for running Monte Carlo simulations
