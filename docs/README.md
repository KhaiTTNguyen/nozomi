# NOZOMI Documentation

This repo hosts the NOZOMI toolbox for generating 3D white matter numerical susbtrates with periodic boundaries and running Monte Carlo simulation of water diffusion in such substrates.

# Installation

## Hardware Requirements

**⚠️ IMPORTANT:**
As of this writing, an **NVIDIA GPU** copatible with **CUDA 12** is required for this framework. 

### System Information
**Development & Testing Environment**:
* **Primary**: NVIDIA RTX A5000 GPUs + AMD EPYC 7513 32-core CPU
* **OS**: Linux


### Check Your System
```bash
# Check GPU availability
nvidia-smi

# Check CUDA version, output for our system is below  
nvcc --version
# Cuda compilation tools, release 12.4, V12.4.131
# Build cuda_12.4.r12.4/compiler.34097967_0

# Check driver version, output for our system is below
cat /proc/driver/nvidia/version
# NVRM version: NVIDIA UNIX x86_64 Kernel Module  570.207 
# GCC version:  gcc version 9.4.0 (Ubuntu 9.4.0-1ubuntu1~20.04.2) 
```

## Installation Steps
Clone NOZOMI repo:
```bash 
git clone git@github.com:KhaiTTNguyen/nozomi.git
```
Since of of the simulations can take a long time to run, it is recommended to use `tmux` to keep the processes running when you turn off your terminal:
```bash
# To start a session
tmux new -s <session-name>          

# To exit a session
Press and hold Ctrl + <b>, then release both keys. This is the tmux prefix.
Immediately after, press the <d> key. 

# To attach to a session
tmux a -t  <session-name> 

# To create additional windows (after attaching to a session)
Press and hold Ctrl + <b>, then release both keys.
Immediately after, [Shift + <5>] OR [Shift + <''>]

# To delete a session
Press and hold Ctrl + <b>, then release both keys.
Immediately after, <x> key. 

```

Create virtual env
```bash
cd nozomi
python3 -m venv sim_venv
```

**ALWAYS** activate virtual environment when opening a new `tmux` window / session.
```bash
source sim_venv/bin/activate
```
All software dependencies are included in `requirements.txt` file and can be installed by running:
```bash
pip install -r requirements.txt

# Then check
python3 check_minimal_requirements.py
```

# User Guide
(Again) **ALWAYS** activate virtual environment when opening a new `tmux` window / session.
```bash
source sim_venv/bin/activate
```

## Validation Experiments
- **[Validation Experiments Guide](https://github.com/KhaiTTNguyen/nozomi/blob/master/docs/1_validation.md)** - Validation experiments for Monte Carlo simulations

## Calibration Experiments
- **[Calibration Experiments Guide](https://github.com/KhaiTTNguyen/nozomi/blob/master/docs/2_calibration.md)** - Calibration experiments for Monte Carlo simulations

## Substrate Generation Guide:
- **[Substrate Generation Guide](https://github.com/KhaiTTNguyen/nozomi/blob/master/docs/3_substrate_generation.md)** - Detailed substrate generation guide

## Monte Carlo Diffusion Simulation (MCDS) guide:
- **[Monte Carlo Diffusion Simulation (MCDS) guide](https://github.com/KhaiTTNguyen/nozomi/blob/master/docs/4_simulation.md)** -  Details for running Monte Carlo simulations

### Acknowledgements
This project was supported by the National Institute of Biomedical Imaging and Bioengineering grant R01EB03195