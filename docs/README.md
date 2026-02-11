# NOZOMI Documentation

This repo hosts the NOZOMI toolbox for generating 3D white matter numerical susbtrates with periodic boundaries and running Monte Carlo simulation of water diffusion in such substrates.

# Installation
* An NVIDIA GPU is needed for fast computation of the framework. 

The framework was developed and tested on NVIDIA RTX A5000 GPU with an AMD EPYC 7513 32-core CPU.

Check GPU availability:
```bash
nvidia-smi  # Shows available GPUs
```
Clone NOZOMI repo:
```bash 
git clone git@github.com:KhaiTTNguyen/nozomi.git
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
- **[Validation Experiments Guide](/nozomi/docs/1_validation.md)** - Complete substrate generation guide

## Calibration Experiments
- **[Calibration Experiments Guide](/nozomi/docs/2_calibration.m)** - Complete substrate generation guide

## Substrate Generation Guide:
- **[Substrate Generation Guide](/nozomi/docs/3_substrate_generation.md)** - Complete substrate generation guide

## Monte Carlo Diffusion Simulation (MCDS) guide:
- **[Monte Carlo Diffusion Simulation (MCDS) guide](/nozomi/docs/4_simulation.md)** -  Complete Monte Carlo simulations generation guide


<!-- ## Reproducibility Guide
- **[Reproducibility Guide for NOZOMI paper](reproducibility.md)** - Complete workflow with config files and example results
 -->
