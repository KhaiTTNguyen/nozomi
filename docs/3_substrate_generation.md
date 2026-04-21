# Substrate Generation Guide

NOZOMI generates 3D white matter axon substrates with configurable microstructural properties. The substrate generation process creates realistic axonal geometries with controlled volume fraction, diameter distributions, orientation dispersion, and beading patterns.

## Substrate configuration

Substrate generation is configured via JSON files located in:
```bash
/nozomi/experiment/setup/substrate/single_substrate/<your-substrate-name>.json
```

### Configuration file structure
```json
{
  "experiment_name": "auto_generated",  
  "parameters": {
      "orientation_shape_parameter": 20,
      "box_length_init": 0,
      "target_volume_fraction": 0.65,
      "num_fibers": 500,
      "mean_diameter": 2.58,
      "sigma_diameter": 0.69,
      "dist_shape": 0.1,
      "space_buffer_starts_ends": 0.23,
      "spheres_spacing": 0.5,
      "space_buffer_repulse": 0.001,
      "w_overlap": 10,
      "w_curve": 3, 
      "w_length": 3,
      "bead_spacing_mean": 5.70,
      "bead_spacing_stdv": 2.88,
      "bead_alpha_mean": 0.21,
      "bead_alpha_stdv": 0.17,
      "repeats": 1
  }
}
```
NOTE: `"experiment_name"` can be 
* `"auto_generated"` and will be auto generated in the format: `VF{target_volume_fraction}_d{mean_diameter}_OD{orientation_shape_parameter}_bead{bead_alpha_mean}_{num_fibers}axons`
* user-specified in the `"experiment_name"` field.

### Parameter descriptions
**User-controlled parameters includes:**
- `orientation_shape_parameter` ($\kappa$): Controls fiber orientation dispersion using Watson distribution. Higher values = more aligned fibers (typical: 7-200).
- `target_volume_fraction`: Desired axonal volume fraction (typical: 0.3-0.7).
- `mean_diameter`: Mean axon diameter in $\mu m$ (typical: 1.0-9.0).
- `sigma_diameter`: Standard deviation of diameter distribution in $\mu m$.
- `bead_alpha_mean`: Dimensionless scaling factor controlling the mean beading amplitude relative to the local axon radius ($r_0$). The bead bump added at each bead location is `alpha × r0 × exp(...)`, so `bead_alpha_mean` directly controls mean CV of radius variation, independent of axon diameter. Typical range: 0.0–2.0 (e.g., 0.21 for mild beading, 0.83 for strong beading with CV≈0.28).
- `num_fibers`: Number of axons to generate in the substrate (typical: 500 for good reproducibility of simulation results).
- `repeats`: Number of substrate realizations to generate with same parameters.

**Default parameters include:**
- `box_length_init`: Initial simulation box size in $\mu m$. Set to 0 for automatic calculation based on `mean_diameter`, `target_volume_fraction` and `num_fibers`.
- `dist_shape`: Shape parameter for Generalized Extreme Value diameter distribution (typical: 0.1)

- `bead_spacing_mean`: Average distance between beads along axon in $\mu m$
- `bead_spacing_stdv`: Standard deviation of bead spacing in $\mu m$
- `bead_alpha_stdv`: Standard deviation of the alpha scaling factor across axons (dimensionless). Controls the spread of CV values across the population of axons. alpha is drawn once per axon from Normal(`bead_alpha_mean`, `bead_alpha_stdv`), so increasing this value increases CV_stdv across axons. Typical range: 0.0–1.0.
- `space_buffer_starts_ends`: Buffer space around fiber start/end points in $\mu m$
- `spheres_spacing`: Spacing between spheres along axon centerline, as a ratio relative to radius.
- `space_buffer_repulse`: Minimum separation distance between fibers in $\mu m$
- `w_overlap`: Weight for overlap penalty in geometric optimization (typical: 10)
- `w_curve`: Weight for curvature smoothness penalty (typical: 3)
- `w_length`: Weight for length preservation penalty (typical: 3)
### Example choices for user-controlled parameters

**High-density, dispersed substrate (e.g., corpus callosum):**
```json
  "orientation_shape_parameter": 10,
  "target_volume_fraction": 0.7,
  "num_fibers": 500,
  "mean_diameter": 1.5
```

**Low-density, aligned substrate (e.g., pathological spinal cord):**
```json
  "orientation_shape_parameter": 200,
  "target_volume_fraction": 0.3,
  "num_fibers": 800,
  "mean_diameter": 4.5
```

## Substrate generation
NOTE: `--gpu` number can be changed if multiple GPUs are available. 

**Default example generation:**
This will use the default substrate configuration at `nozomi/experiment/setup/substrate/default/default-substrate.json`:

```bash
cd nozomi
# ALWAYS activate virtual environment when opening a new `tmux` window / session.
source sim_venv/bin/activate

./run-scripts/run-geometry-gen.sh --gpu=0
```

**With custom configuration file:**
```bash
./run-scripts/run-geometry-gen.sh --gpu=0 --config=./experiment/setup/substrate/single_substrate/d258-K20-substrate.json

./run-scripts/run-geometry-gen.sh --gpu=7 --config=./experiment/setup/substrate/single_substrate/d45-K200-beading4-substrate.json
```

**With custom configuration file and output folder:**
```bash
./run-scripts/run-geometry-gen.sh --gpu=0 --config=./experiment/setup/substrate/single_substrate/d168-K200-single-substrate-for-segment-calibration.json --output_folder_path=./tests/calibration/sim_domain_segment_calibration/data

./run-scripts/run-geometry-gen.sh --gpu=4 --config=./experiment/setup/substrate/single_substrate/2026-03-22-bead0.3/d05-K200-beading0.3-substrate.json --output_folder_path=./experiment/result/2026-03-22_bead_03
```

### Output Structure

Unless `--output_folder_path` is given, generated substrates are auto-saved to `/nozomi/experiment/result/` with the following structure:

```bash
/nozomi/experiment/result/<experiment_name>/<timestamp>_d<mean_diameter>_K<orientation_shape_parameter>_ODI_<odi-value>_bead_<bead_alpha_mean>_<num_fibers>fibers/
├── data/
│   └── <spheres_coordinates>.pkl                                         # 3D coordinates & radius of all spheres
└── figs/
    ├── init2D/                                                           
    │   ├── init2d.pkl                                                    # 2D packing data
    │   ├── <2D_converged_initialization>.png                             # Initial 2D fiber positions
    │   └── <Start_end_points_in_3D>.png                                  # 3D visualization of start end points
    ├── substrate_stats/                                                  # Substrate properties plots
    │   ├── Diameter_distribution_mean<d>_std<std_d>_<date>.png           # Diameter_distribution
    │   ├── Along_axon_radius_variation_<date>.png                        # Along axon radius variation
    │   ├── CV_outer_diameter_<date>_CVmean_<CVmean>_CVstd_<CVstd>.png    # Coefficient of variation (CV) of radius across and along axons
    │   └── ODI/  
    │       ├── <OD_histogram>.png                                         # Fiber orientation distribution (FOD) on unit sphere
    │       ├── <FOD_3D_glyph>.png                                         # FOD as 3D spherical harmonics glyph
    │       └── Watson_samples_kappa_<orientation_shape_parameter>.png     # Samples of fibers orientation from Watson distribution
    └── visual/                    
        └── <3D_substrate_view>.png                                        # 3D substrate view
```

#### Understanding the Output

**Data Files:**
- `<spheres_coordinates>.pkl`: Contains numpy array with `[optimized_fibers, box_length]` or `[[x, y, z, radius, fiber_id, sphere_id], box_length]`. Use this for Monte Carlo simulations.

**Substrate visualization**
- `<3D_substrate_view>.png` show the 3D substrate view. The figure can take a few hours to be generated for a large number of fibers.

**Note for metrics in filenames:**
- `K` or `kappa`: Watson distribution orientation shape parameter (higher = more aligned)
- `ODI_<odi-value>` (Orientation Dispersion Index): 2/π × arctan(1/`K`), range 0-1

### Appendix:
Mathematical details included [here](https://github.com/KhaiTTNguyen/nozomi/blob/master/docs/substrate_generation_references.md) address:
* Beading design
* Substrate optimization design

To plot OD histogram and FOD 3D glyph
```bash
cd /home/nguyt16@ds.vanderbilt.edu/nozomi
source sim_venv/bin/activate
python simulation_toolkit/cli/reprocess_substrate_orientation.py \
    --root experiment/result/2026-03-22_bead_03
```