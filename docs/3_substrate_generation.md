## Substrate Generation Guide:

NOZOMI generates 3D white matter axon substrates with configurable microstructural properties. The substrate generation process creates realistic axonal geometries with controlled volume fraction, diameter distributions, orientation dispersion, and beading patterns.

### Configuration Setup

Substrate generation is configured via JSON files located in:
```bash
/nozomi/experiment/setup/substrate/single_substrate/<your-substrate-name>.json
```

#### Configuration File Structure
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
      "bead_amplitude_mean": 1.0,
      "bead_amplitude_stdv": 0.8,
      "repeats": 4
  }
}
```
NOTE: `experiment_name` can be 
* specified by in `<your-substrate-name>.json` config file above
* `auto_generated` and will be named in the format: `VF{target_volume_fraction}_d{mean_diameter}_OD{orientation_shape_parameter}_bead{bead_amplitude_mean}_{num_fibers}axons`

#### Parameter Descriptions
**User-controlled parameters includes:**
- `orientation_shape_parameter` (K): Controls fiber orientation dispersion using Watson distribution. Higher values = more aligned fibers (typical: 7-200)
- `target_volume_fraction`: Desired axonal volume fraction (0.0-1.0, typical: 0.3-0.7)
- `mean_diameter`: Mean axon diameter in μm (typical: 1.0-9.0)
- `sigma_diameter`: Standard deviation of diameter distribution in μm
- `bead_amplitude_mean`: Mean amplitude of diameter variation due to beading
- `num_fibers`: Number of axons to generate in the substrate (typical: 500 for good reproducibility of simulation results)
- `repeats`: Number of substrate realizations to generate with same parameters

**Default parameters include:**
- `box_length_init`: Initial simulation box size in μm. Set to 0 for automatic calculation based on volume fraction
- `dist_shape`: Shape parameter for Generalized Extreme Value diameter distribution (typical: 0.1)

- `bead_spacing_mean`: Average distance between beads along axon in μm
- `bead_spacing_stdv`: Standard deviation of bead spacing
- `bead_amplitude_stdv`: Standard deviation of beading amplitude
- `space_buffer_starts_ends`: Buffer space around fiber start/end points in μm
- `spheres_spacing`: Spacing between spheres along axon centerline relative to diameter
- `space_buffer_repulse`: Minimum separation distance between fibers in μm
- `w_overlap`: Weight for overlap penalty in geometric optimization (typical: 10)
- `w_curve`: Weight for curvature smoothness penalty (typical: 3)
- `w_length`: Weight for length preservation penalty (typical: 3)
#### Example choices for user-controlled parameters

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

### Running Substrate Generation

**Default example generation:**
```bash
./run-scripts/run-geometry-gen.sh --gpu=0
```

**With custom configuration file:**
```bash
./run-scripts/run-geometry-gen.sh --gpu=1 --config=./experiment/setup/substrate/single_substrate/d258-K20-substrate.json
```

**With custom configuration file and output folder:**
```bash
./run-scripts/run-geometry-gen.sh --gpu=0 --config=./experiment/setup/substrate/single_substrate/d258-K200-substrate.json --output_folder_path=./tests/calibration/sim_domain_segment_calibration/data
```

### Output Structure

Generated substrates are saved to `/nozomi/experiment/result/` with the following structure:

```bash
/nozomi/experiment/result/<experiment_name>/<timestamp>_d<mean_diameter>_K<orientation_shape_parameter>_ODI_<odi-value>_bead_<bead_amplitude>_<num_fibers>fibers/
├── data/
│   └── <spheres_coordinates_file>.pkl                                    # 3D coordinates & radius of all spheres
└── figs/
    ├── init2D/                                                           
    │   ├── init2d.pkl                                                    # 2D packing data
    │   ├── 2D_converged_packing_<...>.png                                # Initial 2D fiber positions
    │   └── PBC_3D_start_end_<date>                                       # 3D visualization of start end points
    ├── substrate_stats/                                                  # Substrate properties plots
    │   ├── Optimized_diameter_distribution_mean<d>_std<std_d>_<date>.png # Diameter_distribution
    │   ├── Optimized_along_axon_radius_variation_<date>.png              #Along axon radius variation
    │   ├── Optimized_CV_outer_diameter_<date>_CVmean_<CVmean>_CVstd_<CVstd>.png # Coefficient of variation (CV) of diameter
    │   └── ODI/  
    │       ├── OD_histogram                                               # Fiber orientation distribution (FOD) on unit sphere
    │       ├── FOD_histogram                                              # FOD as 3D spherical harmonics glyph
    │       └── watson_samples_kappa_<orientation_shape_parameter>.png     # Samples of fibers orientation from Watson distribution
    └── visual/                    
        └── optimized_<view>.png                                           # 3D substrate view
```

#### Understanding the Output

**Data Files:**
- `spheres_coordinates.pkl`: Contains numpy array with columns [x, y, z, radius, fiber_id, sphere_id]. Use this for Monte Carlo simulations.

**Note for metrics in filenames:**
- `K`: Watson distribution orientation shape parameter (higher = more aligned)
- `ODI_<odi-value>` (Orientation Dispersion Index): 2/π × arctan(1/K), range 0-1

