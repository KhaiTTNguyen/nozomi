import simulation_toolkit.toolkit_params as cfg
from simulation_toolkit.cli.substrate_main import substrate_main

params = {
    "orientation_shape_parameter": 2,   # low-K, high dispersion
    "box_length_init": 0,
    "box_length_z_init": 0,             # 0 => auto thin-z rule (45 um)
    "final_volume_fraction": 0.4,
    "num_fibers": 80,
    "mean_diameter": 2.5,
    "sigma_diameter": 0.6695,
    "dist_shape": 0.1,
    "space_buffer_starts_ends": 0.23,
    "spheres_spacing": 0.5,
    "space_buffer_repulse": 0.001,
    "w_overlap": 10,
    "w_curve": 3,
    "w_length": 3,
    "bead_spacing_mean": 5.70,
    "bead_spacing_stdv": 2.88,
    "bead_alpha_mean": 0.5,
    "bead_alpha_stdv": 0.048,
    # no g_ratio -> non-myelin save path
    "inner_sphere_spacing_ratio": 0.5,
    "repeats": 1,
}

substrate_main(params, "_tmp_aniso_smoke", folder_suffix="_test")
print("=== GENERATION RETURNED ===")
print("BOX_LENGTH (Lx=Ly) =", cfg.BOX_LENGTH)
print("BOX_LENGTH_Z (Lz)  =", cfg.BOX_LENGTH_Z)
print("NUM_FIBERS (final) =", cfg.NUM_FIBERS)
