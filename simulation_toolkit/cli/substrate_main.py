from simulation_toolkit.substrate_generator.initialization_2d import Init2D
from simulation_toolkit.substrate_generator.meshing import Meshing
from simulation_toolkit.substrate_generator.geometric_optimization import GeometricOptimization
from simulation_toolkit.substrate_generator.helper.validate_input import validate_parameters
import simulation_toolkit.substrate_generator.helper.watson_distribution as wd
import simulation_toolkit.toolkit_params as config_params
import numpy as np
import time
import torch
import os.path
import simulation_toolkit.utils.common_utils as util
from matplotlib.pyplot import cm
from simulation_toolkit.utils import fiber_3D_plot
from simulation_toolkit.utils import along_fiber_plot
from simulation_toolkit.utils import orientation_plot


def estimate_initial_vf(target_3d_vf, kappa, mean_diameter,
                        bead_alpha_mean, bead_alpha_stdv,
                        bead_spacing_mean, bead_spacing_stdv,
                        space_buffer=None,
                        num_watson_samples=2000, num_bead_realizations=200):
    """
    Estimate the 2D initial volume fraction needed to achieve a target 3D volume fraction.

    The 3D VF amplifies relative to the 2D VF by two factors:
      VF_3D ≈ (VF_2D / A_buffer) × A_path × A_bead

    A_path:   helix arc length / Lz, averaged over Watson(K) fiber directions.
    A_bead:   mean(r_beaded² / r₀²), capturing beading-induced volume inflation.
    A_buffer: (r₀ + b/2)² / r₀², corrects for the 2D packing using buffered radii
              while 3D VF uses real radii.

    Returns the estimated VF_2D = VF_3D_target × A_buffer / (A_path × A_bead).
    """
    # --- A_buffer: space buffer correction ---
    if space_buffer is None:
        space_buffer = 0.137 * mean_diameter
    r0 = mean_diameter / 2
    A_buffer = ((r0 + space_buffer / 2) / r0) ** 2
    # --- A_path: helix arc length amplification ---
    mu_dir = np.array([0, 0, 1])
    watson = wd.WatsonDistribution(mu_dir, kappa)
    directions = watson.sample(num_watson_samples)
    dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]

    # Lateral displacement ratios (normalized by Lz)
    ratio_x = dx / dz
    ratio_y = dy / dz

    # Vectorized arc length computation over all sampled directions
    t = np.linspace(0, 1, 500)
    # Helix derivatives (normalized by Lz), matching meshing.py parameterization
    dxdt = ratio_x[:, None] / 2 * np.pi * np.sin(np.pi * t[None, :])
    dydt = ratio_y[:, None] * np.pi / 2 * np.cos(np.pi / 2 * t[None, :])
    integrand = np.sqrt(dxdt**2 + dydt**2 + 1.0)
    arc_ratios = np.trapz(integrand, t, axis=1)
    A_path = np.mean(arc_ratios)

    # --- A_bead: beading radius² amplification ---
    r0 = mean_diameter / 2
    sigma_bead = 2.74 * r0  # matches meshing.py: sigma = 2.74 * r0
    axon_length = 30.0  # representative axon length in µm

    # Lognormal parameters for bead spacings (matching meshing.py)
    u, v = bead_spacing_mean, bead_spacing_stdv
    mu_ln = np.log(u**2 / np.sqrt(v**2 + u**2))
    sigma_ln = np.sqrt(np.log(1 + (v**2 / u**2)))

    z = np.linspace(0, axon_length, 2000)
    r2_ratios = np.zeros(num_bead_realizations)
    for j in range(num_bead_realizations):
        spacings = np.random.lognormal(mu_ln, sigma_ln, size=len(z))
        bead_positions = np.cumsum(spacings)
        bead_positions = bead_positions[bead_positions < axon_length]

        alpha = np.random.normal(bead_alpha_mean, bead_alpha_stdv)
        r = np.full_like(z, r0)
        if len(bead_positions) > 0:
            diffs = z[:, None] - bead_positions[None, :]
            bumps = np.exp(-diffs**2 / (2 * sigma_bead**2))
            r = r + alpha * r0 * bumps.sum(axis=1)

        # Detrend: replicate process_result_endpoints from meshing.py
        slope = (r[-1] - r[0]) / len(r)
        x_idx = np.linspace(0, len(r), len(r))
        r = r - slope * x_idx
        r = r - r[0] + r0
        r = np.clip(r, a_min=0.415 * r0, a_max=None)

        r2_ratios[j] = np.mean(r**2) / r0**2
    A_bead = np.mean(r2_ratios)

    # --- Combined ---
    A_total = A_path * A_bead
    vf_2d = target_3d_vf * A_buffer / A_total

    print(f"VF estimation: A_path={A_path:.4f}, A_bead={A_bead:.4f}, A_buffer={A_buffer:.4f}, A_total={A_total:.4f}")
    print(f"Target 3D VF={target_3d_vf:.3f} -> Estimated 2D init VF={vf_2d:.4f}")

    return vf_2d


def substrate_main(params, experiment_folder):
    # Validate parameters before proceeding
    validate_parameters(params)
    config_params.ORIENTATION_SHAPE_PARAM = params['orientation_shape_parameter']
    config_params.BOX_LENGTH = params['box_length_init']

    # Determine 2D initial volume fraction
    target_3d_vf = params.get('final_volume_fraction', None)
    if target_3d_vf is not None:
        config_params.VOLUME_FRACTION = estimate_initial_vf(
            target_3d_vf=target_3d_vf,
            kappa=params['orientation_shape_parameter'],
            mean_diameter=params['mean_diameter'],
            bead_alpha_mean=params['bead_alpha_mean'],
            bead_alpha_stdv=params['bead_alpha_stdv'],
            bead_spacing_mean=params['bead_spacing_mean'],
            bead_spacing_stdv=params['bead_spacing_stdv']
        )
    else:
        config_params.VOLUME_FRACTION = params['target_volume_fraction']
    config_params.NUM_FIBERS = params['num_fibers']
    config_params.MEAN_DIAMETER = params['mean_diameter']   
    config_params.SIGMA_DIAMETER = params['sigma_diameter']   
    config_params.DISTRIBUTION_SHAPE = params['dist_shape']   
    config_params.SPACE_BUFFER_STARTS_ENDS = 0.137 * params['mean_diameter']   
    config_params.SPHERE_SPACING = params['spheres_spacing']   
    config_params.SPACE_BUFFER_REPULSE = params['space_buffer_repulse']  
    config_params.W_OVERLAP = params['w_overlap'] 
    config_params.W_CURVE = params['w_curve']
    config_params.W_LENGTH = params['w_length']
    config_params.BEAD_SPACING_MEAN = params['bead_spacing_mean']   
    config_params.BEAD_SPACING_STDV = params['bead_spacing_stdv']
    config_params.BEAD_ALPHA_MEAN = params['bead_alpha_mean']
    config_params.BEAD_ALPHA_STDV = params['bead_alpha_stdv']

    print(f"Building substrate with: \
        volume_fraction={config_params.VOLUME_FRACTION},\
        diameter_mean={config_params.MEAN_DIAMETER},\
        orientation_shape_parameter={config_params.ORIENTATION_SHAPE_PARAM}, \
        bead_alpha_mean={config_params.BEAD_ALPHA_MEAN},\
        num_fibers={config_params.NUM_FIBERS}") 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config_params.EXP_DATE_TIME = util.get_date_time()        
    config_params.ODI_INDEX = np.round(2/np.pi * np.arctan(1/config_params.ORIENTATION_SHAPE_PARAM), 4)
    config_params.SUBSTRATE_OUTPUT_FOLDER_PATH = os.path.join(config_params.OUTPUT_FOLDER_PATH, experiment_folder, \
                                                       str(config_params.EXP_DATE_TIME)+\
                                                        '_d'+str(config_params.MEAN_DIAMETER)+\
                                                        '_K'+str(int(config_params.ORIENTATION_SHAPE_PARAM))+\
                                                    '_ODI_'+str(config_params.ODI_INDEX)+\
                                                    '_bead_'+str(config_params.BEAD_ALPHA_MEAN)+'_'+\
                                                        str(config_params.NUM_FIBERS) +'fibers')

    # --- Iterative VF refinement loop ---
    vf_tolerance = 0.03
    max_vf_refinements = 3 if target_3d_vf is not None else 1
    current_vf_2d = config_params.VOLUME_FRACTION

    for vf_iter in range(max_vf_refinements):
        if vf_iter > 0:
            print(f"\n--- VF refinement iteration {vf_iter + 1}: adjusting VF_2D from {prev_vf_2d:.4f} to {current_vf_2d:.4f} ---")
            config_params.BOX_LENGTH = params['box_length_init']  # reset box length for recalculation
        config_params.VOLUME_FRACTION = current_vf_2d
        not_converged=True
        while not_converged:
            st = time.time()
            # -------- 2D initialization of axon start/end points --------
            initialization2D = Init2D(  device=device,
                                        date_time=config_params.EXP_DATE_TIME,
                                        orientation_shape_parameter=config_params.ORIENTATION_SHAPE_PARAM, 
                                        target_volume_fraction=config_params.VOLUME_FRACTION, 
                                        num_fibers=config_params.NUM_FIBERS,
                                        dist_shape=config_params.DISTRIBUTION_SHAPE,
                                        mean_diameter=config_params.MEAN_DIAMETER, 
                                        sigma_radii=config_params.SIGMA_DIAMETER,
                                        space_buffer=config_params.SPACE_BUFFER_STARTS_ENDS,
                                        box_length_init=config_params.BOX_LENGTH)

            data_folder = os.path.join(config_params.SUBSTRATE_OUTPUT_FOLDER_PATH,'data')
            if not os.path.exists(data_folder):
                        os.makedirs(data_folder)

            init2d_data_folder = os.path.join(config_params.SUBSTRATE_OUTPUT_FOLDER_PATH,'figs','init2D')
            if not os.path.exists(init2d_data_folder):
                        os.makedirs(init2d_data_folder)

            # save data with Pickle format
            init2d_data_file_name = os.path.join(init2d_data_folder ,'init2d.pkl')
            util.save_data_array_to_pickle(init2d_data_file_name, initialization2D.initial_positions, 
                                    initialization2D.box_length.cpu().item()) 

            initialization2D.plot_PBC()
            # -------- Sphere-Based Meshing of Axons --------
            meshing = Meshing(initialization2D, config_params.SPHERE_SPACING, config_params.BEAD_SPACING_MEAN, config_params.BEAD_SPACING_STDV, device)
            # -------- Geometric Optimization --------
            substrate = GeometricOptimization(device, meshing, config_params.SPHERE_SPACING, 
                                                config_params.W_OVERLAP, config_params.W_CURVE, config_params.W_LENGTH, \
                                                config_params.SPACE_BUFFER_REPULSE, initialization2D.mean_d_underlying, initialization2D.sigma_d_underlying,\
                                                config_params.BEAD_SPACING_MEAN, config_params.BEAD_SPACING_STDV)
            not_converged = not substrate.optimized

        # --- Check if VF refinement is needed ---
        if target_3d_vf is not None and substrate.optimized:
            measured_vf = config_params.VOLUME_FRACTION  # overwritten by get_volume_fraction()
            if abs(measured_vf - target_3d_vf) <= vf_tolerance:
                print(f"VF refinement converged: measured={measured_vf:.3f}, target={target_3d_vf:.3f}")
                break
            prev_vf_2d = current_vf_2d
            current_vf_2d = current_vf_2d * (target_3d_vf / measured_vf)
            print(f"VF refinement: measured={measured_vf:.3f}, target={target_3d_vf:.3f}, "
                  f"scaling VF_2D by {target_3d_vf / measured_vf:.4f}")
        else:
            break

    if (substrate.optimized == True):

        # get the execution time
        et = time.time()
        elapsed_time = et - st
        
        file_name_root = 'array'+str(config_params.NUM_FIBERS) +'_fibers_boxL_'+str(config_params.BOX_LENGTH.item())+'_'+str(config_params.EXP_DATE_TIME) + \
                        '_avf_'+str(config_params.VOLUME_FRACTION)+\
                        '_d'+str(config_params.MEAN_DIAMETER)+'_sig'+str(config_params.SIGMA_DIAMETER)+\
                            'wo'+str(config_params.W_OVERLAP)+'_wc'+str(config_params.W_CURVE)+'_wl'+str(config_params.W_LENGTH)+'_K'+str(int(config_params.ORIENTATION_SHAPE_PARAM))+\
                            '_ODI_'+str(config_params.ODI_INDEX)+'_'+\
                                str(round(elapsed_time,2))+'_sec'+'.pkl'
        
        data_folder = os.path.join(config_params.SUBSTRATE_OUTPUT_FOLDER_PATH,'data')
        if not os.path.exists(data_folder):
                    os.makedirs(data_folder)

        # # save data with Pickle format
        data_file_name = os.path.join(data_folder, file_name_root)
        util.save_data_array_to_pickle(data_file_name, substrate.optimized_fibers.cpu(), 
                                initialization2D.box_length.cpu().item()) 
        # Plot
        fiber_list = util.map_matrix_to_list_numpy(substrate.optimized_fibers)
        color = cm.rainbow(np.linspace(0.0, 1.0, len(fiber_list)))
        np.random.shuffle(color)
        '''Choose FOV to plot, can include 3D animation GIF'''
        fiber_3D_plot.plot_fibers(substrate.optimized_fibers, overlap_indices=None, color=color, optimized=True, POV='horizontal_90') 
        # fiber_3D_plot.plot_fibers(substrate.optimized_fibers, overlap_indices=None, color=color, animation_input=False, optimized=True)
        # fiber_3D_plot.plot_fibers(substrate.optimized_fibers, overlap_indices=None, color=color, optimized=True, POV='horizontal_0')                    
        # fiber_3D_plot.plot_fibers(substrate.optimized_fibers, overlap_indices=None, color=color, optimized=True, POV='horizontal_90', animation_input=True)       
        
        along_fiber_plot.plot_along_axon_radius_variation(substrate.optimized_fibers, colors=color)
        along_fiber_plot.plot_diameter_CV_distribution()
        along_fiber_plot.plot_diameter_GEV_distribution(substrate.optimized_fibers)
        orientation_plot.plot_along_axon_OD(substrate.optimized_fibers, optimized=True)
        # Arc-length-based OD/FOD + Watson-kappa fit (Callaghan/ConFiG-style
        # substrate validation). Saved with an "_arclength" suffix alongside
        # the original OD/FOD plots for side-by-side comparison.
        orientation_plot.plot_along_axon_OD_arclength(substrate.optimized_fibers, optimized=True)
        
        print("====== Substrate generated in folder:"+ config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+ " ======")
        print("\n")
    else:
        print("Exited unoptimized")
