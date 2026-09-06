from simulation_toolkit.substrate_generator.initialization_2d import Init2D
from simulation_toolkit.substrate_generator.meshing import Meshing
from simulation_toolkit.substrate_generator.geometric_optimization import GeometricOptimization
from simulation_toolkit.substrate_generator.myelin import generate_inner_fibers
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
from simulation_toolkit.utils import cross_section_plot


def _diffusion_length_box(D0=2.5, t=100.0):
    """Minimum box side (um) to satisfy the 2-sigma diffusion-length rule.

    sigma = sqrt(2*D0*t); box >= 2*sigma, rounded up. Defaults: free-water
    D0=2.5 um^2/ms over t=100 ms -> 45 um.
    """
    return float(np.ceil(2.0 * np.sqrt(2.0 * D0 * t)))


def estimate_initial_vf(target_3d_vf, kappa, mean_diameter,
                        bead_alpha_mean, bead_alpha_stdv,
                        bead_spacing_mean, bead_spacing_stdv,
                        space_buffer=None,
                        num_watson_samples=2000, num_bead_realizations=200):
    """
    Estimate the 2D initial volume fraction needed to achieve a target 3D volume fraction.

    The 3D VF amplifies relative to the 2D VF by two factors:
      VF_3D ≈ (VF_2D / A_buffer) x A_path x A_bead

    A_path:   helix arc length / Lz, averaged over Watson(K) fiber directions.
    A_bead:   mean(r_beaded² / r₀²), capturing beading-induced volume inflation.
    A_buffer: (r₀ + b/2)² / r₀², corrects for the 2D packing using buffered radii
              while 3D VF uses real radii.

    Returns the estimated VF_2D = VF_3D_target x A_buffer / (A_path x A_bead).
    """
    # --- A_buffer: space buffer correction ---
    if space_buffer is None:
        space_buffer = 0.137 * mean_diameter
    r0 = mean_diameter / 2
    A_buffer = ((r0 + space_buffer / 2) / r0) ** 2
    # --- A_path: helix arc length amplification ---
    mu_dir = np.array([0, 0, 1])
    watson = wd.WatsonDistribution(mu_dir, kappa)
    directions = watson.sample(num_watson_samples, dz_floor=config_params.ORIENTATION_DZ_FLOOR)
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


def substrate_main(params, experiment_folder, folder_suffix=""):
    # Validate parameters before proceeding
    validate_parameters(params)
    config_params.ORIENTATION_SHAPE_PARAM = params['orientation_shape_parameter']
    config_params.BOX_LENGTH = params['box_length_init']

    # --- z-height (Lz): thin, decoupled from the in-plane box (Lx=Ly) ---
    _box_length_z_init = params.get('box_length_z_init', 0)
    if _box_length_z_init and _box_length_z_init > 0:
        config_params.BOX_LENGTH_Z = float(_box_length_z_init)
    else:
        # Diffusion-length rule: Lz >= 2*sigma_z, sigma_z = sqrt(2*D0*t).
        config_params.BOX_LENGTH_Z = _diffusion_length_box()
    print(f"Box z-height Lz = {config_params.BOX_LENGTH_Z} um")

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
    g_ratio = params.get('g_ratio')
    inner_sphere_spacing_ratio = params.get('inner_sphere_spacing_ratio', 0.5)

    print(f"Building substrate with: \
        volume_fraction={config_params.VOLUME_FRACTION},\
        diameter_mean={config_params.MEAN_DIAMETER},\
        orientation_shape_parameter={config_params.ORIENTATION_SHAPE_PARAM}, \
        bead_alpha_mean={config_params.BEAD_ALPHA_MEAN},\
        num_fibers={config_params.NUM_FIBERS}") 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config_params.EXP_DATE_TIME = util.get_date_time()        
    config_params.ODI_INDEX = np.round(2/np.pi * np.arctan(1/config_params.ORIENTATION_SHAPE_PARAM), 4)

    # Ensure the auto-derived in-plane box (Lx=Ly) meets the same diffusion-length
    # floor as Lz by scaling the fiber count up. box_length ~ sqrt(N), so this is a
    # deterministic pre-estimate; the in-loop guard enforces it exactly afterwards.
    min_inplane_box = _diffusion_length_box()
    if params['box_length_init'] == 0:
        _r_eff = config_params.MEAN_DIAMETER/2 + config_params.SPACE_BUFFER_STARTS_ENDS/2
        for _ in range(4):
            _est_area = 2 * config_params.NUM_FIBERS * np.pi * _r_eff**2
            _est_box = np.sqrt(_est_area / max(config_params.VOLUME_FRACTION, 1e-6))
            if _est_box >= min_inplane_box:
                break
            _n_new = int(np.ceil(config_params.NUM_FIBERS * (min_inplane_box/_est_box)**2 * 1.05))
            print(f"Auto-increasing N {config_params.NUM_FIBERS} -> {_n_new} so in-plane box >= {min_inplane_box:.0f} um (est {_est_box:.1f} um).")
            config_params.NUM_FIBERS = _n_new

    vf_label = target_3d_vf if target_3d_vf is not None else config_params.VOLUME_FRACTION
    combo_folder = ('d'+str(config_params.MEAN_DIAMETER)+
                    '_K'+f"{config_params.ORIENTATION_SHAPE_PARAM:g}"+
                    '_ODI_'+str(config_params.ODI_INDEX)+
                    '_bead_'+str(config_params.BEAD_ALPHA_MEAN)+
                    '_VF_'+str(vf_label))
    config_params.SUBSTRATE_OUTPUT_FOLDER_PATH = os.path.join(config_params.OUTPUT_FOLDER_PATH, experiment_folder, \
                                                       combo_folder, \
                                                       str(config_params.EXP_DATE_TIME)+\
                                                        '_d'+str(config_params.MEAN_DIAMETER)+\
                                                        '_K'+f"{config_params.ORIENTATION_SHAPE_PARAM:g}"+\
                                                    '_ODI_'+str(config_params.ODI_INDEX)+\
                                                    '_bead_'+str(config_params.BEAD_ALPHA_MEAN)+'_'+\
                                                        str(config_params.NUM_FIBERS) +'fibers'+str(folder_suffix))

    # --- Iterative VF refinement loop ---
    # A single attempt = one Init2D + Meshing + GeometricOptimization pass.
    # Each attempt is consumed regardless of outcome:
    #   - If geometry fails to converge (overlap optimizer gives up): scale
    #     VF_2D *down* by ``vf_backoff_factor`` and retry, because the requested
    #     packing is likely infeasible at this VF_2D.
    #   - If geometry converges but measured 3D VF is off target by more than
    #     ``vf_tolerance``: scale VF_2D by (target / measured) and retry.
    # This prevents the previous behavior of looping forever on the same
    # (infeasible) VF_2D when the optimizer never reaches 0 overlaps.
    vf_tolerance = 0.04
    vf_backoff_factor = 0.9
    max_vf_refinements = int(params.get('max_vf_refinements', 10))
    current_vf_2d = config_params.VOLUME_FRACTION
    prev_vf_2d = current_vf_2d

    for vf_iter in range(max_vf_refinements):
        if vf_iter > 0:
            print(f"\n--- VF refinement iteration {vf_iter + 1}: adjusting VF_2D from {prev_vf_2d:.4f} to {current_vf_2d:.4f} ---")
            config_params.BOX_LENGTH = params['box_length_init']  # reset box length for recalculation
        config_params.VOLUME_FRACTION = current_vf_2d
        st = time.time()
        # -------- 2D initialization of axon start/end points --------
        # Build 2D init; if the auto-derived in-plane box (Lx=Ly) falls below the
        # diffusion-length floor, increase N and regenerate (cap 3 retries).
        for _box_try in range(4):
            initialization2D = Init2D(  device=device,
                                        date_time=config_params.EXP_DATE_TIME,
                                        orientation_shape_parameter=config_params.ORIENTATION_SHAPE_PARAM,
                                        target_volume_fraction=config_params.VOLUME_FRACTION,
                                        num_fibers=config_params.NUM_FIBERS,
                                        dist_shape=config_params.DISTRIBUTION_SHAPE,
                                        mean_diameter=config_params.MEAN_DIAMETER,
                                        sigma_radii=config_params.SIGMA_DIAMETER,
                                        space_buffer=config_params.SPACE_BUFFER_STARTS_ENDS,
                                        box_length_init=config_params.BOX_LENGTH,
                                        box_length_z=config_params.BOX_LENGTH_Z)
            _box_xy = float(initialization2D.box_length.cpu().item())
            if params['box_length_init'] != 0 or _box_xy >= min_inplane_box or _box_try == 3:
                break
            _n_new = int(np.ceil(config_params.NUM_FIBERS * (min_inplane_box/_box_xy)**2 * 1.05))
            print(f"In-plane box {_box_xy:.1f} < {min_inplane_box:.0f} um; auto-increasing N {config_params.NUM_FIBERS} -> {_n_new} and regenerating.")
            config_params.NUM_FIBERS = _n_new

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

        # --- Decide next action based on outcome ---
        if not substrate.optimized:
            # Geometry optimizer gave up (still has overlaps after max iters).
            # Back off VF_2D and try again -- the requested packing is likely
            # infeasible at this VF_2D for this combo of diameter/kappa/beading.
            prev_vf_2d = current_vf_2d
            current_vf_2d = current_vf_2d * vf_backoff_factor
            print(f"VF refinement: geometry did not converge at VF_2D={prev_vf_2d:.4f}; "
                  f"backing off to VF_2D={current_vf_2d:.4f} (x{vf_backoff_factor}).")
            continue

        # Converged. If no 3D VF target, we're done.
        if target_3d_vf is None:
            break

        measured_vf = config_params.VOLUME_FRACTION  # overwritten by get_volume_fraction()
        if abs(measured_vf - target_3d_vf) <= vf_tolerance:
            print(f"VF refinement converged: measured={measured_vf:.3f}, target={target_3d_vf:.3f}")
            break
        prev_vf_2d = current_vf_2d
        current_vf_2d = current_vf_2d * (target_3d_vf / measured_vf)
        print(f"VF refinement: measured={measured_vf:.3f}, target={target_3d_vf:.3f}, "
              f"scaling VF_2D by {target_3d_vf / measured_vf:.4f}")

    if (substrate.optimized == True):

        # get the execution time
        et = time.time()
        elapsed_time = et - st
        
        file_name_root = 'array'+str(config_params.NUM_FIBERS) +'_fibers_boxL_'+str(config_params.BOX_LENGTH.item())+'_'+str(config_params.EXP_DATE_TIME) + \
                        '_avf_'+str(config_params.VOLUME_FRACTION)+\
                        '_d'+str(config_params.MEAN_DIAMETER)+'_sig'+str(config_params.SIGMA_DIAMETER)+\
                            'wo'+str(config_params.W_OVERLAP)+'_wc'+str(config_params.W_CURVE)+'_wl'+str(config_params.W_LENGTH)+'_K'+f"{config_params.ORIENTATION_SHAPE_PARAM:g}"+\
                            '_ODI_'+str(config_params.ODI_INDEX)+'_'+\
                                str(round(elapsed_time,2))+'_sec'+'.pkl'
        
        data_folder = os.path.join(config_params.SUBSTRATE_OUTPUT_FOLDER_PATH,'data')
        if not os.path.exists(data_folder):
                    os.makedirs(data_folder)

        # # save data with Pickle format
        data_file_name = os.path.join(data_folder, file_name_root)
        # z-height (thin box) and realized fiber count (== unique fiber ids, the
        # ground truth after any auto-N bump) recorded with the substrate.
        _lz = float(config_params.BOX_LENGTH_Z)
        _fid_col = util.fiber_id_column(substrate.optimized_fibers)
        _num_fibers = int(torch.unique(substrate.optimized_fibers[:, _fid_col]).numel())
        print(f"Final fiber count (realized) = {_num_fibers}; Lx=Ly={initialization2D.box_length.cpu().item()} um, Lz={_lz} um")
        inner_fibers = None
        if g_ratio is not None:
            inner_fibers = generate_inner_fibers(
                substrate.optimized_fibers,
                g_ratio=g_ratio,
                inner_sphere_spacing_ratio=inner_sphere_spacing_ratio,
                box_length=initialization2D.box_length.cpu().item(),
                box_length_z=_lz,
            )
            util.save_myelinated_substrate_to_pickle(
                data_file_name,
                substrate.optimized_fibers.cpu(),
                inner_fibers,
                initialization2D.box_length.cpu().item(),
                g_ratio,
                inner_sphere_spacing_ratio,
                lz=_lz,
                num_fibers=_num_fibers,
            )
            print(f"Generated myelin geometry: outer_spheres={substrate.optimized_fibers.shape[0]}, inner_spheres={inner_fibers.shape[0]}, g_ratio={g_ratio}")
        else:
            util.save_data_array_to_pickle(data_file_name, substrate.optimized_fibers.cpu(), 
                                    initialization2D.box_length.cpu().item(),
                                    lz=_lz, num_fibers=_num_fibers) 
        # Plot
        fiber_list = util.map_matrix_to_list_numpy(substrate.optimized_fibers)
        color = cm.rainbow(np.linspace(0.0, 1.0, len(fiber_list)))
        np.random.shuffle(color)
        # -------- Cross-section slice at mid-box (z=0) --------
        cross_section_plot.plot_cross_section_z_mid(
            outer_fibers=substrate.optimized_fibers,
            inner_fibers=inner_fibers,
            box_length=initialization2D.box_length.cpu().item(),
            box_length_z=initialization2D.box_length_z.cpu().item(),
            color=color,
        )
        '''Choose FOV to plot, can include 3D animation GIF'''
        fiber_3D_plot.plot_fibers(substrate.optimized_fibers, overlap_indices=None, color=color, optimized=True, POV='horizontal_90', component_label='outer')
        # fiber_3D_plot.plot_myelinated_fibers(substrate.optimized_fibers, inner_fibers, color=color, POV='horizontal_90')
        # if inner_fibers is not None:
        #     fiber_3D_plot.plot_fibers(inner_fibers, overlap_indices=None, color=color, optimized=True, POV='horizontal_90', component_label='inner')
        # fiber_3D_plot.plot_fibers(substrate.optimized_fibers, overlap_indices=None, color=color, animation_input=False, optimized=True)

        # Shared axis limits (data-driven) so this substrate's CV / diameter
        # histograms match the rest of the batch when a limits JSON is provided.
        along_fiber_plot.load_shared_axis_limits(
            getattr(config_params, "SUBSTRATE_STATS_AXIS_LIMITS_FILE", None))

        along_fiber_plot.plot_along_axon_radius_variation(substrate.optimized_fibers, colors=color, component_label='outer')
        along_fiber_plot.plot_diameter_CV_distribution(component_label='outer')
        _diam_outer = along_fiber_plot.extract_radius_all(substrate.optimized_fibers) * 2
        _x_max_diam = float(np.max(_diam_outer)) if len(_diam_outer) > 0 else None
        if inner_fibers is not None:
            _diam_inner = along_fiber_plot.extract_radius_all(inner_fibers) * 2
            if len(_diam_inner) > 0:
                _x_max_diam = max(_x_max_diam, float(np.max(_diam_inner)))
        along_fiber_plot.plot_diameter_GEV_distribution(substrate.optimized_fibers, component_label='outer', x_max=_x_max_diam)
        if inner_fibers is not None:
            along_fiber_plot.plot_along_axon_radius_variation(inner_fibers, colors=color, component_label='inner')
            along_fiber_plot.plot_diameter_CV_distribution(component_label='inner')
            along_fiber_plot.plot_diameter_GEV_distribution(inner_fibers, component_label='inner', x_max=_x_max_diam)
        # Effective-diameter stats JSON is expensive (fine-grid slicing); it is
        # backfilled separately via simulation_engine.helper.reprocess_diameter_stats.
        # along_fiber_plot.save_effective_axon_diameter_stats_from_pickle(data_file_name)
        # # orientation_plot.plot_along_axon_OD(substrate.optimized_fibers, optimized=True, component_label='outer')
        # # if inner_fibers is not None:
        #     # orientation_plot.plot_along_axon_OD(inner_fibers, optimized=True, component_label='inner')
        # # Arc-length-based OD/FOD + Watson-kappa fit (Callaghan/ConFiG-style
        # # substrate validation). Saved with an "_arclength" suffix alongside
        # # the original OD/FOD plots for side-by-side comparison.
        # orientation_plot.plot_along_axon_OD_arclength(substrate.optimized_fibers, optimized=True, component_label='outer')
        # if inner_fibers is not None:
        #     orientation_plot.plot_along_axon_OD_arclength(inner_fibers, optimized=True, component_label='inner')

        # Global (end-to-end) orientation statistics: writes the global OD
        # histogram and the analytic Watson FOD glyph into figs/substrate_stats/ODI.
        orientation_plot.plot_global_axon_OD(substrate.optimized_fibers, optimized=True)

        print("====== Substrate generated in folder:"+ config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+ " ======")
        print("\n")
    else:
        # Raise so the batch driver (or any caller) marks this job as failed
        # instead of silently producing no output.
        raise RuntimeError(
            f"Substrate generation did not converge after {max_vf_refinements} VF refinement "
            f"attempt(s). Last VF_2D={current_vf_2d:.4f}, target_3d_vf={target_3d_vf}, "
            f"d={config_params.MEAN_DIAMETER}, K={int(config_params.ORIENTATION_SHAPE_PARAM)}, "
            f"bead_alpha={config_params.BEAD_ALPHA_MEAN}."
        )
