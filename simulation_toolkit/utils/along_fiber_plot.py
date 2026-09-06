import matplotlib.pyplot as plt
import simulation_toolkit.utils.common_utils as util
import numpy as np
from scipy.stats import genextreme
import torch
import simulation_toolkit.toolkit_params as config_params
import os
import warnings
import json

def _component_file_tag(component_label):
    return component_label.lower().replace(" ", "_")


def _component_title(component_label):
    return component_label.capitalize()


# Fallback axis ranges used when no data-driven shared limits are configured on
# config_params. cli.reprocess_substrate_stats scans the whole dataset and sets
# the config globals so every substrate is drawn on identical axes.
_CV_DIAMETER_XLIM_DEFAULT = (0.0, 0.8)
_CV_DIAMETER_YLIM_DEFAULT = (0.0, 12.0)
_DIAMETER_DIST_XLIM_DEFAULT = (0.0, 20.0)
_DIAMETER_DIST_YLIM_DEFAULT = (0.0, 1.0)


def _resolve_limit(value, default):
    """Return a validated (min, max) tuple from a config global, else default."""
    if value is None:
        return default
    try:
        lo, hi = float(value[0]), float(value[1])
        if hi > lo:
            return (lo, hi)
    except (TypeError, ValueError, IndexError):
        pass
    return default


def compute_cv_values(optimized_fibers):
    """Per-axon CV of radius, matching plot_along_axon_radius_variation.

    Iterates every stored chain (PBC image chains included, no dedup) so the
    returned array equals config_params.CV_RADII used by the CV histogram.
    """
    fiber_list = util.map_matrix_to_list_numpy(optimized_fibers)
    cv = np.empty(0)
    for fiber in fiber_list:
        cv = np.concatenate(
            (cv, get_cv_of_diameter_or_radius_along_each_axon(fiber[:, 3])))
    return cv


_AXIS_LIMIT_KEYS = (
    "CV_DIAMETER_XLIM", "CV_DIAMETER_YLIM",
    "DIAMETER_DIST_XLIM", "DIAMETER_DIST_YLIM",
)


def load_shared_axis_limits(json_path):
    """Load a shared-axis-limits JSON and set the config_params globals.

    Returns the loaded dict, or None when the file is missing/unreadable so
    callers transparently fall back to the per-plot default ranges.
    """
    if not json_path or not os.path.exists(json_path):
        return None
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    for key in _AXIS_LIMIT_KEYS:
        val = data.get(key)
        if val is not None:
            setattr(config_params, key, (float(val[0]), float(val[1])))
    return data


def save_shared_axis_limits(json_path, limits):
    """Write the shared-axis-limits dict to ``json_path`` (creating parents)."""
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(limits, f, indent=4)
    return json_path


def _split_stored_axon_segments(fibers, box_length_z, tol=1e-4):
    """Split a stored substrate array into axon-chain segments.

    The substrate writer stores each chain from z=-Lz/2 to z=+Lz/2, and PBC
    image chains are stored as additional chains. We intentionally keep all
    stored chains here; no canonical filtering is applied.

    ``box_length_z`` is the z-extent (Lz); for thin-z anisotropic substrates it
    differs from the in-plane box length, so the split threshold must use Lz.
    """
    fibers = _geometry_to_numpy(fibers)
    if fibers.shape[0] == 0:
        return []
    half = box_length_z / 2
    segments = []
    start_idx = 0
    for idx in range(fibers.shape[0]):
        if np.isclose(fibers[idx, 2], half, atol=tol):
            segment = fibers[start_idx:idx + 1]
            if segment.shape[0] >= 2:
                segments.append(segment)
            start_idx = idx + 1
    if start_idx < fibers.shape[0]:
        segment = fibers[start_idx:]
        if segment.shape[0] >= 2:
            segments.append(segment)
    return segments


def _geometry_to_numpy(fibers):
    if isinstance(fibers, torch.Tensor):
        return fibers.detach().cpu().numpy()
    return np.asarray(fibers)


def collect_stored_axons_from_substrate_pickle(substrate_file):
    """Load a substrate pickle and return all stored axon-chain segments."""
    substrate = util.load_substrate_geometry(substrate_file)
    # Chains terminate at z = +/- Lz/2; thin-z substrates have Lz != box_length.
    lz = substrate.lz if substrate.lz is not None else substrate.box_length
    components = {
        "outer": _split_stored_axon_segments(substrate.outer_fibers, lz),
    }
    if substrate.inner_fibers is not None:
        components["inner"] = _split_stored_axon_segments(substrate.inner_fibers, lz)

    return {
        "substrate_file": substrate_file,
        "box_length": substrate.box_length,
        "is_myelinated": substrate.is_myelinated,
        "g_ratio": substrate.g_ratio,
        "inner_sphere_spacing_ratio": substrate.inner_sphere_spacing_ratio,
        "components": components,
        "component_segment_counts": {
            label: len(segments) for label, segments in components.items()
        },
    }


def _local_frame_from_endpoints(points):
    axis = points[-1] - points[0]
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 0:
        return None
    axis = axis / axis_norm
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, reference)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return axis, e1, e2, axis_norm


def _estimate_union_disk_area(disk_centers, disk_radii, grid_resolution_um):
    if len(disk_radii) == 0:
        return 0.0
    # Fast path: single disk — area is exact, no raster needed.
    if len(disk_radii) == 1:
        return float(np.pi * disk_radii[0] ** 2)
    min_u = np.min(disk_centers[:, 0] - disk_radii)
    max_u = np.max(disk_centers[:, 0] + disk_radii)
    min_v = np.min(disk_centers[:, 1] - disk_radii)
    max_v = np.max(disk_centers[:, 1] + disk_radii)
    if not np.isfinite([min_u, max_u, min_v, max_v]).all():
        return 0.0

    u = np.arange(min_u, max_u + grid_resolution_um, grid_resolution_um)
    v = np.arange(min_v, max_v + grid_resolution_um, grid_resolution_um)
    if u.size == 0 or v.size == 0:
        return 0.0
    uu, vv = np.meshgrid(u, v, indexing="xy")
    occupied = np.zeros(uu.shape, dtype=bool)
    for center, radius in zip(disk_centers, disk_radii):
        occupied |= ((uu - center[0]) ** 2 + (vv - center[1]) ** 2) <= radius ** 2
    return float(np.count_nonzero(occupied) * grid_resolution_um ** 2)


def _sample_effective_radii_for_axon(
        segment,
        slice_step_um,
        grid_resolution_um,
        slice_step_radius_fraction=None,
):
    """
    Parameters
    ----------
    slice_step_radius_fraction
        If given, overrides ``slice_step_um`` with a value computed as
        ``fraction * mean_sphere_radius_of_segment``.  The step is clamped
        to ``[grid_resolution_um * 2, mean_radius * 2]`` to keep accuracy
        reasonable regardless of axon size.
    """
    segment = _geometry_to_numpy(segment).astype(np.float64, copy=False)
    points = segment[:, :3]
    radii = segment[:, 3]
    frame = _local_frame_from_endpoints(points)
    if frame is None:
        return np.array([], dtype=np.float64)
    axis, e1, e2, axis_length = frame

    # Dynamic slice step: scale with axon size.
    if slice_step_radius_fraction is not None and radii.size > 0:
        mean_r = float(np.mean(radii))
        step = slice_step_radius_fraction * mean_r
        # Clamp lower bound: at least 3× the raster pixel so the raster area
        # estimate is not dominated by aliasing of tiny disks.
        # Clamp upper bound: at most one full sphere diameter so each sphere
        # contributes at least one slice to short segments.
        step = float(np.clip(step, grid_resolution_um * 3.0, mean_r * 2.0))
        slice_step_um = step

    rel = points - points[0]
    sphere_axis_pos = rel @ axis
    sphere_u = rel @ e1
    sphere_v = rel @ e2
    slice_positions = np.arange(0.0, axis_length + 0.5 * slice_step_um, slice_step_um)

    effective_radii = []
    for slice_pos in slice_positions:
        delta_axis = slice_pos - sphere_axis_pos
        intersect_mask = np.abs(delta_axis) <= radii
        if not np.any(intersect_mask):
            continue
        disk_radii = np.sqrt(np.maximum(radii[intersect_mask] ** 2 - delta_axis[intersect_mask] ** 2, 0.0))
        disk_centers = np.column_stack((sphere_u[intersect_mask], sphere_v[intersect_mask]))
        area = _estimate_union_disk_area(disk_centers, disk_radii, grid_resolution_um)
        if area > 0:
            effective_radii.append(np.sqrt(area / np.pi))
    return np.asarray(effective_radii, dtype=np.float64)


def _moment_sums(radii):
    return {f"r{power}": float(np.sum(radii ** power)) for power in (1, 2, 3, 4, 6)}


def _metric_summary(radii):
    radii = np.asarray(radii, dtype=np.float64)
    radii = radii[np.isfinite(radii) & (radii > 0)]
    if radii.size == 0:
        return {
            "num_slices": 0,
            "mean_radius_um": None,
            "mean_diameter_um": None,
            "moments": {f"r{power}": 0.0 for power in (1, 2, 3, 4, 6)},
            "d_eff_p3_q2_um": None,
            "r_app_wide_pulse_um": None,
            "d_app_wide_pulse_um": None,
            "r_app_internal_um": None,
            "d_app_internal_um": None,
        }

    moments = _moment_sums(radii)
    sum_r2 = moments["r2"]
    d_eff = 2.0 * moments["r3"] / sum_r2 if sum_r2 > 0 else None
    r_app_wp = (moments["r6"] / sum_r2) ** 0.25 if sum_r2 > 0 else None
    r_app_internal = np.sqrt(moments["r4"] / sum_r2) if sum_r2 > 0 else None
    return {
        "num_slices": int(radii.size),
        "mean_radius_um": float(np.mean(radii)),
        "mean_diameter_um": float(2.0 * np.mean(radii)),
        "moments": moments,
        "d_eff_p3_q2_um": None if d_eff is None else float(d_eff),
        "r_app_wide_pulse_um": None if r_app_wp is None else float(r_app_wp),
        "d_app_wide_pulse_um": None if r_app_wp is None else float(2.0 * r_app_wp),
        "r_app_internal_um": None if r_app_internal is None else float(r_app_internal),
        "d_app_internal_um": None if r_app_internal is None else float(2.0 * r_app_internal),
    }


def compute_effective_axon_diameter_stats(
        axon_segments,
        component_label="outer",
        slice_step_um=0.05,
        grid_resolution_um=0.01,
        slice_step_radius_fraction=None,
):
    """
    Parameters
    ----------
    slice_step_radius_fraction
        When set, each axon segment uses a dynamic slice step equal to
        ``fraction * mean_sphere_radius`` of that segment, clamped to a
        sensible range.  Overrides ``slice_step_um``.
        Recommended value: ``0.1`` (10 % of the mean sphere radius).
    """
    bundle_radii = []
    for segment in axon_segments:
        sampled_radii = _sample_effective_radii_for_axon(
            segment, slice_step_um, grid_resolution_um,
            slice_step_radius_fraction=slice_step_radius_fraction,
        )
        bundle_radii.append(sampled_radii)

    if bundle_radii:
        bundle_radii = np.concatenate(bundle_radii)
    else:
        bundle_radii = np.array([], dtype=np.float64)

    method_tag = (
        f"dynamic_slice_frac{slice_step_radius_fraction}"
        if slice_step_radius_fraction is not None
        else "fixed_slice_step"
    )
    return {
        "component_label": component_label,
        "method": f"endpoint_axis_{method_tag}_raster_union_area",
        "slice_step_um": float(slice_step_um) if slice_step_radius_fraction is None else None,
        "slice_step_radius_fraction": slice_step_radius_fraction,
        "grid_resolution_um": float(grid_resolution_um),
        "num_axons": int(len(axon_segments)),
        "bundle": _metric_summary(bundle_radii),
    }


def save_effective_axon_diameter_stats_from_pickle(
        substrate_file,
        output_folder=None,
        slice_step_um=0.05,
        grid_resolution_um=0.01,
        slice_step_radius_fraction=None,
):
    collection = collect_stored_axons_from_substrate_pickle(substrate_file)
    if output_folder is None:
        output_folder = os.path.join(config_params.SUBSTRATE_OUTPUT_FOLDER_PATH, "figs", "substrate_stats")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    saved = {}
    for component_label, axon_segments in collection["components"].items():
        component_tag = _component_file_tag(component_label)
        stats = compute_effective_axon_diameter_stats(
            axon_segments,
            component_label=component_label,
            slice_step_um=slice_step_um,
            grid_resolution_um=grid_resolution_um,
            slice_step_radius_fraction=slice_step_radius_fraction,
        )
        stats["substrate"] = {
            "substrate_file": collection["substrate_file"],
            "box_length_um": collection["box_length"],
            "is_myelinated": collection["is_myelinated"],
            "g_ratio": collection["g_ratio"],
            "inner_sphere_spacing_ratio": collection["inner_sphere_spacing_ratio"],
            "stored_segment_counts": collection["component_segment_counts"],
        }
        output_path = os.path.join(
            output_folder,
            f"{component_tag}_effective_axon_diameter_stats_{config_params.EXP_DATE_TIME}.json",
        )
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=4)
        saved[component_label] = output_path
    return saved


def plot_diameter_GEV_distribution(optimized_fibers, component_label="outer", x_max=None):
    diameter = extract_radius_all(optimized_fibers)*2
    if len(diameter) == 0:
        print(f"Warning: plot_diameter_GEV_distribution: no radius data for component '{component_label}'; skipping.")
        return
    xlim = _resolve_limit(getattr(config_params, "DIAMETER_DIST_XLIM", None), _DIAMETER_DIST_XLIM_DEFAULT)
    ylim = _resolve_limit(getattr(config_params, "DIAMETER_DIST_YLIM", None), _DIAMETER_DIST_YLIM_DEFAULT)
    fig = plt.figure()
    nbins=20
    # Fixed hist range so bin widths (and hence densities) match across substrates.
    plt.hist(diameter, bins=nbins, range=xlim, density=True, align='mid', label='Substrate diameter')
    
    x = np.linspace(min(diameter), max(diameter), 10000)
    
    # Suppress runtime warnings during GEV fitting and stats calculation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            shape_gev_fitted, loc_gev_fitted, scale_gev_fitted = genextreme.fit(diameter)

            meanGEV, varGEV, skew, kurt = genextreme.stats(c=shape_gev_fitted, loc=loc_gev_fitted, 
                                                          scale=scale_gev_fitted, moments='mvsk')     
            
            stdvGEV = np.sqrt(varGEV)
            config_params.GEV_DIAMETER_MEAN, config_params.GEV_DIAMETER_STDV = np.round(meanGEV, 3), np.round(stdvGEV, 3)
            plt.plot(x, genextreme.pdf(x, shape_gev_fitted, loc_gev_fitted, scale_gev_fitted), 'r-', lw=2, label='Fitted GEV')
            
        except (ValueError, np.linalg.LinAlgError):
            # Fallback to normal distribution if GEV fitting fails
            mean_diameter = np.mean(diameter)
            std_diameter = np.std(diameter)
            config_params.GEV_DIAMETER_MEAN, config_params.GEV_DIAMETER_STDV = np.round(mean_diameter, 3), np.round(std_diameter, 3)
            from scipy.stats import norm
            plt.plot(x, norm.pdf(x, mean_diameter, std_diameter), 'r-', lw=2, label='Fitted Normal Distribution')
    
    # Adding titles and labels   
    component_title = _component_title(component_label)
    component_tag = _component_file_tag(component_label)
    plt.title(f'Distribution of {component_label} diameter', fontsize=17, pad=20)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.xlabel(f'{component_title} Diameter (µm)', fontsize=15, labelpad=5)
    plt.ylabel('Density', fontsize=15, labelpad=5)
    # Shared/fixed axes (x_max kept for backward-compat; shared limits take precedence).
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend()
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.5)
    folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/substrate_stats"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path+"/"+component_tag+"_Diameter_distribution_mean"+str(config_params.GEV_DIAMETER_MEAN)+'_std'+str(config_params.GEV_DIAMETER_STDV)+'_'+config_params.EXP_DATE_TIME+".png",
                 dpi=500)
    plt.close(fig)

def extract_radius_all(optimized_fibers):
    fiber_list_xyz_r_fid = util.map_matrix_to_list_numpy(optimized_fibers)
    r_values = np.array([])
    seen_ids = set() # filter out XY-wrapped fibers
    for fiber_array in fiber_list_xyz_r_fid:
        rad, fid = fiber_array[:, 3], fiber_array[:, 4][0]
        if fid not in seen_ids:
            r_values = np.concatenate((r_values, rad.flatten()))
            seen_ids.add(fid)
    return r_values
    
def plot_along_axon_radius_variation(optimized_fibers, colors, component_label="outer"):
    '''
    loop through each axon
    get length along axon
    get radius value
    '''
    cv_of_radii = np.empty(0)

    fig = plt.figure()
    fiber_list_xyz_r_fid = util.map_matrix_to_list_numpy(optimized_fibers)
    unique_ids = np.unique([fiber[:, util.fiber_id_column(fiber)][0] for fiber in fiber_list_xyz_r_fid])
    for fiber_xyz_r_fid in fiber_list_xyz_r_fid:
        length_along_axon = get_length_along_axon(fiber_xyz_r_fid)
        radii = fiber_xyz_r_fid[:,3]
        color_idx = np.argwhere(np.isin(unique_ids , fiber_xyz_r_fid[:, util.fiber_id_column(fiber_xyz_r_fid)][0])).ravel()[0]
        fiber_color = colors[color_idx]
        plt.plot(length_along_axon, radii, linewidth=1, color=fiber_color)    

        cv_of_radii = np.concatenate((cv_of_radii, get_cv_of_diameter_or_radius_along_each_axon(radii)))
    component_tag = _component_file_tag(component_label)
    plt.title(f'{_component_title(component_label)} radius variation along axon - Optimized', fontsize=17, pad=20)
    plt.xlabel('Length along axon (µm)',fontsize=15, labelpad=3)
    plt.ylabel(f'{_component_title(component_label)} radius (µm)', fontsize=15, labelpad=5)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.axis('tight')
    plt.subplots_adjust(wspace=0.5)
    folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/substrate_stats"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path+"/"+component_tag+"_Along_axon_radius_variation_"+config_params.EXP_DATE_TIME+".png", 
                dpi=500)
    plt.close(fig)

    config_params.CV_OUTER_MEAN, config_params.CV_OUTER_STDV = np.round(np.mean(cv_of_radii),3), np.round(np.std(cv_of_radii),3)
    config_params.CV_RADII = np.round(cv_of_radii,3)
    return 

def get_length_along_axon( single_fiber_xyz_r_fid):
    '''
    Calculate the cumulative distance along a chain of spheres represented by an array.
    Args:
        single_fiber_xyz_r_fid: An array with shape (n, 5), where n is the number of spheres
                                and each row contains the x, y, z, radius, and fiber ID (fid)
                                of the fiber.
    Returns:
        An array with shape (n,) containing the cumulative distance along the chain
        starting from 0 for the first sphere.
    '''
    sphere1s = single_fiber_xyz_r_fid[0:-1, :3]
    sphere2s = single_fiber_xyz_r_fid[1:, :3]
    distance = np.linalg.norm(sphere2s - sphere1s, axis=1)
    length_along = np.cumsum(distance, axis=0)
    length_along = np.concatenate((np.array([0.0]), length_along))
    return length_along

def get_cv_of_diameter_or_radius_along_each_axon( radii):
    '''get coefficient of variation of radius/diameter along an axon'''
    mean_of_samples = np.mean(radii)
    std_of_samples = np.std(radii)
    return np.array([std_of_samples / mean_of_samples])

def plot_diameter_CV_distribution(component_label="outer"):
    component_tag = _component_file_tag(component_label)
    xlim = _resolve_limit(getattr(config_params, "CV_DIAMETER_XLIM", None), _CV_DIAMETER_XLIM_DEFAULT)
    ylim = _resolve_limit(getattr(config_params, "CV_DIAMETER_YLIM", None), _CV_DIAMETER_YLIM_DEFAULT)
    fig = plt.figure(figsize=(6, 6))
    # Plot the histogram (fixed range so densities match across substrates)
    plt.hist(config_params.CV_RADII, bins=55, range=xlim, density=True)

    # Adding titles and labels
    plt.title(f'Distribution of CV for {component_label} diameter', fontsize=17, pad=20)
    plt.xlabel(f'CV ({component_label} diameter)', fontsize=15, labelpad=3)
    plt.ylabel('Density', fontsize=15, labelpad=5)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    # Calculate the mean + std for the label, ensure it's not directly config_params.CV_OUTER_MEAN which is already defined for the first line
    mean_plus_std_val = config_params.CV_OUTER_MEAN + config_params.CV_OUTER_STDV
    mean_minus_std_val = config_params.CV_OUTER_MEAN - config_params.CV_OUTER_STDV
    # Updated legend label for the mean line to show 'CV Mean: X.XXX ± CV Std Dev: Y.YYY'
    plt.axvline(x=config_params.CV_OUTER_MEAN, color='red', linestyle='--', linewidth=2,  label=f'CV = {config_params.CV_OUTER_MEAN:.3f} \u00B1 {config_params.CV_OUTER_STDV:.3f}')
    plt.axvline(x=mean_plus_std_val, color='grey', linestyle='--', linewidth=2)
    plt.axvline(x=mean_minus_std_val, color='grey', linestyle='--', linewidth=2)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.5)
    folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/substrate_stats"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path+"/CV_"+component_tag+"_diameter"+"_"+config_params.EXP_DATE_TIME+"_CVmean_"+str(config_params.CV_OUTER_MEAN)+"_CVstd_"+str(config_params.CV_OUTER_STDV)+".png", 
                dpi=500)
    plt.close(fig)
    return
