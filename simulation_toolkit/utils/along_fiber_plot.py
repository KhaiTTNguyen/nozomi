import matplotlib.pyplot as plt
import simulation_toolkit.utils.common_utils as util
import numpy as np
from scipy.stats import genextreme
import torch
import simulation_toolkit.toolkit_params as config_params
import os
import warnings

def plot_diameter_GEV_distribution(optimized_fibers):
    diameter = extract_radius_all(optimized_fibers)*2
    fig = plt.figure()
    nbins=20
    plt.hist(diameter, bins=nbins, density=True, align='mid', label='Substrate diameter')
    
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
    plt.title('Distribution of diameter', fontsize=17, pad=20)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.xlabel('Outer Diameter (µm)', fontsize=15, labelpad=5)
    plt.ylabel('Density', fontsize=15, labelpad=5)
    plt.axis('tight')
    plt.legend()
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.5)
    folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/substrate_stats"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path+"/Diameter_distribution_mean"+str(config_params.GEV_DIAMETER_MEAN)+'_std'+str(config_params.GEV_DIAMETER_STDV)+'_'+config_params.EXP_DATE_TIME+".png",
                 dpi=500)
    plt.close(fig)

def extract_radius_all(optimized_fibers):
    fiber_list_xyz_r_fid = util.split_matrix_to_list(optimized_fibers)
    r_values = np.array([])
    seen_ids = set() # filter out XY-wrapped fibers
    for fiber_array in fiber_list_xyz_r_fid:
        rad, fid = fiber_array[:, 3], fiber_array[:, 4][0]
        if fid not in seen_ids:
            r_values = np.concatenate((r_values, rad.flatten()))
            seen_ids.add(fid)
    return r_values
    
def plot_along_axon_radius_variation(optimized_fibers, colors):
    '''
    loop through each axon
    get length along axon
    get radius value
    '''
    cv_of_radii = np.empty(0)

    fig = plt.figure()
    fiber_list_xyz_r_fid = util.split_matrix_to_list(optimized_fibers)
    unique_ids = np.unique([fiber[:,-1][0] for fiber in fiber_list_xyz_r_fid])
    for fiber_xyz_r_fid in fiber_list_xyz_r_fid:
        length_along_axon = get_length_along_axon(fiber_xyz_r_fid)
        radii = fiber_xyz_r_fid[:,3]
        color_idx = np.argwhere(np.isin(unique_ids , fiber_xyz_r_fid[:,-1][0])).ravel()[0]
        fiber_color = colors[color_idx]
        plt.plot(length_along_axon, radii, linewidth=1, color=fiber_color)    

        cv_of_radii = np.concatenate((cv_of_radii, get_cv_of_diameter_or_radius_along_each_axon(radii)))
    plt.title('Radius variation along axon - Optimized', fontsize=17, pad=20)
    plt.xlabel('Length along axon (µm)',fontsize=15, labelpad=3)
    plt.ylabel('r (µm)', fontsize=15, labelpad=5)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.axis('tight')
    plt.subplots_adjust(wspace=0.5)
    folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/substrate_stats"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path+"/Along_axon_radius_variation_"+config_params.EXP_DATE_TIME+".png", 
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

def plot_diameter_CV_distribution():
    fig = plt.figure(figsize=(6, 6))
    # Plot the histogram
    plt.hist(config_params.CV_RADII, bins=55, density=True)

    # Adding titles and labels
    plt.title('Distribution of CV for outer diameter', fontsize=17, pad=20)
    plt.xlabel('CV (outer diameter)', fontsize=15, labelpad=3)
    plt.ylabel('Density', fontsize=15, labelpad=5)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.xlim(0, 0.8)
    plt.ylim(0, 12)
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
    plt.savefig(folder_path+"/CV_outer_diameter"+"_"+config_params.EXP_DATE_TIME+"_CVmean_"+str(config_params.CV_OUTER_MEAN)+"_CVstd_"+str(config_params.CV_OUTER_STDV)+".png", 
                dpi=500)
    plt.close(fig)
    return
