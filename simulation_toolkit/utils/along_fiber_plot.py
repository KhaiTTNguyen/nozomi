import matplotlib.pyplot as plt
import simulation_toolkit.utils.common_utils as util
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from scipy.stats import genextreme
import torch
import simulation_toolkit.defaults.params as config_params
import os

#---------------------------Plot slices--------------------------------
def plot_slice( ax, fiber_slice, fiber_idx, color, overlap_indices=None):
    color_sphere=0
    for sphere_idx, node in enumerate(fiber_slice):    
        ang = np.linspace(0,2*np.pi,100)
        slice_x = node[0] + node[3]*np.cos(ang)
        slice_y = node[1] + node[3]*np.sin(ang)
        # current_idx = (sphere_idx+fiber_idx)
        # if np.any(overlap_indices == current_idx): 
        #     color_sphere = 'red' 
        #     ax.plot(slice_x, slice_y, color=color_sphere, linewidth=5, alpha=0.5)
        # else: 
        color_sphere = color
        # print('slice_x', slice_x.shape)
        # print('slice_y', slice_y.shape)
        ax.plot(slice_x, slice_y, color=color_sphere, linewidth=0.7, alpha=1.) #  alpha=1 MAX

def plot_slices(spheres_xyz_r_fid,  N, color):
    z_coords = np.linspace(-config_params.BOX_LENGTH/N, config_params.BOX_LENGTH/N, N)
    # print(z_coords)
    print('color',color.shape)
    for z in z_coords:
        plot_slice_at_z(z, spheres_xyz_r_fid, color=color)  
        
def plot_slice_at_z(z_coord, spheres_xyz_r_fid, color, num_iter=None, overlap_indices=None):
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # ax1.set_aspect( 1 ), ax2.set_aspect( 1 )
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_aspect( 1 )#, ax2.set_aspect( 1 )
    
    current_fiber_list = util.split_matrix_to_list(spheres_xyz_r_fid)
    #-------------------- plot fibers ----------------------
    'get index of fiber.original_id in unique_ids --> map to color'
    unique_ids = np.unique([fiber[:,-1][0] for fiber in current_fiber_list])
    fiber_idx=0
    for fiber_count, fiber in enumerate(current_fiber_list):
        # get indcies of nodes where (z_coord-0.01) <fiber.z and fiber.z < (z_coord+0.01)
        fiber_count+=1
        # za, z_delta = fiber[:,2], 1.*fiber[:,3][0]
        za, z_delta = fiber[:,2], 2.*fiber[:,3][0]
        z_low, z_high = z_coord-z_delta , z_coord+z_delta #5
        m = np.logical_and((z_low<za),(za<z_high), (fiber_count==fiber[:,-1]))
        # slice_linear, slice_cubic = interpolate_fiber_slice(z_coord, fiber, m)
        slice_linear = interpolate_fiber_slice(z_coord, fiber, m)
        # print('fiber_slice', slice_linear.shape) #, slice_cubic.shape)
        
        color_idx = np.argwhere(np.isin(unique_ids , fiber[:,-1][0])).ravel()[0]
        fiber_color = color[color_idx]
        plot_slice(ax=ax1, fiber_slice=slice_linear, fiber_idx=fiber_idx, 
                            color=fiber_color, overlap_indices=overlap_indices)
        # plot_slice(ax=ax2, fiber_slice=slice_cubic, fiber_idx=fiber_idx, 
        #                   color=fiber_color, overlap_indices=overlap_indices)
        fiber_idx=fiber_idx+len(fiber)
    ax1.set_xlim(-config_params.BOX_LENGTH/2, config_params.BOX_LENGTH/2)
    ax1.set_ylim(-config_params.BOX_LENGTH/2, config_params.BOX_LENGTH/2)
    ax1.set_xlabel("x (µm)", fontsize=15, labelpad=5)
    ax1.set_ylabel("y (µm)", fontsize=15, labelpad=5)
    ax1.tick_params(axis='both', which='major', labelsize=13)
    # ax2.set_xlim(-config_params.BOX_LENGTH/2, config_params.BOX_LENGTH/2)
    # ax2.set_ylim(-config_params.BOX_LENGTH/2, config_params.BOX_LENGTH/2)
    # ax2.set_xlabel("x (µm)")
    # ax2.set_ylabel("y (µm)")
    plt.title("slice plot of fibers at "+str(z_coord), fontsize=17, pad=20)
    folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/substrate_stats"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if num_iter!=None:
        plt.savefig(folder_path+"/slice_at"+str(z_coord)+'_'+str(len(current_fiber_list))+"_fibers"+str(config_params.EXP_DATE_TIME)+"_num_iter_"+str(num_iter)+".png", 
            dpi=500, edgecolor='b', format='png')
    else:
        plt.savefig(folder_path+"/slice_at"+str(z_coord)+'_'+str(len(current_fiber_list))+"_fibers"+str(config_params.EXP_DATE_TIME)+".png", 
            dpi=500, edgecolor='b', format='png')
    plt.close(fig)

def interpolate_fiber_slice( z_coord, fiber, m):
    selected_points = fiber[m]
    x_sel, y_sel, z_sel, r_sel, fid_sel = selected_points[:, 0], selected_points[:, 1], selected_points[:, 2], selected_points[:, 3], selected_points[:, 4]

    # Linear interpolation
    linear_interp_x = interp1d(z_sel, x_sel, kind='linear')
    linear_interp_y = interp1d(z_sel, y_sel, kind='linear')
    x_linear = linear_interp_x(z_coord)
    y_linear = linear_interp_y(z_coord)
    fiber_slice_linear = np.array([x_linear, y_linear, z_coord, r_sel[0], fid_sel[0]])
    # Cubic spline interpolation
    
    # cubic_interp_x = CubicSpline(z_sel, x_sel)
    # cubic_interp_y = CubicSpline(z_sel, y_sel)
    # x_cubic = cubic_interp_x(z_coord)
    # y_cubic = cubic_interp_y(z_coord)
    # fiber_slice_cubic = np.array([x_cubic, y_cubic, z_coord, r_sel[0], fid_sel[0]])
    return fiber_slice_linear[None, :] #, fiber_slice_cubic[None, :]

def plot_diameter_GEV_distribution(optimized_fibers):
    # mu, sigma = mean_d_underlying, sigma_d_underlying
    diameter = extract_radius_all(optimized_fibers)*2

    fig = plt.figure()
    nbins=20
    plt.hist(diameter, bins=nbins, density=True, align='mid', label='Substrate diameter')
    
    x = np.linspace(min(diameter), max(diameter), 10000)
    shape_gev_fitted, loc_gev_fitted, scale_gev_fitted  = genextreme.fit(diameter)
    meanGEV, varGEV, skew, kurt = genextreme.stats(c=shape_gev_fitted, loc=loc_gev_fitted, scale=scale_gev_fitted, moments='mvsk')
    stdvGEV = np.sqrt(varGEV)
    config_params.GEV_DIAMETER_MEAN, config_params.GEV_DIAMETER_STDV = np.round(meanGEV, 3), np.round(stdvGEV, 3)
    plt.plot(x, genextreme.pdf(x, shape_gev_fitted, loc_gev_fitted, scale_gev_fitted), 'r-', lw=2, label='Fitted GEV')
    
    # Adding titles and labels   
    plt.title('Distribution of diameter - Optimized', fontsize=17, pad=20)
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
    plt.savefig(folder_path+"/Optimized_diameter_distribution_mean"+str(config_params.GEV_DIAMETER_MEAN)+'_std'+str(config_params.GEV_DIAMETER_STDV)+'_'+config_params.EXP_DATE_TIME+"_"+".png",
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
    
def get_length_along_axon( single_fiber_xyz_r_fid):
    '''
    Calculates the cumulative distance along a chain of spheres represented by a NumPy array.
    Args:
        single_fiber_xyz_r_fid: A NumPy array with shape (n, 5), where n is the number of spheres
                                and each row contains the x, y, z, radius, and fiber ID (fid)
                                of the fiber.
    Returns:
        A NumPy array with shape (n,) containing the cumulative distance along the chain
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

def plot_along_axon_radius_variation(optimized_fibers, colors):
    '''
    loop through each axon
    get length along axon
    get radius value
    '''
    cv_of_radii = np.empty(0)
    all_spacing_samples = torch.empty(0) 

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
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.5)
    folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/substrate_stats"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path+"/Optimized_along_axon_radius_variation_"+config_params.EXP_DATE_TIME+"_"+".png", 
                dpi=500)
    plt.close(fig)

    config_params.CV_OUTER_MEAN, config_params.CV_OUTER_STDV = np.round(np.mean(cv_of_radii),3), np.round(np.std(cv_of_radii),3)
    config_params.CV_RADII = np.round(cv_of_radii,3)
    return 

def plot_diameter_CV_distribution():
    fig = plt.figure()
    # Plot the histogram
    plt.hist(config_params.CV_RADII, bins=55, density=True)

    # Adding titles and labels
    plt.title('Distribution of CV for outer diameter', fontsize=17, pad=20)
    plt.xlabel('CV (outer diameter)', fontsize=15, labelpad=3)
    plt.ylabel('Density', fontsize=15, labelpad=5)
    plt.tick_params(axis='both', which='major', labelsize=13)
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.5)
    folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/substrate_stats"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path+"/Optimized_CV_outer_diameter"+"_"+config_params.EXP_DATE_TIME+"_CVmean_"+str(config_params.CV_OUTER_MEAN)+"_CVstd_"+str(config_params.CV_OUTER_STDV)+"_"+".png", 
                dpi=500)
    plt.close(fig)

    return
def plot_bead_spacing_distribution():
    return