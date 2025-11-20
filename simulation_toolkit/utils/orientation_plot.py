import simulation_toolkit.utils.common_utils as util
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LightSource
from scipy.stats import iqr
from scipy.special import sph_harm
from scipy import linalg
import simulation_toolkit.defaults.params as config_params
import numpy as np
import matplotlib.ticker as ticker

def plot_along_axon_OD(spheres_xyz_r_fid ,  optimized=True):
    '''
    compute bundle mean orientation,
    for each fiber, 
        compute vectors for each segments 
        interpolate - euqal distances,
        (SKIPPED) compute angle of each vector & fiber bundle vector,
        accumulate
    pass into histogram

    add all histograms of all fibers

    subtract by the macroangle
    (angles are in degrees)'''
    L = config_params.BOX_LENGTH.detach().cpu().numpy()
    fiberlist_xyz_r_fid = util.split_matrix_to_list(spheres_xyz_r_fid.detach().cpu().numpy())
    folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/substrate_stats/ODI"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    points = []
    for fiber in fiberlist_xyz_r_fid:
        fiber = filter_spheres_outside_voxel(fiber, L)
        segs_v = (fiber[1:,:3] - fiber[:-1,:3])
        # print('segs_v pre',segs_v.shape)
        # segs_v = segs_v.squeeze(1)
        segs_v = segs_v/np.linalg.norm(segs_v, axis=1)[:,None] # normalize so seg_v is a unit vector
        # print(np.linalg.norm(segs_v, axis=1)[:,None])
        # angles = np.arccos(segs_v.dot(bundle_v)
        #           / (np.linalg.norm(bundle_v)*np.linalg.norm(segs_v, axis=1)))*180/np.pi
        # print('segs_v',segs_v.shape)
        points.append(segs_v)

    all_points = np.concatenate(points, axis=0)
    # plot density on sphere
    density_map, XX, YY, ZZ = plot_OD_density_sphere(all_points, optimized=True, folder_name=folder_path)
    # plot 3D harmonics glyphs
    lmax=20
    coeffs, fit_error = spherical_harmonics_fit(density_map, XX, YY, ZZ, lmax=lmax)
    plot_3D_glyph(coeffs, fit_error, lmax, folder_path, optimized)
    
def plot_3D_glyph( coeffs, fit_error, lmax, folder_name, optimized):
    theta = np.linspace(0, 2 * np.pi, 180)
    phi = np.linspace(0, np.pi, 180)
    '''
    121 coeffs
    create loop again N theata, N phi
    theta, phi = np.meshgrid(theta, phi)
    121 Y(l,m) with shape (N,N) each
    sum all to get Y(total) with shape (N,N)
    get radii np.abs(Y(total))
    '''
    xx = np.outer(np.cos(theta), np.sin(phi))
    yy = np.outer(np.sin(theta), np.sin(phi))
    zz = np.outer(np.ones(np.size(theta)), np.cos(phi))

    # Calculate radii
    Yvals = np.zeros((xx.shape[0], xx.shape[1], len(coeffs)))
    total_SH_coeffs, degrees = calculate_an(lmax)

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            x = xx[i, j]
            y = yy[i, j]
            z = zz[i, j]
            index = 0
            for l in degrees:
                for m in range(-l, l + 1):
                    theta_i, phi_i = np.arctan2(y, x), np.arctan2(np.sqrt(x**2 + y**2), z)
                    Yvals[i,j, index] = np.real(coeffs[index] * sph_harm(m, l, theta_i, phi_i))
                    index += 1
    # print('Yvals.shape',Yvals.shape)
    Yvals = np.sum(Yvals,axis=2)
    # print('Yvals.shape',Yvals.shape)
    # print('Yvals',Yvals.max())
    # print('Yvals',Yvals.min())
    Ymax, Ymin = Yvals.max(), Yvals.min()
    if (Ymax != Ymin):
    # normalize the values to [1, -1]
        # Yvals = 2 * (Yvals - Ymin)/(Ymax - Ymin) - 1
        # Yvals = 0.5 * (Yvals + 1)
    # Use the absolute value of Y(l,m) as the radius
        radii = np.clip(Yvals, 0, None)
        # radii = (radii - radii.min()) / (radii.max() - radii.min())
        # print('radii.shape',radii.shape)
    # Convert Yvals to spherical coordinates
    x = radii * xx
    y = radii * yy
    z = radii * zz
    x_abs = np.abs(x)
    y_abs = np.abs(y)
    z_abs = np.abs(z)

    # ==================Create the 3D plot=================
    folder_path = folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    '''plotting'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface with the color map
    '''map by location
    large abs x value = red
    large abs y value = green
    large abs z value = blue
    '''
    colors = np.zeros((x_abs.shape[0], x_abs.shape[1], 3))
    colors[:, :, 0]=x_abs
    colors[:, :, 1]=y_abs
    colors[:, :, 2]=z_abs
    ls = LightSource(azdeg=0, altdeg=65)
    # print(colors.shape)
    # print(z_abs.shape)

    ls = LightSource(60, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade_rgb(colors, z, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                        linewidth=0, antialiased=False, shade=False)
    # ax.view_init(0, 0)
    ax.set_xlim(-Ymax,Ymax)
    ax.set_ylim(-Ymax,Ymax)
    ax.set_zlim(-Ymax,Ymax)
    # Set axis labels and title
    ax.set_xlabel('X left-right', fontsize=15, labelpad=10)
    ax.set_ylabel('Y anterior-posterior', fontsize=15, labelpad=10)
    ax.set_zlabel('Z superior-inferior', fontsize=15, labelpad=10)
    ax.set_title('Sum of fitted-harmonics as 3D glyph',fontsize=17, pad=20)
    if optimized==True:
        plt.title("Along axon OD - optimized fibers", fontsize=17, pad=20)
        plt.savefig(folder_path+"/OD_histogram_optimized_lmax_"+str(lmax)+"_rError_"+str(fit_error)+".png", 
        dpi=500, edgecolor='b', format='png')
    else:
        plt.title("Along axon OD - preoptimized fibers", fontsize=17, pad=20)
        plt.savefig(folder_path+"/OD_histogram_preoptimized_lmax_"+str(lmax)+"_rError_"+str(fit_error)+".png", 
        dpi=500, edgecolor='b', format='png')
    plt.close(fig)
    
def plot_OD_density_sphere( all_points, optimized, folder_name):
    folder_path = folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    '''plotting'''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    u = np.linspace(0, 2 * np.pi, 180)
    v = np.linspace(0, np.pi/2, 45)

    # Create the unit sphere surface
    XX = np.outer(np.cos(u), np.sin(v))
    YY = np.outer(np.sin(u), np.sin(v))
    ZZ = np.outer(np.ones(np.size(u)), np.cos(v))
    WW = XX.copy()
    bin_radius=0.1
    for i in range(len(XX)):
        for j in range(len(XX[0])):
            x = XX[i, j]
            y = YY[i, j]
            z = ZZ[i, j]
            WW[i, j] = near(np.array([x, y, z]), all_points, bin_radius)  # Adjusted d0 to fit unit sphere

    WW = WW / np.amax(WW)
    myheatmap = WW
    ax.view_init(90, 0)
    # Plot the surface with colors
    surf = ax.plot_surface(XX, YY, ZZ, cstride=1, rstride=1, facecolors=cm.viridis(myheatmap), alpha=1.)

    # Add a color bar to show the mapping of colors to values
    m = cm.ScalarMappable(cmap=cm.viridis)
    m.set_array(myheatmap)
    plt.colorbar(m, ax=ax, shrink=0.5, aspect=5)

    ax.set_xlabel('X', fontsize=15, labelpad=15)
    ax.set_ylabel('Y', fontsize=15, labelpad=15)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.zaxis.set_tick_params(labelleft=False, bottom=False, top=False, labelbottom=False)
    ax.set_zticks([])
    ax.zaxis.set_ticks_position('none')
    if optimized==True:
        plt.title("Along axon OD - optimized fibers", fontsize=17, pad=3)
        plt.savefig(folder_path+"/OD_histogram_optimized_"+str(len(all_points))+"_fibers.png", 
        dpi=500, edgecolor='b', format='png')
    else:
        plt.title("Along axon OD - preoptimized fibers", fontsize=17, pad=3)
        plt.savefig(folder_path+"/OD_histogram_preoptimized_"+str(len(all_points))+"_fibers.png", 
        dpi=500, edgecolor='b', format='png')
    plt.close(fig)
    return WW, XX, YY, ZZ

def near( p, pntList, d0=0.1):
    distances = np.linalg.norm(p - pntList, axis=1)
    return np.sum(distances < d0)

def spherical_harmonics_fit( density_map, XX, YY, ZZ, lmax):
    """
    Fit spherical harmonics up to order l_max to the given orientation distribution samples.

    Parameters:
    samples (numpy.ndarray): 3D orientation distribution samples (N x 3)
    l_max (int): Maximum order of spherical harmonics to fit
    only get the even harmonics
    a0*
    Returns:
    numpy.ndarray: Fitted spherical harmonics coefficients
    """
    total_SH_coeffs, degrees = calculate_an(lmax)
    print(f"For lmax = {lmax}, total_SH_coeffs = {total_SH_coeffs}")
    SH_basis_m = np.zeros((density_map.flatten().shape[0], total_SH_coeffs), dtype=complex)
    index = 0
    for l in degrees:
        for m in range(-l, l + 1):
            for i in range(XX.shape[0]):
                for j in range(XX.shape[1]):
                    x = XX[i, j]
                    y = YY[i, j]
                    z = ZZ[i, j]
                    theta_i, phi_i = np.arctan2(y, x), np.arctan2(np.sqrt(x**2 + y**2), z)

                    SH_basis_m[i*XX.shape[1]+j, index] = sph_harm(m, l, theta_i, phi_i)
            index += 1
    # Pseudo inverse for complex matrix
    coeffs = linalg.pinvh(SH_basis_m.T @ SH_basis_m) @ SH_basis_m.T @ density_map.flatten()
    density_map_pred = SH_basis_m @ coeffs
    fit_error = np.linalg.norm(density_map_pred - density_map.flatten())/np.linalg.norm(density_map.flatten())
    fit_error = np.round(fit_error, 3)
    print('Error',fit_error)
    return coeffs, fit_error

def calculate_an( n):
    bn_terms = [2*i + 1 for i in range(0, n+1, 2)]
    return sum(bn_terms), [n for n in range(0, n+1, 2)]

def filter_spheres_outside_voxel( fiber, L):
    x = fiber[:, 0]
    y = fiber[:, 1]
    z = fiber[:, 2]
    valid_rows = (np.abs(x) <= L/2) & (np.abs(y) <= L/2) & (np.abs(z) <= L/2)
    return fiber[valid_rows]