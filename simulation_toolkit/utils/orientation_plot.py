import simulation_toolkit.utils.common_utils as util
import simulation_toolkit.utils.watson_fit as watson_fit
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LightSource
from scipy.stats import iqr
from scipy.special import sph_harm
from scipy import linalg
import simulation_toolkit.toolkit_params as config_params
import numpy as np
import matplotlib.ticker as ticker

def plot_along_axon_OD(spheres_xyz_r_fid ,  optimized=True):
    '''
    compute bundle mean orientation,
    for each fiber, 
        compute vectors for each segments 
        interpolate - euqal distances,
        compute angle of each vector & fiber bundle vector,
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
        segs_v = segs_v/np.linalg.norm(segs_v, axis=1)[:,None] # normalize so seg_v is a unit vector
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
    Yvals = np.sum(Yvals,axis=2)
    Ymax, Ymin = Yvals.max(), Yvals.min()
    if (Ymax != Ymin):
        radii = np.clip(Yvals, 0, None)
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
    norm = Ymax if Ymax > 0 else 1.0
    colors[:, :, 0] = x_abs / norm
    colors[:, :, 1] = y_abs / norm
    colors[:, :, 2] = z_abs / norm
    ls = LightSource(azdeg=0, altdeg=65)
    
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
        plt.title("Along axon FOD - optimized fibers", fontsize=17, pad=20)
        plt.savefig(folder_path+"/FOD_3D_glyph.png", 
        dpi=500, edgecolor='b', format='png')
    else:
        plt.title("Along axon FOD - preoptimized fibers", fontsize=17, pad=20)
        plt.savefig(folder_path+"/FOD_3D_glyph_preoptimized.png", 
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
        plt.savefig(folder_path+"/OD_histogram.png", 
        dpi=500, edgecolor='b', format='png')
    else:
        plt.title("Along axon OD - preoptimized fibers", fontsize=17, pad=3)
        plt.savefig(folder_path+"/OD_histogram_preoptimized.png", 
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
    # print(f"For lmax = {lmax}, total_SH_coeffs = {total_SH_coeffs}")
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
    return coeffs, fit_error

def calculate_an( n):
    bn_terms = [2*i + 1 for i in range(0, n+1, 2)]
    return sum(bn_terms), [n for n in range(0, n+1, 2)]


def spherical_harmonics_fit_from_tangents(tangents, lmax):
    """Compute even-SH FOD coefficients directly from unit tangent vectors.

    Uses the closed-form projection:
        c_lm = (1/N) * sum_i  conj(Y_lm(theta_i, phi_i))

    This avoids histogram discretisation, lat-lon area bias, hemisphere
    restriction, and bin_radius tuning — all sources of the equatorial
    ringing / pinched-waist artifact.  Because only even-l harmonics are
    used (antipodal symmetry) and Y_lm(-n) == Y_lm(n) for even l, it does
    not matter whether tangents are pre-flipped to the upper hemisphere.

    Parameters
    ----------
    tangents : ndarray, shape (N, 3)
        Unit tangent vectors (Cartesian).
    lmax : int
        Maximum even SH degree.

    Returns
    -------
    coeffs : ndarray, complex, shape (total_SH_coeffs,)
    fit_error : float  (0.0 — no density-map reference to compare against)
    """
    _, degrees = calculate_an(lmax)
    t = np.asarray(tangents, dtype=float)
    # Spherical angles
    azimuth = np.arctan2(t[:, 1], t[:, 0])          # theta in scipy convention
    polar   = np.arctan2(np.hypot(t[:, 0], t[:, 1]),
                         t[:, 2])                    # phi in scipy convention
    total_coeffs = sum(2 * l + 1 for l in degrees)
    coeffs = np.zeros(total_coeffs, dtype=complex)
    idx = 0
    for l in degrees:
        for m in range(-l, l + 1):
            coeffs[idx] = np.conj(sph_harm(m, l, azimuth, polar)).mean()
            idx += 1
    return coeffs, 0.0

def filter_spheres_outside_voxel( fiber, L):
    x = fiber[:, 0]
    y = fiber[:, 1]
    z = fiber[:, 2]
    valid_rows = (np.abs(x) <= L/2) & (np.abs(y) <= L/2) & (np.abs(z) <= L/2)
    return fiber[valid_rows]


# =====================================================================
# Arc-length based orientation statistics + Watson-kappa fitting
# (Callaghan / ConFiG-style substrate validation)
# =====================================================================

def _box_length_as_float():
    """Return config_params.BOX_LENGTH as a plain float, whether it is a
    torch tensor (during substrate generation) or a python scalar (during
    reprocessing of saved substrates)."""
    L = config_params.BOX_LENGTH
    try:
        return float(L.detach().cpu().numpy())
    except AttributeError:
        return float(L)


def _arclength_tangents_and_fit(spheres_xyz_r_fid, ds=None):
    """Split sphere array into fibers, arc-length-resample each, pool unit
    tangents and fit a Watson distribution.

    Returns (tangents, fit_result).
    """
    if hasattr(spheres_xyz_r_fid, 'detach'):
        arr = spheres_xyz_r_fid.detach().cpu().numpy()
    else:
        arr = np.asarray(spheres_xyz_r_fid)

    L = _box_length_as_float()
    fiber_list = util.split_matrix_to_list(arr)
    # Keep spheres strictly inside or on the voxel boundary, matching the
    # existing plot_along_axon_OD behaviour.
    fiber_list = [filter_spheres_outside_voxel(f, L) for f in fiber_list]
    fiber_list = [f for f in fiber_list if f.shape[0] >= 2]

    tangents = watson_fit.collect_arclength_tangents(fiber_list, ds=ds)
    result = watson_fit.fit_watson_scatter(tangents)
    result['ds'] = ds
    result['n_fibers'] = len(fiber_list)
    return tangents, result


def _kappa_title_suffix(fit):
    k = fit.get('kappa', np.nan)
    k_des = fit.get('kappa_prescribed', None)
    if not np.isfinite(k):
        return "  (arc-length fit failed)"
    if k_des is None or not np.isfinite(float(k_des)):
        return f"\n$\\kappa_{{fit}}$={k:.2f}"
    return (f"\n$\\kappa_{{designed}}$={float(k_des):g}"
            f",  $\\kappa_{{fit}}$={k:.2f}")


def _kappa_file_tag(fit):
    k = fit.get('kappa', np.nan)
    k_des = fit.get('kappa_prescribed', None)
    k_str = "Kfit_nan" if not np.isfinite(k) else f"Kfit_{k:.2f}"
    if k_des is None or not np.isfinite(float(k_des)):
        return k_str
    return f"Kdes_{float(k_des):g}_{k_str}"


def plot_along_axon_OD_arclength(spheres_xyz_r_fid, optimized=True, ds=None,
                                 lmax=8):
    """Arc-length variant of plot_along_axon_OD.

    - Resamples each fiber's centerline at uniform arc length before taking
      tangents (avoids sphere-density bias from meshing / optimisation).
    - Fits a bipolar Watson distribution via scatter-matrix MLE.
    - Embeds the designed (prescribed) and fitted kappa into both plot titles
      and filenames.
    - Saves next to the existing OD / FOD plots but with an ``_arclength``
      suffix so it does NOT overwrite the previous outputs.

    Parameters
    ----------
    lmax : int
        Maximum (even) SH degree used to fit/render the FOD glyph. Lower
        values (e.g. 6-10) suppress Gibbs-style ringing at the equator that
        shows up as a pinched waist for broad FODs; high values sharpen
        glyphs for narrow FODs at the cost of equatorial ringing.
    """
    folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH + "/figs/substrate_stats/ODI"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    tangents, fit = _arclength_tangents_and_fit(spheres_xyz_r_fid, ds=ds)

    # Attach prescribed kappa (if configured) so title/filename helpers can
    # show both designed and fitted values.
    k_prescribed = getattr(config_params, 'ORIENTATION_SHAPE_PARAM', None)
    try:
        k_prescribed = float(k_prescribed) if k_prescribed else None
    except Exception:
        k_prescribed = None
    fit['kappa_prescribed'] = k_prescribed

    if tangents.shape[0] < 3:
        print("[plot_along_axon_OD_arclength] Too few tangent samples; skipping plots.")
        return fit

    # --- density map on unit hemisphere (kept for the 2-D OD visualisation) ---
    density_map, XX, YY, ZZ = _plot_OD_density_sphere_arclength(
        tangents, fit, optimized=optimized, folder_name=folder_path)

    # --- spherical-harmonics glyph via direct closed-form projection ---
    # Bypasses histogram discretisation, lat-lon area bias, and hemisphere
    # restriction — all sources of the equatorial ringing artifact.
    coeffs, fit_error = spherical_harmonics_fit_from_tangents(tangents, lmax=lmax)
    _plot_3D_glyph_arclength(coeffs, fit_error, lmax, folder_path,
                             fit, optimized=optimized)

    # --- achieved Watson-samples scatter (mirrors the prescribed
    #     Watson_samples_kappa_<K>.png style for direct comparison) ---
    _plot_watson_samples_achieved(tangents, fit, folder_path, optimized=optimized)
    return fit


def _plot_OD_density_sphere_arclength(all_points, fit, optimized, folder_name):
    folder_path = folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    u = np.linspace(0, 2 * np.pi, 180)
    v = np.linspace(0, np.pi / 2, 45)

    XX = np.outer(np.cos(u), np.sin(v))
    YY = np.outer(np.sin(u), np.sin(v))
    ZZ = np.outer(np.ones(np.size(u)), np.cos(v))
    WW = XX.copy()
    bin_radius = 0.1
    for i in range(len(XX)):
        for j in range(len(XX[0])):
            x = XX[i, j]
            y = YY[i, j]
            z = ZZ[i, j]
            WW[i, j] = near(np.array([x, y, z]), all_points, bin_radius)

    if WW.max() > 0:
        WW = WW / np.amax(WW)
    myheatmap = WW
    ax.view_init(90, 0)
    ax.plot_surface(XX, YY, ZZ, cstride=1, rstride=1,
                    facecolors=cm.viridis(myheatmap), alpha=1.)
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

    stage = "optimized" if optimized else "preoptimized"
    title = f"Along axon OD (arc-length) - {stage} fibers" + _kappa_title_suffix(fit)
    plt.title(title, fontsize=14, pad=3)
    fname = f"OD_histogram_arclength_{_kappa_file_tag(fit)}"
    if not optimized:
        fname += "_preoptimized"
    plt.savefig(os.path.join(folder_path, fname + ".png"),
                dpi=500, edgecolor='b', format='png')
    plt.close(fig)
    return WW, XX, YY, ZZ


def _plot_3D_glyph_arclength(coeffs, fit_error, lmax, folder_name,
                             fit, optimized):
    theta = np.linspace(0, 2 * np.pi, 180)
    phi = np.linspace(0, np.pi, 180)
    xx = np.outer(np.cos(theta), np.sin(phi))
    yy = np.outer(np.sin(theta), np.sin(phi))
    zz = np.outer(np.ones(np.size(theta)), np.cos(phi))

    Yvals = np.zeros((xx.shape[0], xx.shape[1], len(coeffs)))
    _, degrees = calculate_an(lmax)

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            x = xx[i, j]
            y = yy[i, j]
            z = zz[i, j]
            index = 0
            for l in degrees:
                for m in range(-l, l + 1):
                    theta_i = np.arctan2(y, x)
                    phi_i = np.arctan2(np.sqrt(x**2 + y**2), z)
                    Yvals[i, j, index] = np.real(coeffs[index] *
                                                 sph_harm(m, l, theta_i, phi_i))
                    index += 1
    Yvals = np.sum(Yvals, axis=2)
    Ymax, Ymin = Yvals.max(), Yvals.min()
    radii = np.clip(Yvals, 0, None) if (Ymax != Ymin) else np.zeros_like(Yvals)
    x = radii * xx
    y = radii * yy
    z = radii * zz
    x_abs = np.abs(x)
    y_abs = np.abs(y)
    z_abs = np.abs(z)

    folder_path = folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = np.zeros((x_abs.shape[0], x_abs.shape[1], 3))
    # Normalise to [0, 1] — raw coordinates can exceed 1 when peak radius > 1,
    # which causes matplotlib to raise "RGBA values should be within 0-1 range".
    norm = Ymax if Ymax > 0 else 1.0
    colors[:, :, 0] = x_abs / norm
    colors[:, :, 1] = y_abs / norm
    colors[:, :, 2] = z_abs / norm
    ls = LightSource(60, 45)
    rgb = ls.shade_rgb(colors, z, vert_exag=0.1, blend_mode='soft')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                    linewidth=0, antialiased=False, shade=False)
    ax.set_xlim(-Ymax, Ymax)
    ax.set_ylim(-Ymax, Ymax)
    ax.set_zlim(-Ymax, Ymax)
    ax.set_xlabel('X left-right', fontsize=15, labelpad=10)
    ax.set_ylabel('Y anterior-posterior', fontsize=15, labelpad=10)
    ax.set_zlabel('Z superior-inferior', fontsize=15, labelpad=10)

    stage = "optimized" if optimized else "preoptimized"
    title = (f"Along axon FOD (arc-length, lmax={lmax}) - {stage} fibers"
             + _kappa_title_suffix(fit))
    plt.title(title, fontsize=14, pad=20)
    fname = f"FOD_3D_glyph_arclength_direct_lmax{lmax}_{_kappa_file_tag(fit)}"
    if not optimized:
        fname += "_preoptimized"
    plt.savefig(os.path.join(folder_path, fname + ".png"),
                dpi=500, edgecolor='b', format='png')
    plt.close(fig)


def _plot_watson_samples_achieved(tangents, fit, folder_name,
                                  optimized=True, max_points=3000):
    """Scatter plot of the achieved arc-length tangent vectors on the upper
    hemisphere, styled to mirror ``WatsonDistribution.visualize_watson_samples``
    so the prescribed Watson_samples_kappa_<K>.png can be compared directly
    to the achieved orientations of the final substrate.
    """
    folder_path = folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Force upper-hemisphere (Watson is axial: t and -t are equivalent).
    pts = np.asarray(tangents, dtype=float)
    if pts.shape[0] == 0:
        return
    flip = pts[:, 2] < 0
    pts = pts.copy()
    pts[flip] = -pts[flip]

    # Subsample for plot clarity (matches the ~1000-point prescribed plot).
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    mu = fit.get('mu', np.array([0.0, 0.0, 1.0]))
    if mu[2] < 0:
        mu = -mu

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi / 2, 25)
    x_s = np.outer(np.cos(u), np.sin(v))
    y_s = np.outer(np.sin(u), np.sin(v))
    z_s = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_s, y_s, z_s, alpha=0.3, color='lightgray')

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.8, s=20, c='blue')
    ax.quiver(0, 0, 0, mu[0], mu[1], mu[2],
              color='red', arrow_length_ratio=0.1, linewidth=3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    stage = "optimized" if optimized else "preoptimized"
    title = (f"Achieved tangent samples on upper hemisphere ({stage})"
             + _kappa_title_suffix(fit))
    ax.set_title(title, fontsize=14)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.tight_layout()

    fname = f"Watson_samples_achieved_arclength_{_kappa_file_tag(fit)}"
    if not optimized:
        fname += "_preoptimized"
    plt.savefig(os.path.join(folder_path, fname + ".png"),
                dpi=500, edgecolor='b', format='png')
    plt.close(fig)