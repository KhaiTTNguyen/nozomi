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


def _as_numpy(spheres_xyz_r_fid):
    if hasattr(spheres_xyz_r_fid, 'detach'):
        return spheres_xyz_r_fid.detach().cpu().numpy()
    return np.asarray(spheres_xyz_r_fid)


def _component_suffix(component_label):
    if component_label is None:
        return "", ""
    tag = component_label.lower().replace(" ", "_")
    return f" ({component_label})", f"_{tag}"

def plot_along_axon_OD(spheres_xyz_r_fid ,  optimized=True, component_label=None):
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
    fiberlist_xyz_r_fid = util.split_matrix_to_list(_as_numpy(spheres_xyz_r_fid))
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
    density_map, XX, YY, ZZ = plot_OD_density_sphere(all_points, optimized=True, folder_name=folder_path, component_label=component_label)
    # plot 3D harmonics glyphs
    lmax=20
    coeffs, fit_error = spherical_harmonics_fit(density_map, XX, YY, ZZ, lmax=lmax)
    plot_3D_glyph(coeffs, fit_error, lmax, folder_path, optimized, component_label=component_label)
    
def plot_3D_glyph( coeffs, fit_error, lmax, folder_name, optimized, component_label=None):
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
    component_title, component_tag = _component_suffix(component_label)
    if optimized==True:
        plt.title("Along axon FOD" + component_title + " - optimized fibers", fontsize=17, pad=20)
        plt.savefig(folder_path+"/FOD_3D_glyph"+component_tag+".png", 
        dpi=500, edgecolor='b', format='png')
    else:
        plt.title("Along axon FOD" + component_title + " - preoptimized fibers", fontsize=17, pad=20)
        plt.savefig(folder_path+"/FOD_3D_glyph"+component_tag+"_preoptimized.png", 
        dpi=500, edgecolor='b', format='png')
    plt.close(fig)
    
def plot_OD_density_sphere( all_points, optimized, folder_name, component_label=None):
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
    component_title, component_tag = _component_suffix(component_label)
    if optimized==True:
        plt.title("Along axon OD" + component_title + " - optimized fibers", fontsize=17, pad=3)
        plt.savefig(folder_path+"/OD_histogram"+component_tag+".png", 
        dpi=500, edgecolor='b', format='png')
    else:
        plt.title("Along axon OD" + component_title + " - preoptimized fibers", fontsize=17, pad=3)
        plt.savefig(folder_path+"/OD_histogram"+component_tag+"_preoptimized.png", 
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


def spherical_harmonics_fit_from_tangents(tangents, lmax, sh_smooth=1.0):
    """Compute even-SH FOD coefficients directly from unit tangent vectors.

    Uses the closed-form projection:
        c_lm = (1/N) * sum_i  conj(Y_lm(theta_i, phi_i))

    This avoids histogram discretisation, lat-lon area bias, hemisphere
    restriction, and bin_radius tuning — all sources of the equatorial
    ringing / pinched-waist artifact.  Because only even-l harmonics are
    used (antipodal symmetry) and Y_lm(-n) == Y_lm(n) for even l, it does
    not matter whether tangents are pre-flipped to the upper hemisphere.

    Ring-artifact suppression
    -------------------------
    The raw projection above is the SH representation of a *sum of delta
    functions* (one spike per tangent). Truncating that infinite series at
    ``lmax`` produces Gibbs ringing: damped oscillations around the peak
    whose negative side-lobes and spurious equatorial bumps survive the
    later non-negativity clip as a thin colored ring / pinched waist at the
    glyph's centre.

    To remove it we apodize each SH band by a Laplace-Beltrami heat-kernel
    weight

        w_l = exp(-sh_smooth * l(l+1) / (lmax(lmax+1)))

    which is mathematically equivalent to convolving the delta-FOD with a
    smooth zonal blob on the sphere (a band-limited kernel density estimate).
    High-l bands — the ones carrying the ringing — are tapered while the
    low-l lobe structure is preserved. The ``l(l+1)/(lmax(lmax+1))``
    normalisation makes a given ``sh_smooth`` deliver comparable smoothing
    across different ``lmax`` (``w_lmax = exp(-sh_smooth)``).

    Parameters
    ----------
    tangents : ndarray, shape (N, 3)
        Unit tangent vectors (Cartesian).
    lmax : int
        Maximum even SH degree.
    sh_smooth : float, optional
        Heat-kernel apodization strength (default 1.0). 0 disables smoothing
        and reproduces the raw delta-projection glyph (maximal ringing).
        Larger values (e.g. 1.5-3) further suppress ringing at the cost of a
        broader glyph; useful for sharp FODs (high kappa) where ``lmax`` is
        too low to represent the peak without oscillating.

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
    # Laplace-Beltrami apodization normaliser so sh_smooth scales the same
    # way regardless of lmax (w_lmax = exp(-sh_smooth)).
    lb_norm = float(lmax * (lmax + 1)) if lmax > 0 else 1.0
    idx = 0
    for l in degrees:
        w_l = np.exp(-sh_smooth * (l * (l + 1)) / lb_norm) if sh_smooth > 0 else 1.0
        for m in range(-l, l + 1):
            coeffs[idx] = w_l * np.conj(sph_harm(m, l, azimuth, polar)).mean()
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
    odi = fit.get('ODI', np.nan)
    k_des = fit.get('kappa_prescribed', None)
    if not np.isfinite(k):
        return "  (arc-length fit failed)"
    odi_part = f",  $ODI_{{fit}}$={odi:.4f}" if np.isfinite(odi) else ""
    if k_des is None or not np.isfinite(float(k_des)):
        return f"\n$\\kappa_{{fit}}$={k:.2f}{odi_part}"
    return (f"\n$\\kappa_{{designed}}$={float(k_des):g}"
            f",  $\\kappa_{{fit}}$={k:.2f}{odi_part}")


def _kappa_file_tag(fit):
    k = fit.get('kappa', np.nan)
    odi = fit.get('ODI', np.nan)
    k_des = fit.get('kappa_prescribed', None)
    k_str = "Kfit_nan" if not np.isfinite(k) else f"Kfit_{k:.2f}"
    odi_str = "ODIfit_nan" if not np.isfinite(odi) else f"ODIfit_{odi:.4f}"
    base = f"{k_str}_{odi_str}"
    if k_des is None or not np.isfinite(float(k_des)):
        return base
    return f"Kdes_{float(k_des):g}_{base}"


def plot_along_axon_OD_arclength(spheres_xyz_r_fid, optimized=True, ds=None,
                                 lmax=8, sh_smooth=1.0, component_label=None):
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
    sh_smooth : float
        Laplace-Beltrami heat-kernel apodization strength applied to the SH
        bands to remove the residual equatorial ring / pinched-waist artifact
        (see ``spherical_harmonics_fit_from_tangents``). 0 disables it; the
        default 1.0 removes the ring while keeping the lobes sharp.
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
        tangents, fit, optimized=optimized, folder_name=folder_path, component_label=component_label)

    # --- spherical-harmonics glyph via direct closed-form projection ---
    # Bypasses histogram discretisation, lat-lon area bias, and hemisphere
    # restriction — all sources of the equatorial ringing artifact.
    coeffs, fit_error = spherical_harmonics_fit_from_tangents(tangents, lmax=lmax,
                                                              sh_smooth=sh_smooth)
    _plot_3D_glyph_arclength(coeffs, fit_error, lmax, folder_path,
                             fit, optimized=optimized, component_label=component_label,
                             sh_smooth=sh_smooth)

    # --- analytic Watson model glyph (ring-free, lmax/smooth-independent) ---
    _plot_3D_glyph_analytic_watson(fit, folder_path, optimized=optimized,
                                   component_label=component_label)

    # --- achieved Watson-samples scatter (mirrors the prescribed
    #     Watson_samples_kappa_<K>.png style for direct comparison) ---
    _plot_watson_samples_achieved(tangents, fit, folder_path, optimized=optimized, component_label=component_label)
    return fit


def _plot_OD_density_sphere_arclength(all_points, fit, optimized, folder_name,
                                      method_tag="arclength",
                                      method_label="arc-length",
                                      component_label=None):
    """Smoothed upper-hemisphere density of the SAME tangent samples that
    appear in the achieved Watson-samples scatter plot.

    Pipeline (matches ``_plot_watson_samples_achieved`` exactly):
      1. Flip every tangent with z<0 to the upper hemisphere (Watson is axial;
         t and -t are physically identical). Without this, any tangent landing
         below the equator is silently missed, so broad FODs look too tight.
      2. Evaluate a spherical (von Mises-Fisher) kernel on a lat-lon grid
         covering the upper hemisphere. The kernel bandwidth (``kappa_kde``)
         adapts to the fitted Watson kappa so both sharp and broad
         distributions are rendered with comparable visual smoothness.
      3. Normalise by the max so the colormap uses the full [0,1] range.
    """
    folder_path = folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    pts = np.asarray(all_points, dtype=float)
    if pts.size == 0:
        return None, None, None, None
    # Axial symmetry: mirror lower hemisphere into the upper one, matching
    # the scatter plot in _plot_watson_samples_achieved.
    flip = pts[:, 2] < 0
    if np.any(flip):
        pts = pts.copy()
        pts[flip] = -pts[flip]

    # Kernel bandwidth: use the fitted Watson kappa when available; otherwise
    # fall back to a moderate concentration. Clamp so that very sharp fits
    # still render as visible blobs and very broad fits do not oversmooth.
    k_fit = fit.get('kappa', np.nan)
    if np.isfinite(k_fit) and k_fit > 0:
        kappa_kde = float(np.clip(k_fit, 5.0, 200.0))
    else:
        kappa_kde = 20.0

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    u = np.linspace(0, 2 * np.pi, 180)
    v = np.linspace(0, np.pi / 2, 45)

    XX = np.outer(np.cos(u), np.sin(v))
    YY = np.outer(np.sin(u), np.sin(v))
    ZZ = np.outer(np.ones(np.size(u)), np.cos(v))

    # Vectorised spherical KDE: density at each grid node is
    #   WW(n) = (1/N) * sum_i exp(kappa * (n . t_i))
    # where n is the grid unit vector and t_i are the tangent samples.
    # (We drop the vMF normalisation constant because we only visualise the
    # max-normalised heatmap.)
    grid = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)  # (G, 3)
    # Chunk to cap memory at ~G * chunk * 8B.
    chunk = 4096
    WW_flat = np.zeros(grid.shape[0], dtype=float)
    for start in range(0, pts.shape[0], chunk):
        block = pts[start:start + chunk]             # (B, 3)
        cos_ang = grid @ block.T                      # (G, B)
        WW_flat += np.exp(kappa_kde * cos_ang).sum(axis=1)
    WW_flat /= max(pts.shape[0], 1)
    WW = WW_flat.reshape(XX.shape)

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
    component_title, component_tag = _component_suffix(component_label)
    title = f"Along axon OD{component_title} ({method_label}) - {stage} fibers" + _kappa_title_suffix(fit)
    plt.title(title, fontsize=14, pad=3)
    fname = f"OD_histogram{component_tag}_{method_tag}_{_kappa_file_tag(fit)}"
    if not optimized:
        fname += "_preoptimized"
    plt.savefig(os.path.join(folder_path, fname + ".png"),
                dpi=500, edgecolor='b', format='png')
    plt.close(fig)
    return WW, XX, YY, ZZ


def _plot_3D_glyph_arclength(coeffs, fit_error, lmax, folder_name,
                             fit, optimized,
                             method_tag="arclength",
                             method_label="arc-length",
                             component_label=None,
                             sh_smooth=1.0):
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
    # Clip negative lobes to zero. Truncated SH reconstruction of a sharp
    # Watson PDF can dip slightly negative near the equator; clipping removes
    # those spurious lobes without altering the positive structure.
    if Ymax > 0:
        radii = np.clip(Yvals, 0, None)
    else:
        radii = np.zeros_like(Yvals)
    Ymax = radii.max() if radii.max() > 0 else 1.0
    x = radii * xx
    y = radii * yy
    z = radii * zz

    folder_path = folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Color by DIRECTION (unit-sphere coords), not by radius-scaled coords.
    # Using radius-scaled |x|,|y|,|z| makes every point near the equator black
    # (radius -> 0 so all RGB channels -> 0), producing a spurious dark ring /
    # pinched waist on sharp FODs. Unit-direction coloring is the standard
    # convention for FOD glyphs (e.g. MRtrix, Dipy).
    colors = np.zeros((xx.shape[0], xx.shape[1], 3))
    colors[:, :, 0] = np.abs(xx)
    colors[:, :, 1] = np.abs(yy)
    colors[:, :, 2] = np.abs(zz)
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
    component_title, component_tag = _component_suffix(component_label)
    title = (f"Along axon FOD{component_title} ({method_label}, lmax={lmax}) - {stage} fibers"
             + _kappa_title_suffix(fit))
    plt.title(title, fontsize=14, pad=20)
    fname = (f"FOD_3D_glyph{component_tag}_{method_tag}_direct_lmax{lmax}"
             f"_smooth{sh_smooth:g}_{_kappa_file_tag(fit)}")
    if not optimized:
        fname += "_preoptimized"
    plt.savefig(os.path.join(folder_path, fname + ".png"),
                dpi=500, edgecolor='b', format='png')
    plt.close(fig)


def _plot_3D_glyph_analytic_watson(fit, folder_name, optimized,
                                   method_tag="arclength",
                                   method_label="arc-length",
                                   component_label=None):
    """Render the fitted Watson distribution as a closed-form 3D glyph.

    Unlike the empirical glyph (``_plot_3D_glyph_arclength``), which is a
    truncated spherical-harmonic reconstruction of the *measured* tangents,
    this draws the analytic Watson PDF evaluated at the *fitted* concentration
    ``kappa`` about the fitted mean direction ``mu``:

        r(n) = exp(kappa * (n . mu)^2)

    Because it is a smooth closed-form surface (no SH truncation), it can
    never produce the Gibbs equatorial ring / pinched-waist artifact, and it
    separates concentrations strongly (a thin needle for high kappa vs a fat
    lobe for low kappa). The trade-off is that it visualises the fitted Watson
    *model*, not the raw data FOD, so it cannot reveal non-Watson structure.
    It is independent of ``lmax`` / ``sh_smooth``.
    """
    kappa = float(fit.get('kappa', np.nan))
    mu = np.asarray(fit.get('mu', np.array([0.0, 0.0, 1.0])), dtype=float)
    n = np.linalg.norm(mu)
    if n > 0:
        mu = mu / n
    if not np.isfinite(kappa):
        print("[_plot_3D_glyph_analytic_watson] kappa not finite; skipping.")
        return

    theta = np.linspace(0, 2 * np.pi, 180)
    phi = np.linspace(0, np.pi, 180)
    xx = np.outer(np.cos(theta), np.sin(phi))
    yy = np.outer(np.sin(theta), np.sin(phi))
    zz = np.outer(np.ones(np.size(theta)), np.cos(phi))

    # cos(angle to mu) for every direction on the unit sphere.
    cos_ang = xx * mu[0] + yy * mu[1] + zz * mu[2]
    # Watson PDF shape (drop the normalising constant; we rescale to max=1).
    # Subtract kappa before exp for numerical stability at large kappa
    # (equivalent to dividing by the peak value exp(kappa)).
    radii = np.exp(kappa * (cos_ang ** 2) - kappa)
    Ymax = radii.max() if radii.max() > 0 else 1.0

    x = radii * xx
    y = radii * yy
    z = radii * zz

    folder_path = folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Direction coloring (same convention as the empirical glyph).
    colors = np.zeros((xx.shape[0], xx.shape[1], 3))
    colors[:, :, 0] = np.abs(xx)
    colors[:, :, 1] = np.abs(yy)
    colors[:, :, 2] = np.abs(zz)
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
    component_title, component_tag = _component_suffix(component_label)
    title = (f"Watson model FOD{component_title} ({method_label}) - {stage} fibers"
             + _kappa_title_suffix(fit))
    plt.title(title, fontsize=14, pad=20)
    fname = f"FOD_3D_glyph{component_tag}_{method_tag}_watson_{_kappa_file_tag(fit)}"
    if not optimized:
        fname += "_preoptimized"
    plt.savefig(os.path.join(folder_path, fname + ".png"),
                dpi=500, edgecolor='b', format='png')
    plt.close(fig)


def _plot_watson_samples_achieved(tangents, fit, folder_name,
                                  optimized=True, max_points=3000,
                                  method_tag="arclength",
                                  method_label="arc-length",
                                  component_label=None):
    """Scatter plot of the achieved tangent vectors on the upper
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
    component_title, component_tag = _component_suffix(component_label)
    title = (f"Achieved tangent samples{component_title} ({method_label}) - {stage}"
             + _kappa_title_suffix(fit))
    ax.set_title(title, fontsize=14)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.tight_layout()

    fname = f"Watson_samples_achieved{component_tag}_{method_tag}_{_kappa_file_tag(fit)}"
    if not optimized:
        fname += "_preoptimized"
    plt.savefig(os.path.join(folder_path, fname + ".png"),
                dpi=500, edgecolor='b', format='png')
    plt.close(fig)


# =====================================================================
# Global (end-to-end) orientation statistics + Watson-kappa fitting
# One unit vector per fiber: (last_sphere_xyz - first_sphere_xyz), normalised.
# Captures the *global* bundle dispersion, independent of local waviness.
# =====================================================================

def _global_endpoint_tangents_and_fit(spheres_xyz_r_fid):
    """Extract one unit end-to-end vector per fiber and fit a Watson
    distribution on those vectors.

    Returns (tangents, fit_result). ``tangents`` has shape (n_fibers, 3).
    """
    if hasattr(spheres_xyz_r_fid, 'detach'):
        arr = spheres_xyz_r_fid.detach().cpu().numpy()
    else:
        arr = np.asarray(spheres_xyz_r_fid)

    L = _box_length_as_float()
    fiber_list = util.split_matrix_to_list(arr)
    fiber_list = [filter_spheres_outside_voxel(f, L) for f in fiber_list]
    fiber_list = [f for f in fiber_list if f.shape[0] >= 2]

    vecs = []
    for f in fiber_list:
        v = f[-1, :3] - f[0, :3]
        n = np.linalg.norm(v)
        if n > 0:
            vecs.append(v / n)
    if not vecs:
        tangents = np.zeros((0, 3))
    else:
        tangents = np.vstack(vecs)

    result = watson_fit.fit_watson_scatter(tangents)
    result['n_fibers'] = len(fiber_list)
    return tangents, result


def plot_global_axon_OD(spheres_xyz_r_fid, optimized=True):
    """Global (end-to-end) orientation statistics for the substrate.

    - One unit vector per fiber: (endpoint - startpoint), normalised.
    - Fits a bipolar Watson distribution via scatter-matrix MLE on those
      global orientation vectors.
    - Emits exactly two figures, both tagged ``global`` and carrying the
      fitted kappa / ODI in their filenames:
        * ``OD_histogram_global_*`` : smoothed orientation-density heatmap.
        * ``FOD_3D_glyph_global_watson_*`` : closed-form analytic Watson
          glyph (ring-free, separates concentrations strongly).
    """
    folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH + "/figs/substrate_stats/ODI"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    tangents, fit = _global_endpoint_tangents_and_fit(spheres_xyz_r_fid)

    k_prescribed = getattr(config_params, 'ORIENTATION_SHAPE_PARAM', None)
    try:
        k_prescribed = float(k_prescribed) if k_prescribed else None
    except Exception:
        k_prescribed = None
    fit['kappa_prescribed'] = k_prescribed

    if tangents.shape[0] < 3:
        print("[plot_global_axon_OD] Too few end-to-end vectors; skipping plots.")
        return fit

    _plot_OD_density_sphere_arclength(
        tangents, fit, optimized=optimized, folder_name=folder_path,
        method_tag="global", method_label="global")

    # --- analytic Watson model glyph (ring-free, lmax/smooth-independent) ---
    _plot_3D_glyph_analytic_watson(fit, folder_path, optimized=optimized,
                                   method_tag="global", method_label="global")
    return fit