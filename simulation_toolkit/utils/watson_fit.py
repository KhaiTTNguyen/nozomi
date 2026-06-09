"""
Fit a Watson distribution to the local-tangent orientation distribution of a
sphere-based fiber substrate using the arc-length method (a la Callaghan /
ConFiG substrate validation).

For each axon:
    1. Treat the sphere centers as control points of a smooth centerline.
    2. Fit a cubic spline parameterised by cumulative chord length.
    3. Resample the centerline at uniform arc-length intervals (ds).
    4. Take unit tangents at those uniform samples.

Pool the unit tangents across all axons, then fit a (bipolar) Watson
distribution via the orientation-tensor / scatter-matrix MLE:

    T = (1/M) sum_i  t_i t_i^T,  with eigvalues lam1 >= lam2 >= lam3
    mu    = eigenvector of lam1
    kappa = root of  lam1 = M'(1/2, 3/2, k) / M(1/2, 3/2, k)
          = root of  lam1 = (1/3) * M(3/2, 5/2, k) / M(1/2, 3/2, k)

ODI follows the NODDI convention: ODI = (2/pi) * arctan(1/kappa).


How kappa (K) and ODI are fitted (step by step)
-----------------------------------------------
1. Collect axial unit tangents ``t_i`` (``collect_arclength_tangents`` for the
   arc-length method, or one end-to-end vector per fiber for the global
   method). "Axial" means ``t_i`` and ``-t_i`` are equivalent.
2. Build the 3x3 orientation scatter matrix ``T = (1/M) sum_i t_i t_i^T``
   (``fit_watson_scatter``). T is symmetric positive semi-definite with
   eigenvalues ``lam1 >= lam2 >= lam3`` that sum to 1.
3. The principal eigenvector is the mean fiber direction ``mu``; the largest
   eigenvalue ``lam1`` measures concentration (``lam1 = 1/3`` isotropic,
   ``lam1 -> 1`` perfectly aligned).
4. Solve the Watson MLE relation ``lam1 = (1/3) * M(3/2, 5/2, k) / M(1/2, 3/2, k)``
   for ``kappa = k`` (``_kappa_from_lambda1``), where ``M`` is Kummer's
   confluent hypergeometric function. A bracketed Brent root-find is used,
   falling back to the large-kappa asymptote ``kappa ~ 1 / (2(1 - lam1))``.
5. Convert to the NODDI dispersion index ``ODI = (2/pi) * arctan(1/kappa)``
   in [0, 1]; small ODI = aligned, ODI -> 1 = isotropic.

Validation
----------
This estimator MUST be sanity-checked with
``tests/validation/orientation_fitting/test01_straight_axons_watson_fit.py``.
That test builds **perfectly straight** axons whose directions are drawn from a
Watson distribution with a prescribed ``K``. Because each axon is a single
straight line, it has no local waviness, so:

    * the arc-length fit and the global (end-to-end) fit should agree, and
    * both should recover ``K_fit ~= K_designed`` (equivalently ``ODI_fit``
      close to the designed ODI).

Any substantial divergence between the arc-length and global fits, or from the
designed K, on that straight-axon test indicates a regression in the fitting
pipeline rather than a real substrate property.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import hyp1f1


def _uniform_arclength_tangents(centers: np.ndarray,
                                ds: float | None = None,
                                min_samples: int = 20) -> np.ndarray:
    """Return unit tangent vectors sampled at uniform arc length along one fiber.

    Parameters
    ----------
    centers : (N, 3) sphere-center polyline (one axon, ordered along the fiber).
    ds : target arc-length spacing. If None, chosen automatically as
         max(total_length / 50, median(segment_length)).
    min_samples : minimum number of tangent samples requested per axon.

    Returns
    -------
    (M, 3) array of unit tangents (axial; signs arbitrary).
    """
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[0] < 2 or centers.shape[1] < 3:
        return np.empty((0, 3))
    centers = centers[:, :3]

    # Drop exact duplicates (CubicSpline requires strictly increasing s).
    diffs = np.diff(centers, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    keep = np.concatenate([[True], seg_lens > 1e-12])
    centers = centers[keep]
    if centers.shape[0] < 2:
        return np.empty((0, 3))

    diffs = np.diff(centers, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    L = s[-1]
    if L <= 0:
        return np.empty((0, 3))

    if ds is None:
        ds = max(L / 50.0, float(np.median(seg_lens)))
    n_samples = max(int(np.floor(L / ds)) + 1, min_samples)
    s_query = np.linspace(0.0, L, n_samples)

    if centers.shape[0] >= 4:
        cs = CubicSpline(s, centers, bc_type='natural', axis=0)
        tangents = cs(s_query, 1)  # first derivative -> local tangent
    else:
        # Fallback: linear interpolation + forward differences.
        pts = np.vstack([np.interp(s_query, s, centers[:, k]) for k in range(3)]).T
        tangents = np.gradient(pts, s_query, axis=0)

    norms = np.linalg.norm(tangents, axis=1)
    mask = norms > 1e-12
    return tangents[mask] / norms[mask, None]


def collect_arclength_tangents(fiber_list,
                               ds: float | None = None,
                               min_samples_per_fiber: int = 20) -> np.ndarray:
    """Pool unit tangents across all fibers using the arc-length method.

    Parameters
    ----------
    fiber_list : list of (N_i, >=3) arrays. First three columns are x, y, z.
    ds : uniform arc-length step (same units as centers). If None, per-fiber auto.
    """
    tangents = []
    for fiber in fiber_list:
        fiber = np.asarray(fiber)
        if fiber.shape[0] < 2:
            continue
        t = _uniform_arclength_tangents(fiber[:, :3], ds=ds,
                                        min_samples=min_samples_per_fiber)
        if t.shape[0] > 0:
            tangents.append(t)
    if not tangents:
        return np.empty((0, 3))
    return np.concatenate(tangents, axis=0)


def _kappa_from_lambda1(lam1: float) -> float:
    """Solve for Watson kappa (bipolar, kappa>0) given the largest eigenvalue
    of the orientation scatter matrix.

    Uses the MLE equation  lam1 = (1/3) * M(3/2, 5/2, k) / M(1/2, 3/2, k),
    where M is Kummer's confluent hypergeometric function (scipy.special.hyp1f1).

    For numerical robustness at large kappa we fall back to the asymptotic
    approximation  kappa ~ 1 / (2 * (1 - lam1)).
    """
    # Isotropic limit
    if lam1 <= 1.0 / 3.0 + 1e-9:
        return 0.0

    def g(k: float) -> float:
        # hyp1f1 may overflow for very large k; catch that and fall back.
        num = hyp1f1(1.5, 2.5, k)
        den = hyp1f1(0.5, 1.5, k)
        if not np.isfinite(num) or not np.isfinite(den) or den == 0.0:
            return np.nan
        return (num / den) / 3.0 - lam1

    # Bracket search.
    k_lo = 1e-6
    k_hi = 1.0
    g_lo = g(k_lo)
    g_hi = g(k_hi)
    # Expand upper bracket until sign change or numeric failure.
    while np.isfinite(g_hi) and g_hi < 0.0 and k_hi < 1e8:
        k_hi *= 2.0
        g_hi = g(k_hi)

    if not np.isfinite(g_hi) or g_lo * g_hi > 0:
        # Numeric failure -> asymptotic approximation.
        return float(1.0 / max(2.0 * (1.0 - lam1), 1e-12))

    try:
        return float(brentq(g, k_lo, k_hi, xtol=1e-4, rtol=1e-6, maxiter=200))
    except Exception:
        return float(1.0 / max(2.0 * (1.0 - lam1), 1e-12))


def fit_watson_scatter(tangents: np.ndarray) -> dict:
    """Fit a bipolar Watson distribution to axial unit vectors via the
    scatter-matrix MLE.

    Returns a dict with keys:
        kappa, ODI, mu, eigvals, n_samples, lambda1
    where ODI = (2/pi) * arctan(1/kappa), following the NODDI convention.
    """
    tangents = np.asarray(tangents, dtype=float)
    n = tangents.shape[0]
    if n < 3:
        return {'kappa': np.nan, 'ODI': np.nan,
                'mu': np.array([0.0, 0.0, 1.0]),
                'eigvals': np.array([np.nan] * 3),
                'n_samples': n, 'lambda1': np.nan}

    T = (tangents.T @ tangents) / n
    eigvals, eigvecs = np.linalg.eigh(T)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    mu = eigvecs[:, 0]
    # Convention: mu has non-negative z component (axial sign is arbitrary).
    if mu[2] < 0:
        mu = -mu
    lam1 = float(eigvals[0])
    kappa = _kappa_from_lambda1(lam1)
    odi = (2.0 / np.pi) * np.arctan(1.0 / kappa) if kappa > 0 else 1.0

    return {
        'kappa': float(kappa),
        'ODI': float(odi),
        'mu': mu.astype(float),
        'eigvals': eigvals.astype(float),
        'n_samples': int(n),
        'lambda1': lam1,
    }


def fit_watson_from_fibers(fiber_list,
                           ds: float | None = None) -> dict:
    """Convenience wrapper: arc-length resample + Watson scatter-matrix MLE."""
    tangents = collect_arclength_tangents(fiber_list, ds=ds)
    result = fit_watson_scatter(tangents)
    result['ds'] = ds
    result['n_fibers'] = int(sum(1 for f in fiber_list if len(f) >= 2))
    return result
