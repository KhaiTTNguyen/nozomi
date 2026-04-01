from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import erf, i0e, jv, yv

from .core import DiffusionPulseSequence, SignalModel, coerce_parameter_matrix


def add_rician_noise(data: Any, sigma: float | None) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if sigma is None:
        return data.copy()
    sigma = float(sigma)
    sx = sigma * np.random.randn(*data.shape) + data
    sy = sigma * np.random.randn(*data.shape)
    return np.sqrt(sx**2 + sy**2)


def free_water_diffusivity(T: Any) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    D0 = 16.35
    Ts = 215.05
    gamma = 2.063
    return D0 * (((T + 273.15) / Ts) - 1.0) ** gamma


def _load_table_or_array(value: Any) -> np.ndarray:
    if isinstance(value, (str, Path)):
        return np.loadtxt(str(value), dtype=float)
    return np.asarray(value, dtype=float)


def _filled_parameter(n: int, value: float | None) -> np.ndarray:
    fill_value = np.nan if value is None else float(value)
    return np.full(int(n), fill_value, dtype=float)


def _normalize_gdir(gdir: Any) -> np.ndarray:
    gdir_arr = np.asarray(gdir, dtype=float)
    norms = np.linalg.norm(gdir_arr, axis=1, keepdims=True)
    nz = norms.squeeze(-1) > np.finfo(float).eps
    out = gdir_arr.copy()
    out[nz] = out[nz] / norms[nz]
    out[~nz] = np.array([0.0, 0.0, 0.0], dtype=float)
    return out


def gradient_table_to_pulse(
    *,
    grad: Any | None = None,
    grads: Any | None = None,
    bvals: Any | None = None,
    bvecs: Any | None = None,
    gradient_format: str = "auto",
    b0_threshold_smm2: float = 20.0,
    shape: str = "tpgse",
    delta_ms: float | None = None,
    Delta_ms: float | None = None,
    trise_ms: float | None = None,
    TE_ms: float | None = None,
) -> DiffusionPulseSequence:
    """
    Convert an MRtrix or FSL-style gradient table into a MATI diffusion pulse.

    Parameters
    ----------
    grad, grads
        MRtrix gradient table as Nx4 array or text file with columns
        ``gx gy gz b`` in s/mm^2.
    bvals, bvecs
        FSL gradient table inputs as arrays or text files.
        ``bvals`` is length-N in s/mm^2; ``bvecs`` is 3xN or Nx3.
    gradient_format
        One of ``"auto"``, ``"grad"``, or ``"fslgrad"``.
    """
    if grad is not None and grads is not None:
        raise ValueError("Provide only one of grad=... or grads=....")
    grad_table = grads if grads is not None else grad

    format_key = str(gradient_format).strip().lower()
    format_aliases = {
        "auto": "auto",
        "grad": "grad",
        "grads": "grad",
        "mrtrix": "grad",
        "mrtrixgrad": "grad",
        "fsl": "fslgrad",
        "fslgrad": "fslgrad",
    }
    if format_key not in format_aliases:
        raise ValueError("gradient_format must be one of: auto, grad, fslgrad.")
    selected_format = format_aliases[format_key]

    has_grad = grad_table is not None
    has_fsl = bvals is not None or bvecs is not None
    if selected_format == "auto":
        if has_grad and not has_fsl:
            selected_format = "grad"
        elif has_fsl and not has_grad:
            selected_format = "fslgrad"
        elif has_grad and has_fsl:
            raise ValueError("Provide either grad(s)=... or both bvals=... and bvecs=..., not both.")
        else:
            raise ValueError("Provide either grad(s)=... or both bvals=... and bvecs=....")

    if selected_format == "grad":
        if grad_table is None:
            raise ValueError("gradient_format='grad' requires grad=... or grads=....")
        if bvals is not None or bvecs is not None:
            raise ValueError("gradient_format='grad' cannot be combined with bvals/bvecs.")
        grads_arr = _load_table_or_array(grad_table)
        if grads_arr.ndim != 2 or grads_arr.shape[1] != 4:
            raise ValueError(f"MRtrix gradient table must have shape Nx4, got {grads_arr.shape}.")
        gdir = grads_arr[:, :3]
        bvals_smm2 = grads_arr[:, 3].reshape(-1)
    else:
        if bvals is None or bvecs is None:
            raise ValueError("Provide either grads=... or both bvals=... and bvecs=....")
        bvals_smm2 = _load_table_or_array(bvals).reshape(-1)
        bvecs_arr = _load_table_or_array(bvecs)
        if bvecs_arr.shape == (3, bvals_smm2.size):
            gdir = bvecs_arr.T
        elif bvecs_arr.shape == (bvals_smm2.size, 3):
            gdir = bvecs_arr
        else:
            raise ValueError(f"FSL bvecs must have shape 3xN or Nx3, got {bvecs_arr.shape}.")

    gdir = _normalize_gdir(gdir)
    bvals_msum2 = np.asarray(bvals_smm2, dtype=float) / 1e3
    bvals_msum2[bvals_msum2 < (float(b0_threshold_smm2) / 1e3)] = 0.0
    n = bvals_msum2.size
    payload: dict[str, Any] = {
        "Nacq": int(n),
        "b": bvals_msum2,
        "shape": [str(shape)] * int(n),
        "delta": _filled_parameter(int(n), delta_ms),
        "Delta": _filled_parameter(int(n), Delta_ms),
        "n": _filled_parameter(int(n), None),
        "trise": _filled_parameter(int(n), trise_ms),
        "gdir": gdir,
    }
    payload["TE"] = _filled_parameter(int(n), TE_ms)
    return DiffusionPulseSequence(**payload)


def _log_besseli0(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    mask = x < 700.0
    out[mask] = np.log(i0e(x[mask])) + x[mask]
    out[~mask] = x[~mask] - 0.5 * np.log(2.0 * np.pi * x[~mask])
    return out


def log_likelihood(E: Any, S: Any, sigma: Any) -> np.ndarray:
    E = np.asarray(E, dtype=float)
    S = np.asarray(S, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    logliks = np.log(np.clip(S, np.finfo(float).tiny, None)) - np.log(sigma**2) + _log_besseli0(E * S / (sigma**2)) - (E**2 + S**2) / (2.0 * sigma**2)
    return np.squeeze(np.sum(logliks, axis=0))


def _sphbesselj1_core_prim(x: np.ndarray) -> np.ndarray:
    return -2.0 * np.sin(x) + 2.0 * x * np.cos(x) + x**2 * np.sin(x)


def _sphbessely1_core_prim(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.cos(x) + 2.0 * x * np.sin(x) - x**2 * np.cos(x)


def _sphbesselj(n: int, x: np.ndarray) -> np.ndarray:
    return jv(n + 0.5, x) / np.sqrt(x)


def _sphbessely(n: int, x: np.ndarray) -> np.ndarray:
    return yv(n + 0.5, x) / np.sqrt(x)


def _sphbesselj_prim(n: int, x: np.ndarray) -> np.ndarray:
    return n / x * _sphbesselj(n, x) - _sphbesselj(n + 1, x)


def _safe_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(x, -700.0, 700.0))


def _load_bk_ak(radius: np.ndarray, geometry: str) -> tuple[np.ndarray, np.ndarray]:
    radius = np.asarray(radius, dtype=float).reshape(1, -1, 1)
    nr = radius.shape[1]
    geometry = geometry.lower()
    if geometry == "plane":
        krange = np.arange(1, 31, dtype=float).reshape(1, 1, -1)
        krange = np.tile(krange, (1, nr, 1))
        bk = 8.0 * (2.0 * radius) ** 2 / np.pi**4 / (2.0 * krange - 1.0) ** 4
        ak = np.pi**2 * (2.0 * krange - 1.0) ** 2 / (2.0 * radius) ** 2
        return bk, ak
    if geometry == "cylinder":
        muk = np.array([1.8412, 5.3315, 8.5364, 11.7061, 14.8636, 18.0156, 21.1644, 24.3114, 27.4571, 30.6020, 33.7462, 36.8900, 40.0335, 43.1767, 46.3196, 49.4624, 52.6051, 55.7476, 58.8901, 62.0324, 65.1747, 68.3169, 71.4590, 74.6011, 77.7432, 80.8852, 84.0272, 87.1692, 90.3111], dtype=float).reshape(1, 1, -1)
        muk = np.tile(muk, (1, nr, 1))
        bk = 2.0 * (radius / muk) ** 2 / (muk**2 - 1.0)
        ak = (muk / radius) ** 2
        return bk, ak
    if geometry == "sphere":
        muk = np.array([2.082, 5.941, 9.206, 12.405, 15.580, 18.743, 21.900, 25.053, 28.204, 31.353, 34.500, 37.646, 40.792, 43.937, 47.082, 50.226, 53.370, 56.514, 59.657, 62.801, 65.944, 69.087, 72.229, 75.372, 78.515, 81.657, 84.800, 87.942, 91.085], dtype=float).reshape(1, 1, -1)
        muk = np.tile(muk, (1, nr, 1))
        bk = 2.0 * (radius / muk) ** 2 / (muk**2 - 2.0)
        ak = (muk / radius) ** 2
        return bk, ak
    raise ValueError(f"Unsupported geometry {geometry!r}.")


def _get_hollow_sphere_roots(rin: np.ndarray, rout: np.ndarray, *, num_roots: int = 20, xstep: float = 1e-5, xmax: float = 100.0) -> np.ndarray:
    x = np.arange(xstep, xmax + xstep, xstep, dtype=float).reshape(-1, 1)
    y1 = _sphbessely1_core_prim(x * rin) * _sphbesselj1_core_prim(x * rout) - _sphbesselj1_core_prim(x * rin) * _sphbessely1_core_prim(x * rout)
    x2 = x + xstep
    y2 = _sphbessely1_core_prim(x2 * rin) * _sphbesselj1_core_prim(x2 * rout) - _sphbesselj1_core_prim(x2 * rin) * _sphbessely1_core_prim(x2 * rout)
    out = np.zeros((num_roots, rin.shape[1]), dtype=float)
    for idx in range(rin.shape[1]):
        roots = x2[(y1[:, idx] * y2[:, idx]) < 0.0, 0]
        if roots.size < num_roots:
            raise ValueError("Not enough hollow-sphere roots were found; reduce xstep or increase xmax.")
        out[:, idx] = roots[:num_roots]
    return out


def _load_bk_ak_hollow(rin: np.ndarray, rout: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lambdas = _get_hollow_sphere_roots(rin.reshape(1, -1), rout.reshape(1, -1))
    num_lambda = lambdas.shape[0]
    rin2 = np.tile(rin.reshape(1, -1), (num_lambda, 1))
    rout2 = np.tile(rout.reshape(1, -1), (num_lambda, 1))
    ratio = rout2**3 / rin2**3 * _sphbesselj1_core_prim(lambdas * rin2) / _sphbesselj1_core_prim(lambdas * rout2)
    bk = (
        2.0
        * rin2**3
        * rout2**3
        / (rout2**3 - rin2**3)
        / lambdas**2
        * (ratio - 1.0) ** 2
        / (rin2**3 * (lambdas**2 * rout2**2 - 2.0) * ratio**2 - rout2**3 * (lambdas**2 * rin2**2 - 2.0))
    )
    ak = lambdas**2
    return np.transpose(bk, (1, 0))[np.newaxis, ...], np.transpose(ak, (1, 0))[np.newaxis, ...]


def _apply_tpgse_formula(akd: np.ndarray, Delta: np.ndarray, trise: np.ndarray, tp: np.ndarray) -> np.ndarray:
    return (
        0.25
        * (
            12.0 * _safe_exp(-akd * tp)
            - 24.0 * _safe_exp(-akd * trise)
            - 24.0 * _safe_exp(-Delta * akd)
            - 24.0 * akd * trise
            + 12.0 * _safe_exp(-akd * (Delta - trise))
            + 12.0 * _safe_exp(-akd * (Delta + trise))
            - 6.0 * _safe_exp(-akd * (Delta - tp))
            - 6.0 * _safe_exp(-akd * (Delta + tp))
            - 24.0 * _safe_exp(-akd * (trise + tp))
            + 12.0 * _safe_exp(-akd * (2.0 * trise + tp))
            + 8.0 * akd**3 * trise**3
            + 12.0 * akd**3 * trise**2 * tp
            + 12.0 * _safe_exp(-akd * (Delta - trise - tp))
            + 12.0 * _safe_exp(-akd * (Delta + trise + tp))
            - 6.0 * _safe_exp(-akd * (Delta - 2.0 * trise - tp))
            - 6.0 * _safe_exp(-akd * (Delta + 2.0 * trise + tp))
            + 24.0
        )
        / (3.0 * akd**4 * trise**2)
    )


def restricted_dwi_signal(parms: Any, model: SignalModel) -> np.ndarray:
    geometry = str(model.structure.get("geometry", "sphere"))
    if geometry.lower() in {"plane", "cylinder", "sphere"}:
        p = coerce_parameter_matrix(parms, 2)
        if np.any(p[0] <= 0) or np.any(p[1] <= 0):
            raise ValueError("RestrictedDWISignal requires positive d and D.")
        if p.shape[1] > 1:
            uniq, inverse = np.unique(p.T, axis=0, return_inverse=True)
            p_unique = uniq.T
        else:
            p_unique = p
            inverse = None
        radius = p_unique[0] / 2.0
        diffusivity = p_unique[1]
        bk, ak = _load_bk_ak(radius, geometry)
    elif geometry.lower() in {"hollowsphere", "sphericalshell"}:
        p = coerce_parameter_matrix(parms, 3)
        if np.any(p[0] >= p[1]) or np.any(p[2] <= 0):
            raise ValueError("RestrictedDWISignal requires din < dout and D > 0 for hollow spheres.")
        if p.shape[1] > 1:
            uniq, inverse = np.unique(p.T, axis=0, return_inverse=True)
            p_unique = uniq.T
        else:
            p_unique = p
            inverse = None
        rin = p_unique[0] / 2.0
        rout = p_unique[1] / 2.0
        diffusivity = p_unique[2]
        bk, ak = _load_bk_ak_hollow(rin, rout)
    else:
        raise ValueError(f"Unsupported geometry: {geometry}")
    pulse = model.pulse
    shape = np.char.lower(np.asarray(pulse.shape, dtype=str))
    ind_pgse = shape == "pgse"
    ind_tpgse = shape == "tpgse"
    ind_sin = shape == "sin"
    ind_cos = shape == "cos"
    ind_tcos1 = (shape == "tcos") & (np.asarray(pulse.n) == 1)
    ind_tcos2 = (shape == "tcos") & (np.asarray(pulse.n) == 2)
    ind_tcos3 = (shape == "tcos") & (np.asarray(pulse.n) == 3)
    ind_tsin1 = (shape == "tsin") & (np.asarray(pulse.n) == 1)
    ind_tsin2 = (shape == "tsin") & (np.asarray(pulse.n) == 2)

    _, nr, nk = bk.shape
    nacq = pulse.Nacq
    bk3 = np.tile(bk, (nacq, 1, 1))
    ak3 = np.tile(ak, (nacq, 1, 1))
    D3 = np.tile(np.asarray(diffusivity, dtype=float).reshape(1, nr, 1), (nacq, 1, nk))
    akd = ak3 * D3
    delta = np.tile(np.asarray(pulse.delta, dtype=float).reshape(nacq, 1, 1), (1, nr, nk))
    Delta = np.tile(np.asarray(pulse.Delta, dtype=float).reshape(nacq, 1, 1), (1, nr, nk))

    Etmp = (akd * delta - 1.0 + _safe_exp(-akd * delta) + _safe_exp(-akd * Delta) - 0.5 * _safe_exp(-akd * (Delta + delta)) - 0.5 * _safe_exp(-akd * (Delta - delta))) / (ak3**2 * D3**2)

    if np.any(ind_tpgse):
        trise = np.tile(np.asarray(pulse.trise, dtype=float).reshape(nacq, 1, 1), (1, nr, nk))
        tp = np.tile(np.asarray(pulse.tp, dtype=float).reshape(nacq, 1, 1), (1, nr, nk))
        Etmp_tpgse = _apply_tpgse_formula(akd, Delta, trise, tp)
        Etmp[ind_tpgse, :, :] = Etmp_tpgse[ind_tpgse, :, :]

    if np.any(ind_sin):
        w = np.tile(np.asarray(pulse.w, dtype=float).reshape(nacq, 1, 1), (1, nr, nk))
        Etmp_sin = w**2 / (ak3**2 * D3**2 + w**2) ** 2 * (akd * (ak3**2 * D3**2 + w**2) * delta / (2.0 * w**2) + 1.0 - _safe_exp(-akd * delta) - _safe_exp(-akd * Delta) + 0.5 * _safe_exp(-akd * (delta + Delta)) + 0.5 * _safe_exp(-akd * (Delta - delta)))
        Etmp[ind_sin, :, :] = Etmp_sin[ind_sin, :, :]

    if np.any(ind_cos):
        w = np.tile(np.asarray(pulse.w, dtype=float).reshape(nacq, 1, 1), (1, nr, nk))
        Etmp_cos = ak3**2 * D3**2 / (ak3**2 * D3**2 + w**2) ** 2 * (((ak3**2 * D3**2 + w**2) * delta) / (2.0 * ak3 * D3) - 1.0 + _safe_exp(-akd * delta) + _safe_exp(-akd * Delta) - 0.5 * _safe_exp(-akd * (delta + Delta)) - 0.5 * _safe_exp(-akd * (Delta - delta)))
        Etmp[ind_cos, :, :] = Etmp_cos[ind_cos, :, :]

    if np.any(ind_tcos1):
        idx = np.flatnonzero(ind_tcos1)
        exp = _safe_exp
        akd_t = akd[idx, :, :]
        Delta_t = Delta[idx, :, :]
        trise = np.asarray(pulse.trise, dtype=float)[idx].reshape(-1, 1, 1)
        tp = np.asarray(pulse.tp, dtype=float)[idx].reshape(-1, 1, 1)
        Etmp_tcos1 = (
            0.25
            * (
                8 * exp(-akd_t * tp) - 8 * exp(-akd_t * trise) - 8 * exp(-2 * akd_t * trise) - 16 * exp(-akd_t * Delta_t) - 24 * akd_t * trise
                + 4 * exp(-akd_t * (Delta_t - trise)) + 4 * exp(-akd_t * (Delta_t + trise)) + 4 * exp(-akd_t * (Delta_t + 2 * trise)) + 4 * exp(-akd_t * (Delta_t - 2 * trise))
                - 4 * exp(-akd_t * (Delta_t - tp)) - 4 * exp(-akd_t * (Delta_t + tp)) - 8 * exp(-akd_t * (trise + tp)) + 4 * exp(-akd_t * (trise + 2 * tp))
                - 8 * exp(-akd_t * (2 * trise + tp)) + 8 * exp(-akd_t * (3 * trise + tp)) - 8 * exp(-akd_t * (3 * trise + 2 * tp)) - 8 * exp(-akd_t * (3 * trise + 3 * tp))
                + 8 * exp(-akd_t * (4 * trise + 3 * tp)) + 4 * exp(-akd_t * (5 * trise + 2 * tp)) + 8 * exp(-akd_t * (5 * trise + 3 * tp)) + 4 * exp(-akd_t * (5 * trise + 4 * tp))
                - 8 * exp(-akd_t * (6 * trise + 3 * tp)) - 8 * exp(-akd_t * (6 * trise + 4 * tp)) + 4 * exp(-akd_t * (7 * trise + 4 * tp))
                + 12 * akd_t**3 * trise**3 + 16 * akd_t**3 * trise**2 * tp + 4 * exp(-akd_t * (Delta_t - trise - tp)) - 2 * exp(-akd_t * (Delta_t - trise - 2 * tp))
                + 4 * exp(-akd_t * (Delta_t + trise + tp)) + 4 * exp(-akd_t * (Delta_t - 2 * trise - tp)) - 2 * exp(-akd_t * (Delta_t + trise + 2 * tp)) + 4 * exp(-akd_t * (Delta_t + 2 * trise + tp))
                - 4 * exp(-akd_t * (Delta_t - 3 * trise - tp)) - 4 * exp(-akd_t * (Delta_t + 3 * trise + tp)) + 4 * exp(-akd_t * (Delta_t + 3 * trise + 2 * tp)) + 4 * exp(-akd_t * (Delta_t - 3 * trise - 2 * tp))
                + 4 * exp(-akd_t * (Delta_t + 3 * trise + 3 * tp)) + 4 * exp(-akd_t * (Delta_t - 3 * trise - 3 * tp)) - 4 * exp(-akd_t * (Delta_t + 4 * trise + 3 * tp)) - 4 * exp(-akd_t * (Delta_t - 4 * trise - 3 * tp))
                - 2 * exp(-akd_t * (Delta_t + 5 * trise + 2 * tp)) - 2 * exp(-akd_t * (Delta_t - 5 * trise - 2 * tp)) - 4 * exp(-akd_t * (Delta_t + 5 * trise + 3 * tp)) - 4 * exp(-akd_t * (Delta_t - 5 * trise - 3 * tp))
                - 2 * exp(-akd_t * (Delta_t + 5 * trise + 4 * tp)) - 2 * exp(-akd_t * (Delta_t - 5 * trise - 4 * tp)) + 4 * exp(-akd_t * (Delta_t + 6 * trise + 3 * tp)) + 4 * exp(-akd_t * (Delta_t - 6 * trise - 3 * tp))
                + 4 * exp(-akd_t * (Delta_t + 6 * trise + 4 * tp)) + 4 * exp(-akd_t * (Delta_t - 6 * trise - 4 * tp)) - 2 * exp(-akd_t * (Delta_t + 7 * trise + 4 * tp)) - 2 * exp(-akd_t * (Delta_t - 7 * trise - 4 * tp)) + 16
            )
            / (akd_t**4 * trise**2)
        )
        Etmp[idx, :, :] = Etmp_tcos1

    if np.any(ind_tcos2):
        idx = np.flatnonzero(ind_tcos2)
        exp = _safe_exp
        akd_t = akd[idx, :, :]
        Delta_t = Delta[idx, :, :]
        trise = np.asarray(pulse.trise, dtype=float)[idx].reshape(-1, 1, 1)
        tp = np.asarray(pulse.tp, dtype=float)[idx].reshape(-1, 1, 1)
        Etmp_tcos2 = (
            0.25
            * (
                24*exp(-akd_t*tp) - 24*exp(-akd_t*trise) - 48*exp(-2*akd_t*trise) - 72*exp(-akd_t*Delta_t) - 120*akd_t*trise
                + 12*exp(-akd_t*(Delta_t-trise)) + 12*exp(-akd_t*(Delta_t+trise)) + 24*exp(-akd_t*(Delta_t+2*trise)) + 24*exp(-akd_t*(Delta_t-2*trise))
                - 12*exp(-akd_t*(Delta_t-tp)) - 12*exp(-akd_t*(Delta_t+tp)) - 24*exp(-akd_t*(trise+tp)) + 36*exp(-akd_t*(trise+2*tp))
                - 24*exp(-akd_t*(2*trise+tp)) + 24*exp(-akd_t*(3*trise+tp)) - 72*exp(-akd_t*(3*trise+2*tp)) - 24*exp(-akd_t*(3*trise+3*tp))
                + 24*exp(-akd_t*(4*trise+3*tp)) + 36*exp(-akd_t*(5*trise+2*tp)) - 24*exp(-akd_t*(4*trise+4*tp)) + 24*exp(-akd_t*(5*trise+3*tp))
                - 24*exp(-akd_t*(6*trise+3*tp)) + 48*exp(-akd_t*(6*trise+4*tp)) + 24*exp(-akd_t*(6*trise+5*tp)) - 24*exp(-akd_t*(7*trise+5*tp))
                - 24*exp(-akd_t*(8*trise+4*tp)) + 12*exp(-akd_t*(7*trise+6*tp)) - 24*exp(-akd_t*(8*trise+5*tp)) + 24*exp(-akd_t*(9*trise+5*tp))
                - 24*exp(-akd_t*(9*trise+6*tp)) - 24*exp(-akd_t*(9*trise+7*tp)) + 24*exp(-akd_t*(10*trise+7*tp)) + 12*exp(-akd_t*(11*trise+6*tp))
                + 24*exp(-akd_t*(11*trise+7*tp)) + 12*exp(-akd_t*(11*trise+8*tp)) - 24*exp(-akd_t*(12*trise+7*tp)) - 24*exp(-akd_t*(12*trise+8*tp))
                + 12*exp(-akd_t*(13*trise+8*tp)) + 76*akd_t**3*trise**3 + 96*akd_t**3*trise**2*tp
                + 12*exp(-akd_t*(Delta_t-trise-tp)) - 18*exp(-akd_t*(Delta_t-trise-2*tp)) + 12*exp(-akd_t*(Delta_t+trise+tp)) + 12*exp(-akd_t*(Delta_t-2*trise-tp))
                - 18*exp(-akd_t*(Delta_t+trise+2*tp)) + 12*exp(-akd_t*(Delta_t+2*trise+tp)) - 12*exp(-akd_t*(Delta_t-3*trise-tp)) - 12*exp(-akd_t*(Delta_t+3*trise+tp))
                + 36*exp(-akd_t*(Delta_t+3*trise+2*tp)) + 36*exp(-akd_t*(Delta_t-3*trise-2*tp)) + 12*exp(-akd_t*(Delta_t+3*trise+3*tp)) + 12*exp(-akd_t*(Delta_t-3*trise-3*tp))
                - 12*exp(-akd_t*(Delta_t+4*trise+3*tp)) - 12*exp(-akd_t*(Delta_t-4*trise-3*tp)) - 18*exp(-akd_t*(Delta_t+5*trise+2*tp)) - 18*exp(-akd_t*(Delta_t-5*trise-2*tp))
                + 12*exp(-akd_t*(Delta_t+4*trise+4*tp)) + 12*exp(-akd_t*(Delta_t-4*trise-4*tp)) - 12*exp(-akd_t*(Delta_t+5*trise+3*tp)) - 12*exp(-akd_t*(Delta_t-5*trise-3*tp))
                + 12*exp(-akd_t*(Delta_t+6*trise+3*tp)) + 12*exp(-akd_t*(Delta_t-6*trise-3*tp)) - 24*exp(-akd_t*(Delta_t+6*trise+4*tp)) - 24*exp(-akd_t*(Delta_t-6*trise-4*tp))
                - 12*exp(-akd_t*(Delta_t+6*trise+5*tp)) - 12*exp(-akd_t*(Delta_t-6*trise-5*tp)) + 12*exp(-akd_t*(Delta_t+7*trise+5*tp)) + 12*exp(-akd_t*(Delta_t-7*trise-5*tp))
                + 12*exp(-akd_t*(Delta_t+8*trise+4*tp)) + 12*exp(-akd_t*(Delta_t-8*trise-4*tp)) - 6*exp(-akd_t*(Delta_t+7*trise+6*tp)) - 6*exp(-akd_t*(Delta_t-7*trise-6*tp))
                + 12*exp(-akd_t*(Delta_t+8*trise+5*tp)) + 12*exp(-akd_t*(Delta_t-8*trise-5*tp)) - 12*exp(-akd_t*(Delta_t+9*trise+5*tp)) - 12*exp(-akd_t*(Delta_t-9*trise-5*tp))
                + 12*exp(-akd_t*(Delta_t+9*trise+6*tp)) + 12*exp(-akd_t*(Delta_t-9*trise-6*tp)) + 12*exp(-akd_t*(Delta_t+9*trise+7*tp)) + 12*exp(-akd_t*(Delta_t-9*trise-7*tp))
                - 12*exp(-akd_t*(Delta_t+10*trise+7*tp)) - 12*exp(-akd_t*(Delta_t-10*trise-7*tp)) - 6*exp(-akd_t*(Delta_t+11*trise+6*tp)) - 6*exp(-akd_t*(Delta_t-11*trise-6*tp))
                - 12*exp(-akd_t*(Delta_t+11*trise+7*tp)) - 12*exp(-akd_t*(Delta_t-11*trise-7*tp)) - 6*exp(-akd_t*(Delta_t+11*trise+8*tp)) - 6*exp(-akd_t*(Delta_t-11*trise-8*tp))
                + 12*exp(-akd_t*(Delta_t+12*trise+7*tp)) + 12*exp(-akd_t*(Delta_t-12*trise-7*tp)) + 12*exp(-akd_t*(Delta_t+12*trise+8*tp)) + 12*exp(-akd_t*(Delta_t-12*trise-8*tp))
                - 6*exp(-akd_t*(Delta_t+13*trise+8*tp)) - 6*exp(-akd_t*(Delta_t-13*trise-8*tp)) + 72
            ) / (3*akd_t**4*trise**2)
        )
        Etmp[idx, :, :] = Etmp_tcos2

    if np.any(ind_tcos3):
        idx = np.flatnonzero(ind_tcos3)
        exp = _safe_exp
        akd_t = akd[idx, :, :]
        Delta_t = Delta[idx, :, :]
        trise = np.asarray(pulse.trise, dtype=float)[idx].reshape(-1, 1, 1)
        tp = np.asarray(pulse.tp, dtype=float)[idx].reshape(-1, 1, 1)
        Etmp_tcos3 = (
            0.25 * (
                24*exp(-akd_t*tp) - 24*exp(-akd_t*trise) - 72*exp(-2*akd_t*trise) - 96*exp(-Delta_t*akd_t) - 168*akd_t*trise
                + 12*exp(-akd_t*(Delta_t-trise)) + 12*exp(-akd_t*(Delta_t+trise)) + 36*exp(-akd_t*(Delta_t+2*trise)) + 36*exp(-akd_t*(Delta_t-2*trise))
                - 12*exp(-akd_t*(Delta_t-tp)) - 12*exp(-akd_t*(Delta_t+tp)) - 24*exp(-akd_t*(trise+tp)) + 60*exp(-akd_t*(trise+2*tp))
                - 24*exp(-akd_t*(2*trise+tp)) + 24*exp(-akd_t*(3*trise+tp)) - 120*exp(-akd_t*(3*trise+2*tp)) - 24*exp(-akd_t*(3*trise+3*tp))
                + 24*exp(-akd_t*(4*trise+3*tp)) + 60*exp(-akd_t*(5*trise+2*tp)) - 48*exp(-akd_t*(4*trise+4*tp)) + 24*exp(-akd_t*(5*trise+3*tp))
                - 24*exp(-akd_t*(6*trise+3*tp)) + 96*exp(-akd_t*(6*trise+4*tp)) + 24*exp(-akd_t*(6*trise+5*tp)) - 24*exp(-akd_t*(7*trise+5*tp))
                - 48*exp(-akd_t*(8*trise+4*tp)) + 36*exp(-akd_t*(7*trise+6*tp)) - 24*exp(-akd_t*(8*trise+5*tp)) + 24*exp(-akd_t*(9*trise+5*tp))
                - 72*exp(-akd_t*(9*trise+6*tp)) - 24*exp(-akd_t*(9*trise+7*tp)) + 24*exp(-akd_t*(10*trise+7*tp)) + 36*exp(-akd_t*(11*trise+6*tp))
                - 24*exp(-akd_t*(10*trise+8*tp)) + 24*exp(-akd_t*(11*trise+7*tp)) - 24*exp(-akd_t*(12*trise+7*tp)) + 48*exp(-akd_t*(12*trise+8*tp))
                + 24*exp(-akd_t*(12*trise+9*tp)) - 24*exp(-akd_t*(13*trise+9*tp)) - 24*exp(-akd_t*(14*trise+8*tp)) + 12*exp(-akd_t*(13*trise+10*tp))
                - 24*exp(-akd_t*(14*trise+9*tp)) + 24*exp(-akd_t*(15*trise+9*tp)) - 24*exp(-akd_t*(15*trise+10*tp)) - 24*exp(-akd_t*(15*trise+11*tp))
                + 24*exp(-akd_t*(16*trise+11*tp)) + 12*exp(-akd_t*(17*trise+10*tp)) + 24*exp(-akd_t*(17*trise+11*tp)) + 12*exp(-akd_t*(17*trise+12*tp))
                - 24*exp(-akd_t*(18*trise+11*tp)) - 24*exp(-akd_t*(18*trise+12*tp)) + 12*exp(-akd_t*(19*trise+12*tp))
                + 116*akd_t**3*trise**3 + 144*akd_t**3*trise**2*tp + 12*exp(-akd_t*(Delta_t-trise-tp)) - 30*exp(-akd_t*(Delta_t-trise-2*tp))
                + 12*exp(-akd_t*(Delta_t+trise+tp)) + 12*exp(-akd_t*(Delta_t-2*trise-tp)) - 30*exp(-akd_t*(Delta_t+trise+2*tp)) + 12*exp(-akd_t*(Delta_t+2*trise+tp))
                - 12*exp(-akd_t*(Delta_t-3*trise-tp)) - 12*exp(-akd_t*(Delta_t+3*trise+tp)) + 60*exp(-akd_t*(Delta_t+3*trise+2*tp)) + 60*exp(-akd_t*(Delta_t-3*trise-2*tp))
                + 12*exp(-akd_t*(Delta_t+3*trise+3*tp)) + 12*exp(-akd_t*(Delta_t-3*trise-3*tp)) - 12*exp(-akd_t*(Delta_t+4*trise+3*tp)) - 12*exp(-akd_t*(Delta_t-4*trise-3*tp))
                - 30*exp(-akd_t*(Delta_t+5*trise+2*tp)) - 30*exp(-akd_t*(Delta_t-5*trise-2*tp)) + 24*exp(-akd_t*(Delta_t+4*trise+4*tp)) + 24*exp(-akd_t*(Delta_t-4*trise-4*tp))
                - 12*exp(-akd_t*(Delta_t+5*trise+3*tp)) - 12*exp(-akd_t*(Delta_t-5*trise-3*tp)) + 12*exp(-akd_t*(Delta_t+6*trise+3*tp)) + 12*exp(-akd_t*(Delta_t-6*trise-3*tp))
                - 48*exp(-akd_t*(Delta_t+6*trise+4*tp)) - 48*exp(-akd_t*(Delta_t-6*trise-4*tp)) - 12*exp(-akd_t*(Delta_t+6*trise+5*tp)) - 12*exp(-akd_t*(Delta_t-6*trise-5*tp))
                + 12*exp(-akd_t*(Delta_t+7*trise+5*tp)) + 12*exp(-akd_t*(Delta_t-7*trise-5*tp)) + 24*exp(-akd_t*(Delta_t+8*trise+4*tp)) + 24*exp(-akd_t*(Delta_t-8*trise-4*tp))
                - 18*exp(-akd_t*(Delta_t+7*trise+6*tp)) - 18*exp(-akd_t*(Delta_t-7*trise-6*tp)) + 12*exp(-akd_t*(Delta_t+8*trise+5*tp)) + 12*exp(-akd_t*(Delta_t-8*trise-5*tp))
                - 12*exp(-akd_t*(Delta_t+9*trise+5*tp)) - 12*exp(-akd_t*(Delta_t-9*trise-5*tp)) + 36*exp(-akd_t*(Delta_t+9*trise+6*tp)) + 36*exp(-akd_t*(Delta_t-9*trise-6*tp))
                + 12*exp(-akd_t*(Delta_t+9*trise+7*tp)) + 12*exp(-akd_t*(Delta_t-9*trise-7*tp)) - 12*exp(-akd_t*(Delta_t+10*trise+7*tp)) - 12*exp(-akd_t*(Delta_t-10*trise-7*tp))
                - 18*exp(-akd_t*(Delta_t+11*trise+6*tp)) - 18*exp(-akd_t*(Delta_t-11*trise-6*tp)) + 12*exp(-akd_t*(Delta_t+10*trise+8*tp)) + 12*exp(-akd_t*(Delta_t-10*trise-8*tp))
                - 12*exp(-akd_t*(Delta_t+11*trise+7*tp)) - 12*exp(-akd_t*(Delta_t-11*trise-7*tp)) + 12*exp(-akd_t*(Delta_t+12*trise+7*tp)) + 12*exp(-akd_t*(Delta_t-12*trise-7*tp))
                - 24*exp(-akd_t*(Delta_t+12*trise+8*tp)) - 24*exp(-akd_t*(Delta_t-12*trise-8*tp)) - 12*exp(-akd_t*(Delta_t+12*trise+9*tp)) - 12*exp(-akd_t*(Delta_t-12*trise-9*tp))
                + 12*exp(-akd_t*(Delta_t+13*trise+9*tp)) + 12*exp(-akd_t*(Delta_t-13*trise-9*tp)) + 12*exp(-akd_t*(Delta_t+14*trise+8*tp)) + 12*exp(-akd_t*(Delta_t-14*trise-8*tp))
                - 6*exp(-akd_t*(Delta_t+13*trise+10*tp)) - 6*exp(-akd_t*(Delta_t-13*trise-10*tp)) + 12*exp(-akd_t*(Delta_t+14*trise+9*tp)) + 12*exp(-akd_t*(Delta_t-14*trise-9*tp))
                - 12*exp(-akd_t*(Delta_t+15*trise+9*tp)) - 12*exp(-akd_t*(Delta_t-15*trise-9*tp)) + 12*exp(-akd_t*(Delta_t+15*trise+10*tp)) + 12*exp(-akd_t*(Delta_t-15*trise-10*tp))
                + 12*exp(-akd_t*(Delta_t+15*trise+11*tp)) + 12*exp(-akd_t*(Delta_t-15*trise-11*tp)) - 12*exp(-akd_t*(Delta_t+16*trise+11*tp)) - 12*exp(-akd_t*(Delta_t-16*trise-11*tp))
                - 6*exp(-akd_t*(Delta_t+17*trise+10*tp)) - 6*exp(-akd_t*(Delta_t-17*trise-10*tp)) - 12*exp(-akd_t*(Delta_t+17*trise+11*tp)) - 12*exp(-akd_t*(Delta_t-17*trise-11*tp))
                - 6*exp(-akd_t*(Delta_t+17*trise+12*tp)) - 6*exp(-akd_t*(Delta_t-17*trise-12*tp)) + 12*exp(-akd_t*(Delta_t+18*trise+11*tp)) + 12*exp(-akd_t*(Delta_t-18*trise-11*tp))
                + 12*exp(-akd_t*(Delta_t+18*trise+12*tp)) + 12*exp(-akd_t*(Delta_t-18*trise-12*tp)) - 6*exp(-akd_t*(Delta_t+19*trise+12*tp)) - 6*exp(-akd_t*(Delta_t-19*trise-12*tp)) + 96
            ) / (3*akd_t**4*trise**2)
        )
        Etmp[idx, :, :] = Etmp_tcos3

    if np.any(ind_tsin1):
        trise = np.tile(np.asarray(pulse.trise, dtype=float).reshape(nacq, 1, 1), (1, nr, nk))
        tp = np.tile(np.asarray(pulse.tp, dtype=float).reshape(nacq, 1, 1), (1, nr, nk))
        Etmp_tsin1 = (
            0.25 * (
                24*np.exp(-akd*tp) - 24*np.exp(-akd*trise) - 12*np.exp(-2*akd*trise) - 36*np.exp(-Delta*akd) - 48*akd*trise
                + 12*np.exp(-akd*(Delta-trise)) + 12*np.exp(-akd*(Delta+trise)) + 6*np.exp(-akd*(Delta+2*trise)) + 6*np.exp(-akd*(Delta-2*trise))
                - 12*np.exp(-akd*(Delta-tp)) - 12*np.exp(-akd*(Delta+tp)) - 24*np.exp(-akd*(trise+tp)) - 24*np.exp(-akd*(2*trise+tp)) - 12*np.exp(-akd*(2*trise+2*tp))
                + 24*np.exp(-akd*(3*trise+tp)) + 24*np.exp(-akd*(3*trise+2*tp)) - 12*np.exp(-akd*(4*trise+2*tp)) + 16*akd**3*trise**3 + 24*akd**3*trise**2*tp
                + 12*np.exp(-akd*(Delta-trise-tp)) + 12*np.exp(-akd*(Delta+trise+tp)) + 12*np.exp(-akd*(Delta-2*trise-tp)) + 12*np.exp(-akd*(Delta+2*trise+tp))
                - 12*np.exp(-akd*(Delta-3*trise-tp)) + 6*np.exp(-akd*(Delta+2*trise+2*tp)) + 6*np.exp(-akd*(Delta-2*trise-2*tp)) - 12*np.exp(-akd*(Delta+3*trise+tp))
                - 12*np.exp(-akd*(Delta+3*trise+2*tp)) - 12*np.exp(-akd*(Delta-3*trise-2*tp)) + 6*np.exp(-akd*(Delta+4*trise+2*tp)) + 6*np.exp(-akd*(Delta-4*trise-2*tp)) + 36
            ) / (3*akd**4*trise**2)
        )
        Etmp[ind_tsin1, :, :] = Etmp_tsin1[ind_tsin1, :, :]

    if np.any(ind_tsin2):
        trise = np.tile(np.asarray(pulse.trise, dtype=float).reshape(nacq, 1, 1), (1, nr, nk))
        tp = np.tile(np.asarray(pulse.tp, dtype=float).reshape(nacq, 1, 1), (1, nr, nk))
        Etmp_tsin2 = (
            0.25 * (
                48*np.exp(-akd*tp) - 24*np.exp(-akd*trise) - 36*np.exp(-2*akd*trise) - 60*np.exp(-Delta*akd) - 96*akd*trise
                + 12*np.exp(-akd*(Delta-trise)) + 12*np.exp(-akd*(Delta+trise)) + 18*np.exp(-akd*(Delta+2*trise)) + 18*np.exp(-akd*(Delta-2*trise))
                - 24*np.exp(-akd*(Delta-tp)) - 24*np.exp(-akd*(Delta+tp)) - 24*np.exp(-akd*(trise+tp)) - 72*np.exp(-akd*(2*trise+tp)) - 36*np.exp(-akd*(2*trise+2*tp))
                + 24*np.exp(-akd*(3*trise+tp)) + 24*np.exp(-akd*(3*trise+2*tp)) + 24*np.exp(-akd*(4*trise+tp)) + 48*np.exp(-akd*(4*trise+2*tp)) + 24*np.exp(-akd*(4*trise+3*tp))
                - 24*np.exp(-akd*(5*trise+2*tp)) - 24*np.exp(-akd*(5*trise+3*tp)) - 12*np.exp(-akd*(6*trise+2*tp)) - 24*np.exp(-akd*(6*trise+3*tp))
                - 12*np.exp(-akd*(6*trise+4*tp)) + 24*np.exp(-akd*(7*trise+3*tp)) + 24*np.exp(-akd*(7*trise+4*tp)) - 12*np.exp(-akd*(8*trise+4*tp))
                + 32*akd**3*trise**3 + 48*akd**3*trise**2*tp + 12*np.exp(-akd*(Delta-trise-tp)) + 12*np.exp(-akd*(Delta+trise+tp)) + 36*np.exp(-akd*(Delta-2*trise-tp))
                + 36*np.exp(-akd*(Delta+2*trise+tp)) - 12*np.exp(-akd*(Delta-3*trise-tp)) + 18*np.exp(-akd*(Delta+2*trise+2*tp)) + 18*np.exp(-akd*(Delta-2*trise-2*tp))
                - 12*np.exp(-akd*(Delta+3*trise+tp)) - 12*np.exp(-akd*(Delta-4*trise-tp)) - 12*np.exp(-akd*(Delta+3*trise+2*tp)) - 12*np.exp(-akd*(Delta-3*trise-2*tp))
                - 12*np.exp(-akd*(Delta+4*trise+tp)) - 24*np.exp(-akd*(Delta+4*trise+2*tp)) - 24*np.exp(-akd*(Delta-4*trise-2*tp)) - 12*np.exp(-akd*(Delta+4*trise+3*tp))
                - 12*np.exp(-akd*(Delta-4*trise-3*tp)) + 12*np.exp(-akd*(Delta+5*trise+2*tp)) + 12*np.exp(-akd*(Delta-5*trise-2*tp)) + 12*np.exp(-akd*(Delta+5*trise+3*tp))
                + 12*np.exp(-akd*(Delta-5*trise-3*tp)) + 6*np.exp(-akd*(Delta+6*trise+2*tp)) + 6*np.exp(-akd*(Delta-6*trise-2*tp)) + 12*np.exp(-akd*(Delta+6*trise+3*tp))
                + 12*np.exp(-akd*(Delta-6*trise-3*tp)) + 6*np.exp(-akd*(Delta+6*trise+4*tp)) + 6*np.exp(-akd*(Delta-6*trise-4*tp)) - 12*np.exp(-akd*(Delta+7*trise+3*tp))
                - 12*np.exp(-akd*(Delta-7*trise-3*tp)) - 12*np.exp(-akd*(Delta+7*trise+4*tp)) - 12*np.exp(-akd*(Delta-7*trise-4*tp)) + 6*np.exp(-akd*(Delta+8*trise+4*tp))
                + 6*np.exp(-akd*(Delta-8*trise-4*tp)) + 60
            ) / (3*akd**4*trise**2)
        )
        Etmp[ind_tsin2, :, :] = Etmp_tsin2[ind_tsin2, :, :]

    G2 = np.tile(np.asarray(pulse.G, dtype=float).reshape(nacq, 1), (1, nr))
    E = _safe_exp(-2.0 * (pulse.gamma * G2) ** 2 * np.sum(bk3 * Etmp, axis=2))
    if inverse is not None:
        E = E[:, inverse]
    return E


def r1rho_dispersion(parms: Any, model: SignalModel) -> np.ndarray:
    p = coerce_parameter_matrix(parms, 3)
    R2 = p[0].reshape(1, -1)
    g2D = p[1].reshape(1, -1)
    q2D = p[2].reshape(1, -1)
    gamma = model.pulse.gamma / (2.0 * np.pi) * 10.0
    FSL = np.asarray(model.pulse.FSL, dtype=float).reshape(-1, 1)
    return R2 + gamma**2 * g2D / (q2D**2 + FSL**2)


def sirqmt_signal(parms: Any, model: SignalModel) -> np.ndarray:
    p = coerce_parameter_matrix(parms, 7)
    R1f, pmf, kmf, Sf, Mfinf, R1m, Sm = [row.reshape(1, -1) for row in p]
    ti = np.asarray(model.pulse.TI, dtype=float).reshape(-1, 1)
    td = np.asarray(model.pulse.TD, dtype=float).reshape(-1, 1)
    R1diff = np.sqrt((R1f - R1m + (pmf - 1.0) * kmf) ** 2 + 4.0 * pmf * kmf * kmf)
    R1plus = 0.5 * (R1f + R1m + (1.0 + pmf) * kmf + R1diff)
    R1minus = R1plus - R1diff
    bftdplus = -(R1f - R1minus) / R1diff
    bftdminus = (R1f - R1plus) / R1diff
    bmtdplus = -(R1m - R1minus) / R1diff
    bmtdminus = (R1m - R1plus) / R1diff
    Mftd = bftdplus * np.exp(-td * R1plus) + bftdminus * np.exp(-td * R1minus) + 1.0
    Mmtd = bmtdplus * np.exp(-td * R1plus) + bmtdminus * np.exp(-td * R1minus) + 1.0
    bfplus = ((Sf * Mftd - 1.0) * (R1f - R1minus) + (Sf * Mftd - Sm * Mmtd) * pmf * kmf) / R1diff
    bfminus = -((Sf * Mftd - 1.0) * (R1f - R1plus) + (Sf * Mftd - Sm * Mmtd) * pmf * kmf) / R1diff
    signal = (bfplus * np.exp(-ti * R1plus) + bfminus * np.exp(-ti * R1minus) + 1.0) * Mfinf
    if str(model.structure.get("signalType", "abs")).lower() == "abs":
        signal = np.abs(signal)
    return signal


def sirqmt_t1obs(parms: Any, _: SignalModel) -> np.ndarray:
    p = coerce_parameter_matrix(parms, 7)
    R1f, pmf, kmf, _Sf, _Mfinf, R1m, _Sm = [row for row in p]
    R1minus = 0.5 * (R1f + R1m + (1.0 + pmf) * kmf - np.sqrt((R1f - R1m + (pmf - 1.0) * kmf) ** 2 + 4.0 * pmf * kmf * kmf))
    return 1.0 / R1minus


def general_solution_bloch_eq(Ain: Any, Aex: Any, kin: Any, kex: Any, Min0: Any, Mex0: Any, t: Any) -> tuple[np.ndarray, np.ndarray]:
    Ain = np.asarray(Ain, dtype=float)
    Aex = np.asarray(Aex, dtype=float)
    kin = np.asarray(kin, dtype=float)
    kex = np.asarray(kex, dtype=float)
    Min0 = np.asarray(Min0, dtype=float)
    Mex0 = np.asarray(Mex0, dtype=float)
    t = np.asarray(t, dtype=float)
    Q = np.sqrt((Ain - Aex) ** 2 + 4.0 * kin * kex)
    Q_safe = np.where(Q == 0.0, np.finfo(float).eps, Q)
    R1 = (Ain + Aex - Q) / 2.0
    R2 = (Ain + Aex + Q) / 2.0
    exp_r1 = _safe_exp(-R1 * t)
    exp_r2 = _safe_exp(-R2 * t)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        M_in = (((Aex - Ain + Q) * Min0 + 2.0 * kex * Mex0) / (2.0 * Q_safe)) * exp_r1 + (((-Aex + Ain + Q) * Min0 - 2.0 * kex * Mex0) / (2.0 * Q_safe)) * exp_r2
        M_ex = (((-Aex + Ain + Q) * Mex0 + 2.0 * kin * Min0) / (2.0 * Q_safe)) * exp_r1 + (((Aex - Ain + Q) * Mex0 - 2.0 * kin * Min0) / (2.0 * Q_safe)) * exp_r2
    no_exchange = (kin == 0.0) | (kex == 0.0)
    if np.any(no_exchange):
        M_in = np.where(no_exchange, Min0 * _safe_exp(-(Ain - kin) * t), M_in)
        M_ex = np.where(no_exchange, Mex0 * _safe_exp(-(Aex - kex) * t), M_ex)
    M_in = np.where(np.isfinite(M_in), M_in, 0.0)
    M_ex = np.where(np.isfinite(M_ex), M_ex, 0.0)
    return M_in, M_ex


def smt_powder_average_signal(parms: Any, model: SignalModel) -> np.ndarray:
    p = coerce_parameter_matrix(parms, 2)
    vax = p[0].reshape(1, -1)
    Dax = p[1].reshape(1, -1)
    b = np.asarray(model.pulse.b, dtype=float).reshape(-1, 1)
    x_ax = np.sqrt(np.clip(b * Dax, 0.0, None))
    signal_ax = np.ones_like(x_ax)
    mask_ax = x_ax > 0.0
    signal_ax[mask_ax] = np.sqrt(np.pi) * erf(x_ax[mask_ax]) / (2.0 * x_ax[mask_ax])
    Dex = (1.0 - vax) * Dax
    x_ex = np.sqrt(np.clip(b * (Dax - Dex), 0.0, None))
    ex_ratio = np.ones_like(x_ex)
    mask_ex = x_ex > 0.0
    ex_ratio[mask_ex] = np.sqrt(np.pi) * erf(x_ex[mask_ex]) / (2.0 * x_ex[mask_ex])
    signal_ex = np.exp(-b * Dex) * ex_ratio
    out = vax * signal_ax + (1.0 - vax) * signal_ex
    return np.where(b < np.finfo(float).eps, 1.0, out)
