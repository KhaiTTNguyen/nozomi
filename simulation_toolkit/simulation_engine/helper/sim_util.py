import numpy as np
from simulation_toolkit.simulation_engine.waveforms import (
    PGDiffWaveform,
    ApodizedCosineOGSEWaveform,
    gamma,
)

# -----------------------------------------------------------------------
# Analytical b-value formulas
# -----------------------------------------------------------------------
# Reference units throughout:
#   gamma  : rad/ms/mT   (= 267.513 rad/ms/mT for protons)
#   gmax   : mT/m        (1 mT/m = 1e-6 mT/µm)
#   timing : ms
#   b      : ms/µm²      (1 ms/µm² = 1000 s/mm²)
# -----------------------------------------------------------------------

def bvalue_pgse_ms_um2(big_delta_ms: float, little_delta_ms: float,
                       gmax_mT_per_m: float = 1.0) -> float:
    """
    Analytical b-value for a PGSE waveform (Stejskal-Tanner equation):

        b = (gamma G δ)² (Δ - δ/3)

    Parameters
    ----------
    big_delta_ms    : Δ – separation between gradient lobe centres (ms)
    little_delta_ms : δ – gradient lobe duration (ms)
    gmax_mT_per_m   : peak gradient amplitude (mT/m); default 1.0

    Returns
    -------
    float  b-value in ms/µm²
    """
    G_mT_um = gmax_mT_per_m * 1e-6      # mT/m → mT/µm
    return float((gamma * G_mT_um * little_delta_ms) ** 2
                 * (big_delta_ms - little_delta_ms / 3.0))


def bvalue_ogse_apodized_cosine_ms_um2(N_cycles: int, T_duration_ms: float,
                                       gmax_mT_per_m: float = 1.0) -> float:
    """
    Analytical b-value for an apodized-cosine OGSE waveform (Does et al. 2003):

        b = (1/4) (gamma G / (π N))² T³ (1 - 1/(8N))

    Parameters
    ----------
    N_cycles      : number of cosine cycles
    T_duration_ms : active gradient window duration T (ms)
    gmax_mT_per_m : peak gradient amplitude (mT/m); default 1.0

    Returns
    -------
    float  b-value in ms/µm²
    """
    G_mT_um = gmax_mT_per_m * 1e-6      # mT/m → mT/µm
    N = float(N_cycles)
    T = float(T_duration_ms)
    return float(0.25 * (gamma * G_mT_um / (np.pi * N)) ** 2
                 * T ** 3 * (1.0 - 1.0 / (8.0 * N)))


def gmax_from_bvalue_pgse(target_ms_um2: float,
                          big_delta_ms: float,
                          little_delta_ms: float) -> float:
    """Return gmax (mT/m) that achieves target_ms_um2 for a PGSE waveform."""
    G_mT_um = (np.sqrt(target_ms_um2)
               / (gamma * little_delta_ms
                  * np.sqrt(big_delta_ms - little_delta_ms / 3.0)))
    return float(G_mT_um / 1e-6)  # mT/µm → mT/m


def gmax_from_bvalue_ogse_apodized_cosine(target_ms_um2: float,
                                          N_cycles: int,
                                          T_duration_ms: float) -> float:
    """Return gmax (mT/m) that achieves target_ms_um2 for an apodized-cosine OGSE waveform."""
    N = float(N_cycles)
    T = float(T_duration_ms)
    G_mT_um = (np.sqrt(4.0 * target_ms_um2) * np.pi * N
               / (gamma * T ** 1.5 * np.sqrt(1.0 - 1.0 / (8.0 * N))))
    return float(G_mT_um / 1e-6)  # mT/µm → mT/m


def _s_mm2_to_ms_um2(b_s_mm2: float) -> float:
    """Convert b-value: s/mm² → ms/µm²  (exact: 1 s/mm² = 1×10⁻³ ms/µm²)."""
    return float(b_s_mm2) * 1e-3



def _safe_interp_on_lag(lag_ms, diffusion_time_ms, diffusion_coeff):
    t = np.asarray(diffusion_time_ms, dtype=float)
    d = np.asarray(diffusion_coeff, dtype=float)

    if t.ndim != 1 or d.ndim != 1 or t.size != d.size or t.size == 0:
        raise ValueError("diffusion_time_ms and diffusion_coeff must be 1D arrays with equal non-zero length")

    order = np.argsort(t)
    t = t[order]
    d = d[order]

    if t[0] > 0:
        t = np.insert(t, 0, 0.0)
        d = np.insert(d, 0, d[0])

    return np.interp(lag_ms, t, d, left=d[0], right=d[-1])

def wide_pulse_gradient_integration(diffusion_time_ms, diffusion_coeff, gradient_waveform, gradient_dt_ms):
    """
    Compute apparent diffusion coefficient using GPA wide-pulse integration.

    Implements
    D_app = -gamma^2 / b * ∫∫ G(t1)G(t2)(t1-t2)D(t1-t2) dt2 dt1,
    but evaluated in an equivalent autocorrelation form where constants cancel
    in the ratio:

    D_app = [∫ Δ C_G(Δ) D(Δ) dΔ] / [∫ Δ C_G(Δ) dΔ]

    Parameters
    ----------
    diffusion_time_ms : array-like
        Time axis for simulated diffusion coefficient D(t), in ms.
    diffusion_coeff : array-like
        Simulated D(t), e.g. RD(t), in um^2/ms.
    gradient_waveform : array-like
        Gradient samples G(t) at uniform spacing.
    gradient_dt_ms : float
        Time step for gradient waveform in ms.

    Returns
    -------
    float
        Apparent diffusion coefficient sampled by the waveform, in um^2/ms.
    """
    g = np.asarray(gradient_waveform, dtype=float).reshape(-1)
    if g.size < 3:
        raise ValueError("gradient_waveform must have at least 3 samples")
    if gradient_dt_ms <= 0:
        raise ValueError("gradient_dt_ms must be positive")

    n = g.size
    nfft = 1 << (2 * n - 1).bit_length()
    g_fft = np.fft.rfft(g, n=nfft)
    autocorr = np.fft.irfft(g_fft * np.conjugate(g_fft), n=nfft)[:n]

    corr = autocorr * gradient_dt_ms
    lag_ms = np.arange(n, dtype=float) * gradient_dt_ms
    d_lag = _safe_interp_on_lag(lag_ms, diffusion_time_ms, diffusion_coeff)

    weights = lag_ms * corr
    denom = np.trapz(weights, lag_ms)
    if np.isclose(denom, 0.0):
        raise RuntimeError("Gradient weighting denominator is zero; check waveform and timing")

    numer = np.trapz(weights * d_lag, lag_ms)
    return float(numer / denom)


def build_gradient_waveform(gradient_config, total_sim_time_ms, time_step_ms,
                            target_bvalue_s_mm2=None):
    """
    Build a waveform object from gradient configuration.

    Supported config:
      - PGSE: {type: PGSE, big_delta, little_delta}
      - OGSE: {type: OGSE, num_oscil / N_cycles, duration / T_duration}

    B-value / gmax:
    ---------------
    If `target_bvalue_s_mm2` is given (or in gradient_config as
    ``b_value_s_mm2``), the required gradient amplitude (gmax, mT/m) is
    computed using the closed-form analytical formulas:

      PGSE     : b = (gamma G δ)² (Δ - δ/3)
      OGSE AC  : b = (1/4)(gamma G / πN)² T³ (1 - 1/(8N))

    Note: ``wide_pulse_gradient_integration`` uses only the *shape* of the
    normalised waveform, so D_app is independent of gmax. The gmax / b-value
    are stored for reporting and physical-unit verification.

    Returned waveform carries:
      .bvalue_gmax1_ms_um2  - b-value (ms/µm²) at gmax = 1 mT/m
      .bvalue_ms_um2        - b-value (ms/µm²) at the stored gmax
      .bvalue_s_mm2         - b-value (s/mm²) at the stored gmax
      .gmax_mT_per_m        - gradient amplitude (mT/m) for target b
                              (= 1.0 when no target is set)

    Parameters
    ----------
    gradient_config      : dict  - waveform parameters
    total_sim_time_ms    : float - echo time / simulation window (ms)
    time_step_ms         : float - waveform sample period (ms)
    target_bvalue_s_mm2  : float or None
        Target b-value in s/mm².  Overrides gradient_config["b_value_s_mm2"].
    """
    if not isinstance(gradient_config, dict):
        raise ValueError("params['gradient'] must be a dictionary")

    gtype = str(gradient_config.get("type", "PGSE")).upper()

    # Resolve target b-value: keyword arg takes priority over config dict
    _target_s_mm2 = target_bvalue_s_mm2
    if _target_s_mm2 is None:
        _target_s_mm2 = gradient_config.get("b_value_s_mm2", None)

    if gtype == "PGSE":
        big_delta    = float(gradient_config["big_delta"])
        little_delta = float(gradient_config["little_delta"])
        waveform = PGDiffWaveform(
            big_delta=big_delta,
            little_delta=little_delta,
            te=float(total_sim_time_ms),
            time_step=float(time_step_ms),
        )
        # Analytical b-value at gmax = 1 mT/m
        b_gmax1 = bvalue_pgse_ms_um2(big_delta, little_delta, gmax_mT_per_m=1.0)
        # Analytical inverse: gmax from target b
        _gmax_from_target = lambda b_ms_um2: gmax_from_bvalue_pgse(
            b_ms_um2, big_delta, little_delta)

    elif gtype == "OGSE":
        n_cycles   = int(gradient_config.get("N_cycles",
                         gradient_config.get("num_oscil", 1)))
        t_duration = float(gradient_config.get("T_duration",
                           gradient_config.get("duration",
                           total_sim_time_ms / 2.0)))
        waveform = ApodizedCosineOGSEWaveform(
            N_cycles=n_cycles,
            T_duration=t_duration,
            te=float(total_sim_time_ms),
            gmax=float(gradient_config.get("gmax", 1.0)),
            time_step=float(time_step_ms),
        )
        # Analytical b-value at gmax = 1 mT/m
        b_gmax1 = bvalue_ogse_apodized_cosine_ms_um2(n_cycles, t_duration,
                                                      gmax_mT_per_m=1.0)
        # Analytical inverse: gmax from target b
        _gmax_from_target = lambda b_ms_um2: gmax_from_bvalue_ogse_apodized_cosine(
            b_ms_um2, n_cycles, t_duration)

    else:
        raise ValueError(f"Unsupported gradient type: {gtype}")

    # ------------------------------------------------------------------
    # Attach b-value metadata to the waveform object
    # ------------------------------------------------------------------
    waveform.bvalue_gmax1_ms_um2 = b_gmax1

    if _target_s_mm2 is not None:
        target_ms_um2 = _s_mm2_to_ms_um2(_target_s_mm2)
        gmax = _gmax_from_target(target_ms_um2)
        waveform.gmax_mT_per_m = float(gmax)
        waveform.bvalue_ms_um2 = float(target_ms_um2)
        waveform.bvalue_s_mm2  = float(_target_s_mm2)
    else:
        waveform.gmax_mT_per_m = 1.0
        waveform.bvalue_ms_um2 = b_gmax1
        waveform.bvalue_s_mm2  = b_gmax1 / 1e-3   # ms/µm² → s/mm²

    return waveform