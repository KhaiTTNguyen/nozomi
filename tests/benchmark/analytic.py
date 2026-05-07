"""
van Gelderen 1994 analytic perpendicular PGSE signal for a solid cylinder.

Used as the ground-truth reference for the intra-axonal benchmark experiment
(walkers confined inside impermeable z-oriented cylinders, x-gradient
perpendicular to fibers).

Reference: van Gelderen, P., DesPres, D., van Zijl, P. C. M. and Moonen,
C. T. W. (1994). "Evaluation of Restricted Diffusion in Cylinders.
Phosphocreatine in Rabbit Leg Muscle." JMR B 103:255-260.
"""
from __future__ import annotations

import numpy as np
from scipy.special import jnp_zeros

from tests.benchmark.config import GAMMA_RAD_S_T, gmax_T_per_m_for_bvalue_s_mm2


def vangelderen_perp(G_T_per_m: float, R_m: float, D_m2_s: float,
                     delta_s: float, Delta_s: float,
                     n_terms: int = 20) -> float:
    """
    Normalised perpendicular PGSE signal S/S0 for a walker confined in a
    solid impermeable cylinder of radius ``R_m`` with free diffusivity
    ``D_m2_s``, gradient magnitude ``G_T_per_m`` perpendicular to the axis,
    rectangular Stejskal-Tanner pulses of width ``delta_s`` separated by
    ``Delta_s``.
    """
    if G_T_per_m <= 0.0:
        return 1.0
    # Roots of J1'(x) = 0. These are alpha_m * R in the paper.
    alphas = jnp_zeros(1, n_terms) / R_m  # shape (n_terms,)
    gamma2 = GAMMA_RAD_S_T * GAMMA_RAD_S_T
    G2 = G_T_per_m * G_T_per_m
    acc = 0.0
    for a in alphas:
        a2 = a * a
        a4 = a2 * a2
        a6 = a4 * a2
        aR2 = a2 * R_m * R_m
        exp_d = np.exp(-a2 * D_m2_s * delta_s)
        exp_D = np.exp(-a2 * D_m2_s * Delta_s)
        exp_Dmd = np.exp(-a2 * D_m2_s * (Delta_s - delta_s))
        exp_Dpd = np.exp(-a2 * D_m2_s * (Delta_s + delta_s))
        bracket = (
            2.0 * a2 * D_m2_s * delta_s
            - 2.0
            + 2.0 * exp_d
            + 2.0 * exp_D
            - exp_Dmd
            - exp_Dpd
        )
        # Coefficient: 1 / ( alpha^6 D^2 (alpha^2 R^2 - 1) )
        acc += bracket / (a6 * D_m2_s * D_m2_s * (aR2 - 1.0))
    logS = -2.0 * gamma2 * G2 * acc
    return float(np.exp(logS))


def analytic_signals(cfg) -> list:
    """Per-b-value van Gelderen signal for the benchmark substrate."""
    out = []
    for b in cfg.bvals_s_mm2:
        G = gmax_T_per_m_for_bvalue_s_mm2(b, cfg.delta_s, cfg.Delta_s)
        out.append(vangelderen_perp(
            G, cfg.cylinder_radius_m, cfg.D0_m2_s, cfg.delta_s, cfg.Delta_s,
        ))
    return out
