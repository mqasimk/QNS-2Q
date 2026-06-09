"""
dc_ramsey_prototype.py
======================

STANDALONE proof-of-concept (not wired into the pipeline): infer the DC /
zero-frequency self-spectra S_aa(0) from a free-induction-decay (Ramsey) SLOPE,
instead of the heavy frequency-comb DC reconstruction in
``spectral_inversion.recon_S_11_dc`` / ``recon_S_22_dc`` / ``recon_S_1212_dc``.

Why this exists
---------------
The comb has no tooth at omega = 0 and cannot resolve the narrow DC features
(their half-widths, 80-160 kHz, are below the comb spacing 1/T = 250 kHz), so the
current self-spectrum DC estimator backs S_aa(0) out of FID data by *subtracting*
the comb-reconstructed S_bb / S_1212 harmonic contributions -- inheriting their
full error budget.

Physics of the replacement
---------------------------
Under pure dephasing the FID decay exponent is

    chi_a(t) = (1/2) \\int dw/2pi  S_aa(w) |F_FID(w,t)|^2 ,
    |F_FID(w,t)|^2 = (2 - 2 cos w t) / w^2 ,

which in the motional-narrowing limit (t >> tau_c) grows LINEARLY:

    chi_a(t)  ->  (S_aa(0)/2) * t          =>     S_aa(0) = 2 * d chi_a/dt .

The codebase already encodes this identity as T2 = 2/S(0) in
``cz_optimize._calculate_T2``. So the dedicated DC experiment is just a Ramsey
decay whose late-time slope is read off -- no comb, no cross-subtraction, no
20x20 inversion.

The Ising subtlety (handled below)
----------------------------------
On the real 2-qubit device qubit 1's coherence feels  zeta_1 + Z2 * zeta_12 , so a
naive FID/FID slope returns S_11(0) + S_1212(0). Decoupling the PARTNER's Ising
term (a fast CPMG on qubit 2) pushes zeta_12 sensitivity to high frequency and
restores a clean S_11(0). This script demonstrates:

    Experiment A  (zeta_12 switched off in the model): 2*slope == S_aa(0) exactly.
    Experiment B  (full model): naive FID/FID picks up S_1212(0); a fast CPMG on
                  the partner suppresses it back toward S_11(0).

It drives the REAL simulator (trajectories.solver_prop + observables.make_c_a_0_mt).

Run from src/ :   python dc_ramsey_prototype.py
"""

import os
import time

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qns2q.characterize.experiments import QNSExperimentConfig
from qns2q.model.observables import make_c_a_0_mt
from qns2q.model.trajectories import make_noise_mat_arr, solver_prop
from qns2q.noise.spectra import S_11, S_22, S_1212
from qns2q.paths import run_folder, project_root

# --- knobs (kept light: this is a demonstrator, not a production run) ----------
T_GRAIN  = 300            # time points per base period T (dt = T/T_GRAIN ~ 13 ns << 1/wmax)
W_GRAIN  = 1500           # noise frequency grain (dw ~ 3.3 kHz: resolves the DC features)
N_SHOTS  = 4000           # noise realizations per point
M_VALUES = [1, 2, 4, 6, 8, 10]    # total free-evolution time t = M*T (4..40 us)
FIT_FROM = 16e-6          # fit the slope on the linear (t >= FIT_FROM) tail
CPMG_DIV = 6             # partner CPMG cycle = T/CPMG_DIV  (Exp B isolation knob)
SEED     = 20260608


def S_zero(w):
    """A null spectrum, used to switch off the Ising noise in Experiment A."""
    return jnp.zeros_like(w)


def chi_theory_fid(S, t, wmax_int=2 * np.pi * 200e6, n=400001):
    """Analytic FID decay exponent chi(t) = (1/2)\\int dw/2pi S(w)|F_FID|^2 (one qubit)."""
    w = np.linspace(0.0, wmax_int, n)
    F2 = np.empty_like(w)
    F2[0] = t ** 2
    F2[1:] = (2 - 2 * np.cos(w[1:] * t)) / w[1:] ** 2
    return (1.0 / (2 * np.pi)) * 2 * np.trapezoid(S(w) * F2, w) * 0.5


def run_point(m, pulse, l, spec_vec, cpmg_div=None, midpoint=True):
    """One (total-time) point: returns C_{l,0}(MT) and its stderr from the real sim.

    midpoint=True uses the bin-midpoint noise-synthesis grid (no spurious w=0
    static tone); midpoint=False reproduces the legacy endpoint-grid artifact.
    """
    cfg = QNSExperimentConfig(M=m, t_grain=T_GRAIN, n_shots=N_SHOTS)
    noise_mats = jnp.array(make_noise_mat_arr(
        'make', spec_vec=spec_vec, t_vec=cfg.t_vec, w_grain=W_GRAIN, wmax=cfg.wmax,
        truncate=cfg.truncate, gamma=cfg.gamma, gamma_12=cfg.gamma_12, midpoint=midpoint))
    # FID ignores ctime; CPMG partner uses ctime = T/cpmg_div.
    ct = cfg.T if cpmg_div is None else cfg.T / cpmg_div
    means, errs = make_c_a_0_mt(
        solver_prop, pulse, cfg.t_vec, jnp.array([ct]), cfg.CM, cfg.spMit,
        l=l, n_shots=cfg.n_shots, m=cfg.M, t_b=cfg.t_b, a_m=cfg.a_m,
        delta=cfg.delta, noise_mats=noise_mats, a_sp=cfg.a_sp, c=cfg.c)
    return float(means[0]), float(errs[0]), float(cfg.T)


def sweep(label, pulse, l, spec_vec, cpmg_div=None, midpoint=True):
    """Sweep total time t = M*T and return (t_array, C_array, err_array)."""
    print(f"\n[{label}] pulse={pulse} l={l} cpmg_div={cpmg_div} midpoint={midpoint}")
    ts, Cs, Es = [], [], []
    for m in M_VALUES:
        t0 = time.time()
        C, E, T = run_point(m, pulse, l, spec_vec, cpmg_div, midpoint=midpoint)
        t_tot = m * T
        ts.append(t_tot); Cs.append(C); Es.append(E)
        print(f"   M={m:2d}  t={t_tot*1e6:5.1f} us   C_{l},0 = {C:9.5f} +/- {E:.5f}   "
              f"({time.time()-t0:.1f}s)")
    return np.array(ts), np.array(Cs), np.array(Es)


def slope_estimate(ts, Cs):
    """2 * late-time slope of C(t) = estimate of the DC spectral value."""
    mask = ts >= FIT_FROM
    coef = np.polyfit(ts[mask], Cs[mask], 1)
    return 2.0 * coef[0], coef          # (S(0)_est, [slope, intercept])


def main():
    np.random.seed(SEED)
    t_start = time.time()

    S11_0 = float(S_11(jnp.array([0.0]))[0])
    S22_0 = float(S_22(jnp.array([0.0]))[0])
    S1212_0 = float(S_1212(jnp.array([0.0]))[0])
    print("=" * 78)
    print("Analytic DC values (ground truth):")
    print(f"   S_11(0)   = {S11_0:9.1f}")
    print(f"   S_22(0)   = {S22_0:9.1f}")
    print(f"   S_1212(0) = {S1212_0:9.1f}")
    print("=" * 78)

    spec_full = [S_11, S_22, S_1212]
    spec_noIsing = [S_11, S_22, S_zero]

    # --- Experiment A: clean principle (Ising switched off) -> 2*slope == S_aa(0) ---
    # Run A1 on BOTH noise grids: the legacy endpoint grid carries a spurious w=0
    # static tone (DC bias ~ O(dw)); the midpoint grid removes it.
    tA1e, CA1e, EA1e = sweep("A1-endpoint S_11 (legacy w=0 grid)", ['FID', 'FID'], 1,
                             spec_noIsing, midpoint=False)
    tA1, CA1, EA1 = sweep("A1-midpoint S_11 (fixed grid)", ['FID', 'FID'], 1,
                          spec_noIsing, midpoint=True)
    tA2, CA2, EA2 = sweep("A2-midpoint S_22 (fixed grid)", ['FID', 'FID'], 2,
                          spec_noIsing, midpoint=True)
    S11_estA_e, _ = slope_estimate(tA1e, CA1e)
    S11_estA, fitA1 = slope_estimate(tA1, CA1)
    S22_estA, fitA2 = slope_estimate(tA2, CA2)

    # --- Experiment B: full model. naive FID/FID leaks S_1212; partner CPMG cleans it.
    tB0, CB0, EB0 = sweep("B0: full FID/FID (leak)", ['FID', 'FID'], 1, spec_full)
    tB1, CB1, EB1 = sweep("B1: full FID/CPMG (isolated)", ['FID', 'CPMG'], 1, spec_full,
                          cpmg_div=CPMG_DIV)
    S11_estB0, fitB0 = slope_estimate(tB0, CB0)
    S11_estB1, fitB1 = slope_estimate(tB1, CB1)

    print("\n" + "=" * 78)
    print("RESULTS")
    print("-" * 78)
    print("Experiment A (Ising off): S_aa(0) = 2 * d C_{a,0}/dt  should equal truth")
    print(f"   S_11(0) legacy w=0 grid:  est {S11_estA_e:9.1f}   true {S11_0:9.1f}   "
          f"ratio {S11_estA_e/S11_0:5.3f}   <- biased by spurious static tone")
    print(f"   S_11(0) midpoint grid:    est {S11_estA:9.1f}   true {S11_0:9.1f}   "
          f"ratio {S11_estA/S11_0:5.3f}   <- estimator is exact")
    print(f"   S_22(0) midpoint grid:    est {S22_estA:9.1f}   true {S22_0:9.1f}   "
          f"ratio {S22_estA/S22_0:5.3f}")
    print("-" * 78)
    print("Experiment B (full model), estimating S_11(0):")
    print(f"   naive FID/FID:        2*slope = {S11_estB0:9.1f}   "
          f"(expect S_11(0)+S_1212(0) = {S11_0+S1212_0:9.1f}, ratio {S11_estB0/(S11_0+S1212_0):5.3f})")
    print(f"   FID + partner CPMG:   2*slope = {S11_estB1:9.1f}   "
          f"(target S_11(0) = {S11_0:9.1f}, ratio {S11_estB1/S11_0:5.3f})")
    print("=" * 78)

    # --- figure ---
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axs[0]
    tt = np.linspace(0, max(tA1), 200)
    ax.plot(tt * 1e6, [chi_theory_fid(S_11, t) for t in tt], '-', color='black', lw=1,
            label=r'theory $\chi_{11}(t)$ (slope $S_{11}(0)/2$)')
    ax.errorbar(tA1e * 1e6, CA1e, yerr=EA1e, fmt='x', color='#999999',
                label=f'legacy $w{{=}}0$ grid (ratio {S11_estA_e/S11_0:.2f})')
    ax.errorbar(tA1 * 1e6, CA1, yerr=EA1, fmt='o', color='#0072B2',
                label=f'midpoint grid (ratio {S11_estA/S11_0:.2f})')
    ax.axvline(FIT_FROM * 1e6, color='grey', ls=':', lw=0.8)
    ax.set_xlabel(r'free-evolution time $t = MT$ ($\mu$s)')
    ax.set_ylabel(r'$C_{1,0}(t)$  (FID decay exponent $\chi_1$)')
    ax.set_title('(a) $2\\,d\\chi/dt$ recovers $S_{11}(0)$ exactly;\n'
                 'legacy $w{=}0$ noise grid biases it high')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    ax = axs[1]
    ax.errorbar(tB0 * 1e6, CB0, yerr=EB0, fmt='o', color='#999999',
                label=r'FID/FID (leaks $S_{1212}$)')
    ax.errorbar(tB1 * 1e6, CB1, yerr=EB1, fmt='^', color='#009E73',
                label=r'FID/CPMG (partner decoupled)')
    # reference slopes
    ax.plot(tB1 * 1e6, 0.5 * S11_0 * tB1, '--', color='#009E73', lw=1,
            label=r'slope $=S_{11}(0)/2$')
    ax.plot(tB0 * 1e6, 0.5 * (S11_0 + S1212_0) * tB0, ':', color='#999999', lw=1,
            label=r'slope $=(S_{11}{+}S_{1212})(0)/2$')
    ax.axvline(FIT_FROM * 1e6, color='grey', ls=':', lw=0.8)
    ax.set_xlabel(r'free-evolution time $t = MT$ ($\mu$s)')
    ax.set_ylabel(r'$C_{1,0}(t)$')
    ax.set_title('(b) full model: partner CPMG isolates $S_{11}(0)$\n'
                 f'naive ratio {S11_estB0/(S11_0+S1212_0):.3f} vs leak target; '
                 f'isolated ratio {S11_estB1/S11_0:.3f}')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(project_root(), run_folder(), "figures", "reconstruction")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "dc_ramsey_prototype.pdf")
    plt.savefig(out, bbox_inches='tight')
    print(f"\nSaved figure to {out}")
    print(f"Total wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
