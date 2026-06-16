"""Solve the amplitude constants for the experimentally-anchored noise model.

Self-contained (does NOT import qns2q.noise.spectra — it derives the constants
that get baked INTO that module). See NOISE_MODEL_SPEC.md for provenance.

Construction (tau units, two-sided spectra, synthesis convention
<b(t)b(t')> = (1/pi) * int_0^{2wmax} S(w) cos(w(t-t')) dw):

  S_el_A(w) = A_EL1 * (w^2 + W_IR^2)^(-0.35)        # gamma = 0.7  [npj 2025, qubit R]
  S_el_B(w) = A_EL2 * (w^2 + W_IR^2)^(-0.20)        # gamma = 0.4  [npj 2025, qubit L]
  S_nuc_l(w) = A_NUCl * (w^2 + W_IR^2)^(-0.60)      # gamma = 1.2  [sub-Hz hyperfine slope]
               + class-F lines (qubit-local)
  zeta_1 = e_A + n_1 ; zeta_2 = e_B + n_2 ; zeta_12 = A_J*e_A - B_J*e_B
  cross(e_A, e_B) = C2_SHARE * sqrt(S_el_A S_el_B) * exp(-i w DT_SHIFT)

Targets:
  T2* (chi(T2*) = 1 under FID) = 800 tau per qubit          [purified 28Si,
      Yoneda 2018 (20 us at the 25 ns anchor); retargeted from 260 tau after
      the 2026-06-10 acceptance-gate run -- see NOISE_MODEL_SPEC.md sec. 6]
  electrical fraction at W_MID: f1 = 0.88, f2 = 0.80        [qubit 2 extra nuclear]
  S_1212 / sqrt(S_11 S_22) at W_MID = 0.10                  [gate-operating-point J noise]
  in-band coherence pattern (+, +, -)                       [Yoneda 2023]

FID dephasing integral (H = (1/2) b Z convention of make_Hamiltonian):
  chi(t) = (2/pi) * int_0^{2wmax} S(w) sin^2(w t / 2) / w^2 dw
"""
import numpy as np

# --- fixed model constants (spec section 3) -----------------------------------
W_IR = 0.02
G_EL1, G_EL2, G_NUC = 0.7, 0.4, 1.2
C2_SHARE = 0.8
DT_SHIFT = 1.5
BJ_OVER_AJ = 1.05            # b_J > a_J * C2 makes c_{2,12} anti-phase
T2_TARGET = 800.0
F_EL1, F_EL2 = 0.88, 0.80
R_1212 = 0.10                # S_1212 / sqrt(S_11 S_22) at W_MID
W_MID = 0.35

# band / synthesis window (QNSExperimentConfig defaults: T = 160, truncate = 20)
T_COMB = 160.0
TRUNCATE = 20
WMAX = 2 * np.pi * TRUNCATE / T_COMB      # 0.7854
W_SYNTH = 2 * WMAX                        # synthesis band-limit

# class-F lines: GaAs nuclear-difference triplet at B_eff = 600 mT
# (Malinowski 2017; two near-degenerate + one at twice, positions ∝ B)
LINE_CENTERS = np.array([0.261, 0.273, 0.534])
LINE_SIGMA = 0.02
LINE_FACTOR_Q2 = 8.0         # peak height over the smooth TOTAL S_22 at center
LINE_FACTOR_Q1 = 3.2
# (2026-06-10: reduced from 20/8 -- at x20 the comb harmonic ON the line decays
# to coherence ~1e-4, unmeasurable at any shot count; at x8 the at-line
# coefficient is C ~ 1.8, coherence ~3e-2, measurable at 64k shots. Reserved
# knob #2 of NOISE_MODEL_SPEC.md; gate-side NT-margin impact to be re-checked
# after the reconstruction is nailed.)


def plaw(w, g):
    return (w ** 2 + W_IR ** 2) ** (-g / 2)


def gauss_pair(w, w0, sig):
    return 0.5 * (np.exp(-(w - w0) ** 2 / (2 * sig ** 2))
                  + np.exp(-(w + w0) ** 2 / (2 * sig ** 2)))


def chi(spec, t, n=200001):
    w = np.linspace(1e-6, W_SYNTH, n)
    return (2 / np.pi) * np.trapezoid(spec(w) * np.sin(w * t / 2) ** 2 / w ** 2, w)


def solve_qubit(g_el, f_el):
    """Solve (A_el, A_nuc) so that the electrical fraction at W_MID is f_el and
    chi(T2_TARGET) = 1. chi is linear in the amplitudes -> one-shot solve."""
    # fraction condition: A_nuc = A_el * (1/f_el - 1) * plaw(W_MID, g_el)/plaw(W_MID, G_NUC)
    ratio = (1 / f_el - 1) * plaw(W_MID, g_el) / plaw(W_MID, G_NUC)
    chi_unit = chi(lambda w: plaw(w, g_el) + ratio * plaw(w, G_NUC), T2_TARGET)
    a_el = 1.0 / chi_unit
    return a_el, a_el * ratio


def main():
    a_el1, a_nuc1 = solve_qubit(G_EL1, F_EL1)
    a_el2, a_nuc2 = solve_qubit(G_EL2, F_EL2)

    s_el_a = lambda w: a_el1 * plaw(w, G_EL1)
    s_el_b = lambda w: a_el2 * plaw(w, G_EL2)
    s_n1 = lambda w: a_nuc1 * plaw(w, G_NUC)
    s_n2 = lambda w: a_nuc2 * plaw(w, G_NUC)
    s_11 = lambda w: s_el_a(w) + s_n1(w)
    s_22 = lambda w: s_el_b(w) + s_n2(w)

    # line amplitudes (relative to the smooth total at each center)
    line_amp_q2 = LINE_FACTOR_Q2 * s_22(LINE_CENTERS)
    line_amp_q1 = LINE_FACTOR_Q1 * s_11(LINE_CENTERS)

    def s_n1_f(w):
        out = s_n1(w)
        for a, w0 in zip(line_amp_q1, LINE_CENTERS):
            out = out + a * gauss_pair(w, w0, LINE_SIGMA)
        return out

    def s_n2_f(w):
        out = s_n2(w)
        for a, w0 in zip(line_amp_q2, LINE_CENTERS):
            out = out + a * gauss_pair(w, w0, LINE_SIGMA)
        return out

    # J-channel scale: S_1212(W_MID) = R_1212 * sqrt(S_11 S_22)|_mid
    def s_1212(w, aj, bj):
        cross = C2_SHARE * np.sqrt(s_el_a(w) * s_el_b(w)) * np.cos(w * DT_SHIFT)
        return aj ** 2 * s_el_a(w) + bj ** 2 * s_el_b(w) - 2 * aj * bj * cross

    target = R_1212 * np.sqrt(s_11(W_MID) * s_22(W_MID))
    kappa = np.sqrt(target / s_1212(np.array([W_MID]), 1.0, BJ_OVER_AJ)[0])
    a_j, b_j = kappa, kappa * BJ_OVER_AJ

    # --- report --------------------------------------------------------------
    print("# Baked constants (paste into qns2q/noise/spectra.py):")
    print(f"A_EL_1  = {a_el1:.6e}")
    print(f"A_EL_2  = {a_el2:.6e}")
    print(f"A_NUC_1 = {a_nuc1:.6e}")
    print(f"A_NUC_2 = {a_nuc2:.6e}")
    print(f"A_J     = {a_j:.6e}")
    print(f"B_J     = {b_j:.6e}")
    print(f"LINE_AMP_Q1 = {np.array2string(line_amp_q1, precision=6)}")
    print(f"LINE_AMP_Q2 = {np.array2string(line_amp_q2, precision=6)}")

    # verification
    for tag, s11v, s22v, n1v, n2v in (("class M", s_11, s_22, s_n1, s_n2),
                                      ("class F",
                                       lambda w: s_el_a(w) + s_n1_f(w),
                                       lambda w: s_el_b(w) + s_n2_f(w),
                                       s_n1_f, s_n2_f)):
        t = np.linspace(1, 1100, 1100)
        chi1 = np.array([chi(s11v, ti, 40001) for ti in t])
        chi2 = np.array([chi(s22v, ti, 40001) for ti in t])
        t2_1 = t[np.argmin(np.abs(chi1 - 1))]
        t2_2 = t[np.argmin(np.abs(chi2 - 1))]
        print(f"[{tag}] T2*(1) = {t2_1:.0f} tau, T2*(2) = {t2_2:.0f} tau")

    w = np.linspace(0.02, W_SYNTH, 2000)
    cross = C2_SHARE * np.sqrt(s_el_a(w) * s_el_b(w)) * np.exp(-1j * w * DT_SHIFT)
    s1212 = s_1212(w, a_j, b_j)
    s112 = a_j * s_el_a(w) - b_j * cross
    s212 = a_j * np.conj(cross) - b_j * s_el_b(w)
    s11_f = s_el_a(w) + s_n1_f(w)
    s22_f = s_el_b(w) + s_n2_f(w)
    for tag, s11v, s22v in (("M", s_11(w), s_22(w)), ("F", s11_f, s22_f)):
        c12 = np.abs(cross) / np.sqrt(s11v * s22v)
        c112 = s112 / np.sqrt(s11v * s1212)
        c212 = s212 / np.sqrt(s22v * s1212)
        i_mid = np.argmin(np.abs(w - W_MID))
        print(f"[class {tag}] at W_MID: |c12| = {c12[i_mid]:.2f}, "
              f"c1,12 = {c112[i_mid]:.2f}, c2,12 = {c212[i_mid]:.2f}")
        # PSD min eigenvalue over the band
        mins = []
        for i in range(len(w)):
            mat = np.array([[s11v[i], cross[i], s112[i]],
                            [np.conj(cross[i]), s22v[i], s212[i]],
                            [np.conj(s112[i]), np.conj(s212[i]), s1212[i]]])
            mins.append(np.linalg.eigvalsh(mat)[0])
        print(f"[class {tag}] PSD min eigenvalue over band: {min(mins):.3e}")
    i_line = np.argmin(np.abs(w - 0.273))
    c12_m = np.abs(cross) / np.sqrt(s_11(w) * s_22(w))
    c12_f = np.abs(cross) / np.sqrt(s11_f * s22_f)
    print(f"coherence at line 0.273: class M {c12_m[i_line]:.2f} -> class F {c12_f[i_line]:.2f} (dip)")
    print(f"S(0): S_11 = {float(s_11(np.array([0.0]))[0]):.3e}, "
          f"S_22 = {float(s_22(np.array([0.0]))[0]):.3e}, "
          f"S_1212 = {float(s_1212(np.array([0.0]), a_j, b_j)[0]):.3e}")
    print(f"S_1212/sqrt(S11*S22) at W_MID = "
          f"{s1212[np.argmin(np.abs(w - W_MID))] / np.sqrt(s_11(W_MID) * s_22(W_MID)):.3f}")


if __name__ == "__main__":
    main()
