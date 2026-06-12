"""Solve the amplitude constants for the SHOWCASE noise landscape (SHOWCASE-0612).

Self-contained (does NOT import qns2q.noise.spectra -- it derives the constants
that get baked INTO that module's `showcase` regime branch). Companion to
scripts/calibrate_noise_model.py, which owns the anchored bland/featured classes.

Design (paper-repo REVIEW_TRACKER.md, REGIME-0612 post-meeting direction):
an engineered "trap" landscape -- contrived contrast, measured feature TYPES --
on which (i) the full spectral reconstruction is load-bearing, (ii) NT beats the
whole CDD/mqCDD library dramatically, (iii) protected gates land at infidelities
realistic for today's devices (1e-3 .. 1e-4).

Channel composition (tau units, two-sided spectra, same synthesis convention as
the anchored model: <b(t)b(t')> = (1/pi) int_0^{2wmax} S(w) cos(w(t-t')) dw):

  zeta_l  = e_X + n_l,  l in {1, 2}
  zeta_12 = A_J*e_A - B_J*e_B + j(t)          # j: independent coupler defect

  e_X     : quiet electrical floor, S_FL = A_FL_l * (w^2 + W_IR^2)^(-G_FL/2)
            [gamma 0.9 -- in the measured 1/f-charge-noise range; amplitude is
            the stylized knob: ~30 dB below the natural-Si anchored model]
  n_l     : qubit-local low-frequency + featured component =
            quasistatic hyperfine  A_QS_l * (w^2 + W_QS^2)^(-G_QS/2)
            + local TLF knee       H_TLF_l / (1 + (w/W_TLF)^2)   [Connors-2022
              knee SHAPE; catches CDD1-2, whose passbands sit below tooth 1]
            + trap lines           sum_k h_kl * Gauss(w, w_k, sig_k)
            [nuclear-difference-type lines ON comb teeth 1, 2, 4 (catch CDD3-5)
             plus one covering the top reachable window (catches the densest
             trains and any line-blind design fleeing up the falling floor)]
  j       : coupler TLF resonance, S_J = H_ZZL * Gauss(w, w_zz, sig_zz)
            + H_ZZK / (1 + (w/W_TLF)^2)        [structure ONLY the two-qubit
            spectra can reveal -- the rung-(c) ablation channel]

Targets:
  T2*(FID; chi(T2*) = 1) = 3500 tau per qubit   [Ge-hole 17.6 us at tau = 5 ns,
      Hendrickx 2024; equivalently purified-28Si at tau ~ 7 ns]
  best CDD/mqCDD at Tg = 320 tau  >= 3e-3       [today's-device gate error]
  NT(full recon) at Tg = 320 tau  <= 2e-4       [the dramatic-margin target]

Control scenario parameter: min pulse separation 8 tau (40 ns pi-pulses at the
5 ns anchor) => n <= 39 pulses/qubit at 320 tau, filter passbands confined to
the measured comb band [0.039, 0.785].

The proxy table below evaluates first-order chi for the actual cddn() pulse-time
generator plus uniform (CPMG-n) trains on the analytic landscape -- a fast
stand-in for the real library/optimizer used to pre-tune heights before the
gates_helper probe runs.
"""
import numpy as np

# --- comb / band geometry (QNSExperimentConfig defaults) -------------------------
T_COMB = 160.0
TRUNCATE = 20
WMAX = 2 * np.pi * TRUNCATE / T_COMB          # 0.7854
W_SYNTH = 2 * WMAX
TOOTH = 2 * np.pi / T_COMB                    # 0.0393 comb spacing

# --- fixed showcase constants -----------------------------------------------------
T2_TARGET = 3500.0
# Quasistatic-class slow bath (sets T2*). W_QS = 2.5e-3 (correlation time
# ~400 tau = 2 us at the 5 ns anchor) keeps the FID-slope DC protocol inside
# its linear window (t_max ~ 1600 tau >> 1/W_QS; at 1e-3 the dc_systematic
# bias hit ~50% of S(0) and the test-suite sanity bounds) while ALSO raising
# the in-band w^-2 tail that punishes CDD1-2 (S_QS(w >> W) ~ A ~ W at fixed
# T2*).
G_QS, W_QS = 2.0, 2.5e-3
G_FL, W_IR = 0.9, 0.02         # electrical floor: falling 1/f^0.9
W_TLF = 0.025                  # Connors-type knee position (catches CDD1-2)

# floor amplitude: the stylized "quiet electrical environment" knob.
# S_FL(0.30) ~ 1.2e-7 puts the NT parking-spot floor at the 2e-4 target.
S_FLOOR_AT = 0.30
S_FLOOR_VAL_1 = 0.85e-7
S_FLOOR_VAL_2 = 1.10e-7

# local TLF knee plateau (per qubit): CDD1 passband (w ~ 0.0098 at 320 tau)
# sits on the plateau; chi_CDD1 ~ 0.9 * H_TLF * 320 -> 3e-3 needs ~1e-5.
H_TLF_1 = 1.2e-5
H_TLF_2 = 1.5e-5

# trap lines: a HARMONIC FAMILY n*w0 (n = 1..4) of a single coherent defect at
# w0 = 0.051, plus one independent line covering the top reachable window.
# Rationale: (i) the dedup'd cddn() pulse counts at Tg = 320 tau are n = {1, 2,
# 5, 10, 21} so the CDD-ladder fundamentals sit at pi*n/Tg = {0.0098, 0.0196,
# 0.049, 0.098, 0.206} -- CDD1-2 live on the TLF-knee plateau, CDD3/4/5 land on
# harmonics 1/2/4 to within a line width, and harmonic 3 (0.153) closes the
# CPMG-16-shaped gap between them; (ii) min_sep = 8 tau caps passbands at
# pi*39/320 = 0.383, and the 0.357 line covers the [0.33, 0.39] top window
# where the densest trains AND any line-blind design fleeing up the falling
# floor must park; (iii) an f, 2f harmonic pair is a measured fingerprint
# (73Ge: Hendrickx 2024 f_L; HRL 2020 f_L + 2f_L) -- one defect + harmonics
# narrates cleaner than five unrelated lines. Centers are off-tooth but within
# ~1 tooth; sigma ~ half a tooth keeps each visible on 1-2 teeth (NNLS handles
# off-tooth centers).
# columns: center, sigma, height_q1, height_q2
W0_DEFECT = 0.051
LINES = np.array([
    [1 * W0_DEFECT, 0.016, 1.10e-5, 1.50e-5],   # CDD3 (n=5)
    [2 * W0_DEFECT, 0.020, 1.25e-5, 1.60e-5],   # CDD4 (n=10)
    [3 * W0_DEFECT, 0.018, 1.25e-5, 1.60e-5],   # CPMG-16-class gap filler
    [4 * W0_DEFECT, 0.022, 1.90e-5, 2.40e-5],   # CDD5 (n=21) + CPMG-21+-3
    [0.3650,        0.026, 1.60e-5, 2.00e-5],   # top window: CPMG-33..39 + blind
])

# coupler (ZZ) channel: smooth electrical difference (inherited, tiny) + an
# independent TLF resonance + knee only the 2Q spectra can see.
A_J_OVER = 0.43                # |A_J|: electrical-difference weights as in the
B_J_OVER = 0.45                # anchored model (S_1212 floor ~ 0.2 x selfs floor)
C2_SHARE = 0.8
DT_SHIFT = 1.5
ZZ_LINE_W0 = 6 * TOOTH         # 0.2356, inside NT's preferred gap
ZZ_LINE_SIG = 0.020
H_ZZ_LINE = 1.0e-5
H_ZZ_KNEE = 3.0e-6

# control scenario
MIN_SEP = 8.0
TG = 320.0


def plaw(w, g, wir):
    return (np.asarray(w) ** 2 + wir ** 2) ** (-g / 2)


def gauss_pair(w, w0, sig):
    w = np.asarray(w)
    return 0.5 * (np.exp(-(w - w0) ** 2 / (2 * sig ** 2))
                  + np.exp(-(w + w0) ** 2 / (2 * sig ** 2)))


def knee(w, h, wc):
    return h / (1 + (np.asarray(w) / wc) ** 2)


def chi_fid(spec, t, n=400001):
    """FID dephasing integral chi(t) = (2/pi) int S(w) sin^2(wt/2)/w^2 dw."""
    w = np.linspace(1e-7, W_SYNTH, n)
    return (2 / np.pi) * np.trapezoid(spec(w) * np.sin(w * t / 2) ** 2 / w ** 2, w)


# --- first-order chi for arbitrary pulse-time sequences ---------------------------
# Matches the gate models' convention: A(w) = sum_j (-1)^j (e^{iw t_{j+1}} -
# e^{iw t_j}); chi = (1/pi) int_0^inf S(w) |A(w)|^2 / w^2 dw  (validated below
# against the anchored model's chi_FID(320) = 0.38 acceptance number).

def cdd(t0, T, n):
    if n == 1:
        return [t0, t0 + T * 0.5]
    return [t0] + cdd(t0, T * 0.5, n - 1) + [t0 + T * 0.5] + cdd(t0 + T * 0.5, T * 0.5, n - 1)


def _dedup(lst):
    out, i = [], 0
    while i < len(lst):
        if i + 1 < len(lst) and lst[i] == lst[i + 1]:
            i += 2
        else:
            out.append(lst[i])
            i += 1
    return out


def cddn(T, n):
    out = _dedup(cdd(0., T, n))
    return out + [T] if out and out[0] == 0. else [0.] + out + [T]


def cpmg(T, n):
    return [0.] + [T * (k + 0.5) / n for k in range(n)] + [T]


def chi_seq(spec, tk, n=200001):
    tk = np.asarray(tk, dtype=float)
    w = np.linspace(1e-6, W_SYNTH, n)
    expt = np.exp(1j * np.outer(tk, w))
    signs = (-1.0) ** np.arange(len(tk) - 1)
    A = np.sum(signs[:, None] * (expt[1:] - expt[:-1]), axis=0)
    return (1 / np.pi) * np.trapezoid(spec(w) * np.abs(A) ** 2 / w ** 2, w)


def main():
    # --- solve the per-qubit quasistatic amplitude for T2* = 3500 -----------------
    a_fl1 = S_FLOOR_VAL_1 / plaw(S_FLOOR_AT, G_FL, W_IR)
    a_fl2 = S_FLOOR_VAL_2 / plaw(S_FLOOR_AT, G_FL, W_IR)

    def solve_qubit(a_fl, h_tlf, hts):
        fixed = lambda w: (a_fl * plaw(w, G_FL, W_IR) + knee(w, h_tlf, W_TLF)
                           + sum(h * gauss_pair(w, w0, s)
                                 for (w0, s), h in zip(LINES[:, :2], hts)))
        chi_fixed = chi_fid(fixed, T2_TARGET)
        if chi_fixed >= 1.0:
            raise RuntimeError(f"fixed components alone give chi(T2*) = "
                               f"{chi_fixed:.2f} >= 1 -- reduce knee/lines")
        chi_qs_unit = chi_fid(lambda w: plaw(w, G_QS, W_QS), T2_TARGET)
        a_qs = (1.0 - chi_fixed) / chi_qs_unit
        return a_qs, chi_fixed

    a_qs1, chif1 = solve_qubit(a_fl1, H_TLF_1, LINES[:, 2])
    a_qs2, chif2 = solve_qubit(a_fl2, H_TLF_2, LINES[:, 3])

    def s_nuc(l):
        a_qs, h_tlf = (a_qs1, H_TLF_1) if l == 1 else (a_qs2, H_TLF_2)
        hts = LINES[:, 1 + l]
        return lambda w: (a_qs * plaw(w, G_QS, W_QS) + knee(w, h_tlf, W_TLF)
                          + sum(h * gauss_pair(w, w0, s)
                                for (w0, s), h in zip(LINES[:, :2], hts)))

    s_el_1 = lambda w: a_fl1 * plaw(w, G_FL, W_IR)
    s_el_2 = lambda w: a_fl2 * plaw(w, G_FL, W_IR)
    s_11 = lambda w: s_el_1(w) + s_nuc(1)(w)
    s_22 = lambda w: s_el_2(w) + s_nuc(2)(w)

    def cross_el(w):
        return C2_SHARE * np.sqrt(s_el_1(w) * s_el_2(w)) * np.exp(-1j * w * DT_SHIFT)

    s_zz_extra = lambda w: (H_ZZ_LINE * gauss_pair(w, ZZ_LINE_W0, ZZ_LINE_SIG)
                            + knee(w, H_ZZ_KNEE, W_TLF))
    s_1212 = lambda w: (A_J_OVER ** 2 * s_el_1(w) + B_J_OVER ** 2 * s_el_2(w)
                        - 2 * A_J_OVER * B_J_OVER * np.real(cross_el(w))
                        + s_zz_extra(w))
    s_112 = lambda w: A_J_OVER * s_el_1(w) - B_J_OVER * cross_el(w)
    s_212 = lambda w: A_J_OVER * np.conj(cross_el(w)) - B_J_OVER * s_el_2(w)

    print("# Baked constants (paste into qns2q/noise/spectra.py, showcase branch):")
    print(f"A_FL_1   = {a_fl1:.6e}")
    print(f"A_FL_2   = {a_fl2:.6e}")
    print(f"A_QS_1   = {a_qs1:.6e}")
    print(f"A_QS_2   = {a_qs2:.6e}")
    print(f"H_TLF_1  = {H_TLF_1:.6e}")
    print(f"H_TLF_2  = {H_TLF_2:.6e}")
    print(f"LINE_CENTERS = {np.array2string(LINES[:, 0], precision=6)}")
    print(f"LINE_SIGMAS  = {np.array2string(LINES[:, 1], precision=6)}")
    print(f"LINE_AMP_Q1  = {np.array2string(LINES[:, 2], precision=6)}")
    print(f"LINE_AMP_Q2  = {np.array2string(LINES[:, 3], precision=6)}")
    print(f"H_ZZ_LINE = {H_ZZ_LINE:.6e}  (w0 = {ZZ_LINE_W0:.4f}, sig = {ZZ_LINE_SIG})")
    print(f"H_ZZ_KNEE = {H_ZZ_KNEE:.6e}  (W_TLF = {W_TLF})")
    print(f"[fixed-component chi(T2*) share: q1 {chif1:.3f}, q2 {chif2:.3f}]")

    # --- verification: T2*, PSD positivity ----------------------------------------
    t = np.linspace(100, 6000, 1200)
    for tag, s in (("q1", s_11), ("q2", s_22)):
        chis = np.array([chi_fid(s, ti, 60001) for ti in t])
        t2 = t[np.argmin(np.abs(chis - 1))]
        print(f"T2*({tag}) = {t2:.0f} tau   (target {T2_TARGET:.0f})")

    w = np.linspace(0.002, W_SYNTH, 3000)
    cr, s112v, s212v = cross_el(w), s_112(w), s_212(w)
    s11v, s22v, s1212v = s_11(w), s_22(w), s_1212(w)
    mins = []
    for i in range(len(w)):
        mat = np.array([[s11v[i], cr[i], s112v[i]],
                        [np.conj(cr[i]), s22v[i], s212v[i]],
                        [np.conj(s112v[i]), np.conj(s212v[i]), s1212v[i]]])
        mins.append(np.linalg.eigvalsh(mat)[0])
    print(f"PSD min eigenvalue over band: {min(mins):.3e}")
    print(f"S(0): S11 = {float(s_11(np.array([0.0]))[0]):.3e}, "
          f"S22 = {float(s_22(np.array([0.0]))[0]):.3e}, "
          f"S1212 = {float(s_1212(np.array([0.0]))[0]):.3e}")

    # --- convention check against the anchored featured model ---------------------
    # (A_EL_1 etc. from noise/spectra.py; expect chi_FID(320) ~ 0.38/qubit.)
    s11_anch = lambda w: (1.067936e-04 * plaw(w, 0.7, 0.02)
                          + 8.622470e-06 * plaw(w, 1.2, 0.02))
    print(f"\n[convention check] anchored-model chi_FID(320) = "
          f"{chi_fid(s11_anch, 320.0):.3f}  (spec: 0.38)")

    # --- proxy ladder at Tg = 320 tau, min separation 8 tau -----------------------
    print(f"\n# Proxy chi ladder at Tg = {TG:.0f} tau (per qubit; first order):")
    nmax = int(TG / MIN_SEP) - 1
    print(f"min_sep = {MIN_SEP} tau -> n_max = {nmax} pulses/qubit")

    seqs = [("FID", [0., TG])]
    k = 1
    while True:
        tk = cddn(TG, k)
        if np.min(np.diff(tk)) < MIN_SEP:
            break
        seqs.append((f"CDD{k} (n={len(tk) - 2})", tk))
        k += 1
    for n in (8, 12, 16, 20, 24, 28, 32, 36, nmax):
        if (n + 1) * MIN_SEP <= TG:
            seqs.append((f"CPMG-{n}", cpmg(TG, n)))

    chi_table = []
    for tag, tk in seqs:
        c1, c2 = chi_seq(s_11, tk), chi_seq(s_22, tk)
        # rough two-qubit error proxy: 1 - exp(-(c1 + c2)) (ZZ handled by probe)
        err = 1 - np.exp(-(c1 + c2))
        chi_table.append((tag, c1, c2, err))
        print(f"  {tag:16s} chi1 = {c1:.3e}  chi2 = {c2:.3e}  err ~ {err:.3e}")

    # NT parking proxy: best uniform train is an UPPER bound on NT (free
    # timings only help), so the gap floor is what matters:
    best_cpmg = min((e for t_, c1, c2, e in chi_table if t_.startswith("CPMG")))
    best_cdd = min((e for t_, c1, c2, e in chi_table if t_.startswith("CDD")))
    print(f"\nbest CDD proxy error : {best_cdd:.3e}   (target >= 3e-3)")
    print(f"best CPMG proxy error: {best_cpmg:.3e}   (NT upper bound)")

    # where would NT park? scan gap centers
    print("\n# Gap survey (S_11 + S_22 at candidate parking frequencies):")
    for wpark in (0.075, 0.14, 0.16, 0.26, 0.28, 0.30, 0.32, 0.44, 0.55, 0.70):
        print(f"  w = {wpark:.3f}: S11+S22 = "
              f"{float(s_11(np.array([wpark]))[0] + s_22(np.array([wpark]))[0]):.3e}"
              f"  -> chi*320*0.9 ~ {0.9 * TG * float(s_11(np.array([wpark]))[0] + s_22(np.array([wpark]))[0]):.3e}")

    # tooth-signal feasibility: weakest interesting teeth at 64k shots
    print("\n# Reconstruction-feasibility sketch (sweep T_eff ~ 1600 tau):")
    wk = TOOTH * np.arange(1, TRUNCATE + 1)
    s11k = s_11(wk)
    for kk in (1, 2, 4, 6, 8, 10, 12, 16, 20):
        C = 0.8 * s11k[kk - 1] * 1600
        print(f"  tooth {kk:2d} (w = {wk[kk - 1]:.3f}): S11 = {s11k[kk - 1]:.2e}, "
              f"decay C ~ {C:.2e} {'(>= 64k floor 4e-3)' if C >= 4e-3 else '(BELOW 64k floor)'}")


if __name__ == "__main__":
    main()
