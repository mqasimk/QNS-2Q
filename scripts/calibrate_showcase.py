"""Solve the amplitude constants for the SHOWCASE noise landscape (SHOWCASE-0612).

PLAIN-LANGUAGE ORIENTATION (read this first if you don't have the paper draft
open): this script is a one-off, by-hand "calibration notebook" -- NOT part of
the normal Stage-1..4 pipeline described in the repo-root CLAUDE.md. It does
not read any pipeline .npz file and it does not write any output file; it only
PRINTS a set of amplitude/height constants and some sanity-check diagnostics to
stdout. A human then copy-pastes those printed numbers, by hand, into the
`showcase` regime branch of `qns2q/noise/spectra.py` (grep that file for
"SHOWCASE-0612" -- its comment block there names this script as the source and
records the design rationale summarized here). Once pasted in, `spectra.py` is
the file every real pipeline stage (`characterize/`, `control/`, the
`scripts/*.py` entry points) actually imports at run time; THIS file is never
imported by anything else and is not re-run as part of any normal pipeline
invocation. Nothing else in the repo calls into it, and it deliberately does
NOT `import qns2q.noise.spectra` itself, since the entire point of running it
is to derive the very numbers that get hard-coded there afterward -- importing
the module here would be circular in spirit even though not literally circular
in Python. The anchored `bland`/`featured` classes (calibrated against real
published measurements, see NOISE_MODEL_SPEC.md) were instead calibrated by the
now-retired calibrate_noise_model.py; `showcase` differs from those in that its
component SHAPES are still physically motivated (same 1/f electrical floor,
quasistatic hyperfine term, Lorentzian "two-level-fluctuator" (TLF) knees, and
Gaussian spectral lines used elsewhere in this codebase) but their exact
heights/positions are chosen -- solved for, here -- to hit PERFORMANCE targets
for the paper's demonstration (see "Targets" below), not to match one specific
measured dataset.

Design (paper-repo REVIEW_TRACKER.md, REGIME-0612 post-meeting direction --
REVIEW_TRACKER.md lives in the separate companion-paper repo, not here, so this
tag is kept only as a provenance breadcrumb, not a thing you can grep locally):
build an engineered noise "trap" -- a landscape with deliberately stark
contrast between frequency bands, reusing the SAME feature TYPES as the
physically anchored bland/featured models -- such that, in the resulting
numerical experiment: (i) the full spectral reconstruction is load-bearing
(a control sequence designed from only PART of the reconstructed spectrum
cannot find the good frequency window), (ii) a noise-tailored (NT) gate --
one whose pulse timings are optimized against the actual reconstructed
spectrum, as opposed to a generic fixed-pattern sequence -- beats the whole
CDD/mqCDD library (CDD = Concatenated Dynamical Decoupling; mqCDD = per-qubit
CDD orders chosen independently; both built in `control/cz.py`/
`control/idle.py`) dramatically, and (iii) the resulting protected-gate
infidelities land in the range realistic for today's spin-qubit devices
(1e-3 .. 1e-4) -- so the demonstration isn't just "we picked a noise model with
a conveniently empty band."

Channel composition (tau units [tau = the minimum control-pulse separation,
the time unit used everywhere in this repo -- see CLAUDE.md], two-sided
spectra [the power spectral density S(w) is defined for w on the whole real
line, mirrored about w = 0], same synthesis convention as the anchored model:
<b(t)b(t')> = (1/pi) int_0^{2wmax} S(w) cos(w(t-t')) dw -- i.e. this is how a
one-sided S(w>=0) turns into an actual real-valued classical noise
trajectory b(t) in the rest of the pipeline):

  zeta_l  = e_X + n_l,  l in {1, 2}            # zeta_l: total dephasing noise
                                                # felt by qubit l; its PSD is
                                                # the self-spectrum S_11/S_22.
  zeta_12 = A_J*e_A - B_J*e_B + j(t)          # zeta_12: noise on the two-qubit
                                                # Ising (ZZ) coupling; its PSD
                                                # is S_1212. j: an independent
                                                # coupler-only defect term with
                                                # no single-qubit counterpart.

  e_X     : quiet electrical floor, S_FL = A_FL_l * (w^2 + W_IR^2)^(-G_FL/2)
            [gamma 0.9 -- in the measured 1/f-charge-noise range; amplitude is
            the stylized knob: ~30 dB below the natural-Si anchored model --
            i.e. deliberately turned down so the "quiet floor" leaves room for
            the noise-tailored gate to reach the 2e-4 target below]
  n_l     : qubit-local low-frequency + featured component =
            quasistatic hyperfine  A_QS_l * (w^2 + W_QS^2)^(-G_QS/2)
            [the slow nuclear-spin-bath term that mainly sets T2*, below]
            + local TLF knee       H_TLF_l / (1 + (w/W_TLF)^2)   [Connors-2022
              knee SHAPE; catches CDD1-2, whose passbands sit below tooth 1.
              "TLF" = two-level fluctuator, a single bistable charge/spin
              defect; "knee" = flat plateau below the corner frequency W_TLF,
              falling off as 1/w^2 above it. "Catches CDD1-2" means: those two
              decoupling sequences' frequency-domain sensitivity windows
              (filter functions, see chi_seq() below) sit in this knee's
              plateau, so this term is what makes THEM pay a real error cost.]
            + trap lines           sum_k h_kl * Gauss(w, w_k, sig_k)
            [nuclear-difference-type lines ON comb teeth 1, 2, 4 (catch CDD3-5)
             plus one covering the top reachable window (catches the densest
             trains and any line-blind design fleeing up the falling floor).
             "Comb teeth" = the discrete harmonic frequencies 2*pi*k/T_COMB
             that a real QNS reconstruction experiment actually measures;
             putting a line ON one means a sequence whose passband lands there
             (CDD3, CDD4, CDD5, ...) is punished, and that punishment is
             something the reconstruction can actually resolve/report.]
  j       : coupler TLF resonance, S_J = H_ZZL * Gauss(w, w_zz, sig_zz)
            + H_ZZK / (1 + (w/W_TLF)^2)        [structure ONLY the two-qubit
            spectra can reveal -- the rung-(c) ablation channel: in the
            paper's knowledge-ladder figure (showcase_design.pdf, see
            FIGURE_PROVENANCE.md) some characterization variants only ever see
            single-qubit spectra, never the cross-spectra; this term exists so
            that restriction has a concrete, measurable cost to lose, rather
            than the ablation being invisible/no-op]

Targets:
  T2*(FID; chi(T2*) = 1) = 3500 tau per qubit   [Ge-hole 17.6 us at tau = 5 ns,
      Hendrickx 2024; equivalently purified-28Si at tau ~ 7 ns]
      -- chi(t) is the first-order dephasing exponent of a free-induction-decay
      (Ramsey) experiment: coherence decays as exp(-chi(t)), so chi(t) = 1 is
      the standard definition of T2* (see chi_fid() below).
  best CDD/mqCDD at Tg = 320 tau  >= 3e-3       [today's-device gate error]
      -- Tg is the two-qubit gate duration. This says the BEST generic
      (non-noise-tailored) decoupling sequence run for that long should still
      leave an error no better than real present-day devices achieve, i.e. the
      landscape must be hard enough that naive decoupling doesn't accidentally
      already solve the problem.
  NT(full recon) at Tg = 320 tau  <= 2e-4       [the dramatic-margin target]
      -- NT = the noise-tailored gate (pulse timings optimized against the
      full reconstructed 6-spectrum model). This says THAT gate, and only that
      gate, should reach a much lower error -- the numerical demonstration of
      the paper's central claim.

Control scenario parameter: min pulse separation 8 tau (40 ns pi-pulses at the
5 ns anchor) => n <= 39 pulses/qubit at 320 tau, filter passbands confined to
the measured comb band [0.039, 0.785].
    -- the minimum allowed gap between consecutive pi-pulses caps how many
    pulses fit in the gate time (n <= 39) and therefore how high in frequency
    any sequence's filter function can reach. Confining that reach to the same
    band [0.039, 0.785] the QNS comb actually reconstructs (TOOTH..WMAX below)
    keeps this exercise self-consistent: no sequence considered here can
    "cheat" by hiding at a frequency a real characterization experiment could
    never see.

The proxy table below evaluates first-order chi for the actual cddn() pulse-time
generator plus uniform (CPMG-n) trains on the analytic landscape -- a fast
stand-in for the real library/optimizer used to pre-tune the noise-landscape
heights.
    -- i.e., instead of re-running the full (expensive, Monte-Carlo-based) QNS
    simulation and gate optimizer in `control/cz.py`/`control/idle.py` every
    time a constant here is nudged, this script re-implements just the
    first-order error estimate (chi) for the same pulse-sequence families the
    real optimizer would try, evaluated directly against the smooth analytic
    spectra defined above. That's enough to rank sequences and sanity-check
    the Targets before handing the tuned constants to the real (much slower)
    pipeline for the final, authoritative numbers.
"""
import numpy as np

# --- comb / band geometry (QNSExperimentConfig defaults) -------------------------
# These mirror the *default* values of characterize.experiments.QNSExperimentConfig
# (T, truncate) exactly, by hand, so the frequencies probed here line up with what
# a real QNS reconstruction of this same landscape would measure.
T_COMB = 160.0     # QNSExperimentConfig.T: total duration of one comb block (tau)
TRUNCATE = 20      # QNSExperimentConfig.truncate: highest reconstructed harmonic index k
WMAX = 2 * np.pi * TRUNCATE / T_COMB          # 0.7854  -- highest reconstructed
                                               # angular frequency omega_TRUNCATE
W_SYNTH = 2 * WMAX  # upper integration cutoff used below for ALL chi integrals;
                    # matches the "int_0^{2wmax}" upper limit in the module
                    # docstring's synthesis convention (real trajectories are
                    # synthesized over a band twice as wide as the reconstructed
                    # one), so these proxy integrals see the same physical band
                    # the real simulator would.
TOOTH = 2 * np.pi / T_COMB                    # 0.0393 comb spacing -- the spacing
                                               # between adjacent reconstructed
                                               # harmonics ("teeth") omega_k = k*TOOTH

# --- fixed showcase constants -----------------------------------------------------
T2_TARGET = 3500.0
# Quasistatic-class slow bath (sets T2*). W_QS = 2.5e-3 (correlation time
# ~400 tau = 2 us at the 5 ns anchor) keeps the FID-slope DC protocol inside
# its linear window (t_max ~ 1600 tau >> 1/W_QS; at 1e-3 the dc_systematic
# bias hit ~50% of S(0) and the test-suite sanity bounds) while ALSO raising
# the in-band w^-2 tail that punishes CDD1-2 (S_QS(w >> W) ~ A ~ W at fixed
# T2*).
# In plain terms: W_QS sets how "slow" the quasistatic hyperfine noise is (its
# correlation time is 1/W_QS); picking it too small (too slow/too close to DC)
# would break the DC-point measurement's assumption that the qubit's decay
# looks linear-in-time over the fit window, while picking it too large would
# stop this term from also reaching into the CDD1-2 passbands (see docstring).
G_QS, W_QS = 2.0, 2.5e-3
G_FL, W_IR = 0.9, 0.02         # electrical floor: falling 1/f^0.9 power law
                               # (G_FL = the power-law exponent gamma, W_IR =
                               # the infrared/low-frequency regularization
                               # cutoff that keeps S_FL(w=0) finite -- see
                               # plaw() below)
W_TLF = 0.025                  # Connors-type knee position (catches CDD1-2):
                               # the corner frequency of the Lorentzian "knee"
                               # shape, i.e. where it stops being flat and
                               # starts falling as 1/w^2 (see knee() below)

# floor amplitude: the stylized "quiet electrical environment" knob.
# S_FL(0.30) ~ 1.2e-7 puts the NT parking-spot floor at the 2e-4 target.
# "NT parking spot/window" = the frequency sub-band, described further below
# (around w ~ 0.28), where a noise-tailored sequence's filter function can sit
# with the least residual noise -- these two values are chosen so that the
# noise level AT that gap is exactly the one that produces the paper's 2e-4
# target NT gate error.
S_FLOOR_AT = 0.30
S_FLOOR_VAL_1 = 1.00e-7
S_FLOOR_VAL_2 = 1.30e-7

# Local TLF knee on the SELFS: dropped (probe iteration 4, i.e. the fourth
# round of hand-tuning these constants against the Targets above -- "probe
# iteration" markers throughout this file are just a lab-notebook trail of
# which round of manual tuning a choice/rejection came from). With W_QS =
# 2.5e-3 the T2*-carrying quasistatic tail already punishes CDD1 at ~2e-2 and
# CDD2 at ~6e-3 by itself (S_QS(0.0196) ~ 1e-5), while the knee's w^-2 tail
# was the DOMINANT toll on the NT parking window (2.2e-7 of 4.7e-7 at 0.28).
# The Connors-knee citation survives on the ZZ channel's small knee.
# In short: a knee on the single-qubit (self-spectrum) channels turned out to
# be redundant with the quasistatic term for punishing CDD1-2, AND it leaked
# too much unwanted extra noise into the NT parking window, so its height is
# set to zero here (self-spectra) while a separate, much smaller knee is kept
# on the two-qubit (ZZ) channel only -- see H_ZZ_KNEE below.
H_TLF_1 = 0.0
H_TLF_2 = 0.0

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
#
# In plain terms: rather than scattering five unrelated noise peaks, this
# places ONE physical defect's fundamental frequency w0 plus its first four
# overtones (n*w0), because that is what a real coherent defect's spectral
# fingerprint looks like (and matches measured Ge/HRL devices, cited above).
# Their positions are deliberately chosen to line up with where each rung of
# the CDD ladder (CDD1, CDD2, ... -- built by cddn() below) is most sensitive,
# so that increasing the CDD order does NOT monotonically help (each new rung
# just walks onto the next line) -- this is what makes "the whole CDD/mqCDD
# library" underperform in the Design goal above, not just one or two orders
# of it. "NNLS" = non-negative least-squares, the reconstruction algorithm
# used downstream (`characterize/inversion.py`) that enforces S(w) >= 0; it
# can still resolve a line whose center falls between two comb teeth, so the
# exact center frequencies don't need to land exactly ON a tooth.
# columns: center, sigma, height_q1, height_q2
# The NT parking window lives between the 4*w0 line's upper 3-sigma edge and
# the top line's lower 3-sigma edge: [0.258, 0.312] with the widths below.
# ("Parking window" = the frequency gap where a noise-tailored sequence's
# filter function can sit with the lowest possible noise exposure -- "3-sigma
# edge" = three standard deviations out from a Gaussian line's center, i.e.
# essentially where that line's contribution has died off.)
# Probe iteration 2026-06-12: the first cut (sigma 0.022/0.026, top at 0.365)
# left only [0.27, 0.287] -- the NT winner sat on the top line's 2.5-sigma
# shoulder and paid ~2x the floor. Narrower flanks + a recentred top line open
# the window; heights go UP to keep CDD5 and the blind flee-up punished.
W0_DEFECT = 0.051
LINES = np.array([
    [1 * W0_DEFECT, 0.016, 2.05e-5, 2.75e-5],   # CDD3 (n=5)
    [2 * W0_DEFECT, 0.020, 2.45e-5, 3.15e-5],   # CDD4 (n=10)
    [3 * W0_DEFECT, 0.018, 2.25e-5, 2.85e-5],   # CPMG-16-class gap filler
    [4 * W0_DEFECT, 0.014, 1.15e-4, 1.44e-4],   # CDD5 (n=21) + CPMG-21+-3
    [0.3720,        0.015, 9.10e-5, 1.13e-4],   # top window: CPMG-34..39 + blind
])

# coupler (ZZ) channel: smooth electrical difference (inherited, tiny) + an
# independent TLF resonance + knee only the 2Q spectra can see.
# zeta_12 = A_J*e_A - B_J*e_B + j(t) (see module docstring); A_J_OVER/B_J_OVER
# below are |A_J|, |B_J| -- how strongly each qubit's own electrical field
# noise leaks into the coupling channel.
A_J_OVER = 0.43                # |A_J|: electrical-difference weights as in the
B_J_OVER = 0.45                # anchored model (S_1212 floor ~ 0.2 x selfs floor)
C2_SHARE = 0.8                 # fraction of each qubit's electrical floor noise
                                # that is SHARED (common-mode) between the two
                                # qubits, vs. independently drawn; feeds cross_el()
                                # below, which builds the qubit1-qubit2 cross-spectrum.
DT_SHIFT = 4.0  # showcase Im-part generator (matches noise/spectra.py, 2026-06-16):
                # a causal time lag (in tau) between the shared component as seen
                # by qubit 1 vs qubit 2. A real (non-zero) lag is what makes the
                # cross-spectral density genuinely COMPLEX (nonzero imaginary
                # part) rather than just a positive real correlation -- see
                # cross_el() below, where it enters as a phase factor exp(-i*w*DT_SHIFT).
ZZ_LINE_W0 = 6 * TOOTH         # 0.2356, inside NT's preferred gap
ZZ_LINE_SIG = 0.020
H_ZZ_LINE = 1.4e-5
# Probe iteration: 3e-6 cost the full-NT design ~1.1e-4 through the CZ's
# FORCED low-frequency ZZ exposure (dc_12 >= pi/(4 Jmax) -- no design dodges
# it); 1e-6 keeps the 2Q-only structure visible while the rung-(c)
# punishment rides the dodgeable LINE, not the undodgeable knee.
# In plain terms: any CZ (controlled-Z) gate needs the accumulated ZZ coupling
# exposure `dc_12` (in `control/cz.py`) to reach a minimum value set by the
# coupling strength Jmax, no matter how the pulses are timed -- so noise sitting
# right at zero frequency (DC) on the ZZ channel can never be dodged by ANY
# design, tailored or not. Putting the punishment on the movable Gaussian LINE
# instead of on this always-present DC "knee" keeps the demonstration fair:
# the ablation-ladder rung that loses cross-spectrum knowledge should lose
# because it can't see/avoid the (avoidable) line, not because of an
# unavoidable DC floor every design pays regardless.
H_ZZ_KNEE = 0.5e-6

# control scenario: matches the "Control scenario parameter" paragraph of the
# module docstring above (minimum pulse separation and total gate time, both
# in tau).
MIN_SEP = 8.0
TG = 320.0


def plaw(w, g, wir):
    """IR-regularized power-law spectral shape: S(w) = (w^2 + wir^2)^(-g/2).

    Generic building block reused for both the "quiet electrical floor" (e_X)
    and the quasistatic hyperfine term (with different (g, wir) each time);
    `wir` keeps S(0) finite (a bare 1/f^g power law would diverge at w = 0).
    """
    return (np.asarray(w) ** 2 + wir ** 2) ** (-g / 2)


def gauss_pair(w, w0, sig):
    """Two-sided Gaussian line: mirrored peaks at +w0 and -w0.

    A physical resonance lives at a single positive frequency w0, but these
    are two-sided spectra (S(w) defined for all real w, per the module
    docstring's synthesis convention), so the peak is mirrored about w = 0 to
    keep S(w) = S(-w) as required for a real-valued classical noise process.
    """
    w = np.asarray(w)
    return 0.5 * (np.exp(-(w - w0) ** 2 / (2 * sig ** 2))
                  + np.exp(-(w + w0) ** 2 / (2 * sig ** 2)))


def knee(w, h, wc):
    """Lorentzian "TLF knee": flat at height h below wc, falling as 1/w^2 above it.

    Phenomenological shape (Connors et al. 2022) for a two-level-fluctuator
    (TLF) -- a single bistable charge/spin defect switching between two states.
    """
    return h / (1 + (np.asarray(w) / wc) ** 2)


def chi_fid(spec, t, n=400001):
    """FID dephasing integral chi(t) = (2/pi) int S(w) sin^2(wt/2)/w^2 dw.

    `spec` is a spectral-density function S(w) (any of the s_11/s_22/... below);
    `t` is the free-evolution (Ramsey / free-induction-decay, "FID") time. The
    returned chi(t) is the first-order dephasing exponent: coherence decays as
    exp(-chi(t)). `n` sets the number of points in the trapezoid-rule frequency
    grid used for the integral -- purely a numerical-accuracy knob, not a
    physics parameter.
    """
    w = np.linspace(1e-7, W_SYNTH, n)
    return (2 / np.pi) * np.trapezoid(spec(w) * np.sin(w * t / 2) ** 2 / w ** 2, w)


# --- first-order chi for arbitrary pulse-time sequences ---------------------------
# Matches the gate models' convention: A(w) = sum_j (-1)^j (e^{iw t_{j+1}} -
# e^{iw t_j}); chi = (1/pi) int_0^inf S(w) |A(w)|^2 / w^2 dw  (validated below
# against the anchored model's chi_FID(320) = 0.38 acceptance number).
# In plain terms: A(w) is the "filter function" of a pulse sequence with
# pi-pulses at times t_0=0, t_1, ..., t_{n+1}=T -- it encodes how sensitive the
# sequence is to noise at each frequency w (an ideal decoupling sequence makes
# |A(w)|^2 small at the frequencies where the noise power S(w) is large). This
# chi_seq() below generalizes chi_fid() (a plain FID has zero pulses) to any
# such pulse train, so the same "chi(t)=1 <=> T2*" logic and the same
# first-order dephasing-exponent meaning applies.

def cdd(t0, T, n):
    """Recursively build the pi-pulse times of a CDD_n sequence on [t0, t0+T].

    CDD (Concatenated Dynamical Decoupling) of order n is defined by nesting a
    single pi-pulse at the sequence's midpoint inside two copies of a CDD_(n-1)
    sequence, one on each half of the interval -- that's exactly what the
    recursion below does. Returns a plain Python list of pulse times (not
    including the interval endpoints t0/t0+T); see cddn() for the full,
    endpoint-including, duplicate-free sequence actually used below.
    """
    if n == 1:
        return [t0, t0 + T * 0.5]
    return [t0] + cdd(t0, T * 0.5, n - 1) + [t0 + T * 0.5] + cdd(t0 + T * 0.5, T * 0.5, n - 1)


def _dedup(lst):
    """Collapse adjacent equal entries in a sorted time list into nothing.

    CDD's recursive construction can place two pulses back-to-back at the same
    instant (the midpoint of one sub-interval coincides with the start/end of
    the next); two coincident pi-pulses are physically a no-op (they cancel),
    so this helper removes such pairs before the sequence is used. `lst` is a
    plain list of pulse times; not a physics quantity itself, just scratch data.
    """
    out, i = [], 0
    while i < len(lst):
        if i + 1 < len(lst) and lst[i] == lst[i + 1]:
            i += 2
        else:
            out.append(lst[i])
            i += 1
    return out


def cddn(T, n):
    """Full CDD_n pulse-time list on [0, T], endpoints included, duplicates removed.

    Wraps cdd() + _dedup() and makes sure the list starts at 0 and ends at T
    (cdd()'s raw recursion may or may not already include those endpoints,
    hence the conditional). This is the actual sequence-time generator used
    both here and by the real optimizer in control/cz.py, control/idle.py.
    """
    out = _dedup(cdd(0., T, n))
    return out + [T] if out and out[0] == 0. else [0.] + out + [T]


def cpmg(T, n):
    """CPMG-n pulse-time list: n equally spaced pi-pulses over [0, T].

    (Carr-Purcell-Meiboom-Gill sequence -- pulses at the centers of n equal
    sub-intervals, plus the interval endpoints 0 and T.)
    """
    return [0.] + [T * (k + 0.5) / n for k in range(n)] + [T]


def chi_seq(spec, tk, n=200001):
    """First-order dephasing exponent chi for an arbitrary pulse-time sequence.

    Generalizes chi_fid() to any pi-pulse train: `tk` is the full list of pulse
    times (including both endpoints, e.g. from cddn()/cpmg() above), and `A` is
    that sequence's filter function (see the block comment above this
    function). `n` is again just the frequency-grid resolution.
    """
    tk = np.asarray(tk, dtype=float)
    w = np.linspace(1e-6, W_SYNTH, n)
    expt = np.exp(1j * np.outer(tk, w))
    signs = (-1.0) ** np.arange(len(tk) - 1)
    A = np.sum(signs[:, None] * (expt[1:] - expt[:-1]), axis=0)
    return (1 / np.pi) * np.trapezoid(spec(w) * np.abs(A) ** 2 / w ** 2, w)


def main():
    """Derive, print, and sanity-check the showcase-regime amplitude constants.

    Runs top to bottom as one script (there is no other entry point): (1)
    solves the per-qubit quasistatic-hyperfine amplitude that makes T2* land
    on target given all the other (fixed) noise components, (2) assembles the
    full six analytic spectra (S_11, S_22, S_1212, and the three cross-spectra)
    from those solved-for and hand-set constants, (3) prints the constants in a
    form meant to be pasted directly into noise/spectra.py's showcase branch,
    (4) re-derives T2* from the assembled spectra as an independent check that
    step (1)'s solve actually worked, checks the 3x3 cross-spectral matrix is
    positive-semi-definite (PSD) at every frequency (a physical requirement:
    it must be a valid, that is realizable, correlation matrix -- a Fourier
    transform of a real physical cross-correlation cannot have negative
    "power" in any direction), and (5) runs the first-order chi proxy scan
    described in the module docstring to check the Targets are plausible
    before handing these constants to the real (Monte Carlo) pipeline.
    """
    # --- solve the per-qubit electrical-floor amplitude from its pinned value at
    # S_FLOOR_AT, then the per-qubit quasistatic amplitude that makes T2* land on
    # T2_TARGET given everything else already fixed --------------------------------
    a_fl1 = S_FLOOR_VAL_1 / plaw(S_FLOOR_AT, G_FL, W_IR)
    a_fl2 = S_FLOOR_VAL_2 / plaw(S_FLOOR_AT, G_FL, W_IR)

    def solve_qubit(a_fl, h_tlf, hts):
        """Solve for the quasistatic amplitude a_qs that puts chi(T2_TARGET) = 1.

        `fixed` bundles every noise component whose amplitude is ALREADY
        pinned (the electrical floor a_fl, the TLF knee height h_tlf, and this
        qubit's trap-line heights hts) -- i.e. everything except the
        quasistatic hyperfine term. Because chi is linear in the spectral
        density (first-order perturbation theory: it's just a weighted
        integral of S(w)), the quasistatic amplitude needed to make up
        whatever chi "deficit" is left after the fixed components is a simple
        rescaling: (1 - chi_fixed) / (chi contributed by a unit-amplitude
        quasistatic term). Returns (a_qs, chi_fixed) -- the solved amplitude
        and the fixed components' own share of chi(T2_TARGET), which is also
        printed below as a diagnostic.
        """
        fixed = lambda w: (a_fl * plaw(w, G_FL, W_IR) + knee(w, h_tlf, W_TLF)
                           + sum(h * gauss_pair(w, w0, s)
                                 for (w0, s), h in zip(LINES[:, :2], hts)))
        chi_fixed = chi_fid(fixed, T2_TARGET)
        if chi_fixed >= 1.0:
            # Physically: the fixed components ALONE already dephase the qubit
            # past T2* by the target time, so no (non-negative) quasistatic
            # amplitude could fix that -- one of the fixed knobs above must be
            # turned down instead.
            raise RuntimeError(f"fixed components alone give chi(T2*) = "
                               f"{chi_fixed:.2f} >= 1 -- reduce knee/lines")
        chi_qs_unit = chi_fid(lambda w: plaw(w, G_QS, W_QS), T2_TARGET)
        a_qs = (1.0 - chi_fixed) / chi_qs_unit
        return a_qs, chi_fixed

    a_qs1, chif1 = solve_qubit(a_fl1, H_TLF_1, LINES[:, 2])
    a_qs2, chif2 = solve_qubit(a_fl2, H_TLF_2, LINES[:, 3])

    def s_nuc(l):
        """Qubit-l "nuclear" component n_l: quasistatic term + TLF knee + trap lines.

        `l` selects qubit 1 or 2 (matching the paper's zeta_l = e_X + n_l
        decomposition in the module docstring); LINES[:, 1 + l] picks out that
        qubit's line-height column (index 2 for l=1, index 3 for l=2 -- see
        the LINES array's "columns: center, sigma, height_q1, height_q2" comment).
        """
        a_qs, h_tlf = (a_qs1, H_TLF_1) if l == 1 else (a_qs2, H_TLF_2)
        hts = LINES[:, 1 + l]
        return lambda w: (a_qs * plaw(w, G_QS, W_QS) + knee(w, h_tlf, W_TLF)
                          + sum(h * gauss_pair(w, w0, s)
                                for (w0, s), h in zip(LINES[:, :2], hts)))

    # s_el_1/s_el_2: pure electrical-floor component e_X for each qubit (no
    # nuclear/hyperfine part); s_11/s_22: full self-spectra zeta_l = e_X + n_l.
    s_el_1 = lambda w: a_fl1 * plaw(w, G_FL, W_IR)
    s_el_2 = lambda w: a_fl2 * plaw(w, G_FL, W_IR)
    s_11 = lambda w: s_el_1(w) + s_nuc(1)(w)
    s_22 = lambda w: s_el_2(w) + s_nuc(2)(w)

    def cross_el(w):
        """Cross-spectral density between qubits 1 and 2's SHARED electrical field.

        C2_SHARE sets how much of each qubit's electrical floor is common-mode
        (see its definition above); the exp(-i*w*DT_SHIFT) phase factor is the
        causal lag that gives this cross-spectrum a genuine imaginary part
        (see DT_SHIFT above). This feeds the two-qubit cross-spectra below
        (S_1_2 is not printed separately here since it's exactly this
        function, up to the paper's normalization -- what IS printed is the
        constants that reproduce it once plugged into noise/spectra.py).
        """
        return C2_SHARE * np.sqrt(s_el_1(w) * s_el_2(w)) * np.exp(-1j * w * DT_SHIFT)

    # s_zz_extra: the independent coupler defect j(t)'s PSD (line + knee, see
    # module docstring); s_1212/s_112/s_212: the ZZ self-spectrum and the two
    # qubit-coupler cross-spectra, built from zeta_12 = A_J*e_A - B_J*e_B + j(t)
    # by expanding <zeta_a* zeta_b> in terms of the s_el_*/cross_el pieces above
    # (e.g. s_1212 = <zeta_12* zeta_12> picks up A_J^2*S(e_A) + B_J^2*S(e_B) -
    # 2*A_J*B_J*Re(cross-term) + S_J, since j(t) is independent of e_A, e_B).
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
    # Re-derive T2* directly from the assembled s_11/s_22 (rather than trusting
    # solve_qubit()'s algebra blindly): scan candidate FID times t and report
    # whichever one makes chi_fid(t) closest to 1 -- this should reproduce
    # T2_TARGET if the solve above is self-consistent.
    t = np.linspace(100, 6000, 1200)
    for tag, s in (("q1", s_11), ("q2", s_22)):
        chis = np.array([chi_fid(s, ti, 60001) for ti in t])
        t2 = t[np.argmin(np.abs(chis - 1))]
        print(f"T2*({tag}) = {t2:.0f} tau   (target {T2_TARGET:.0f})")

    # Physical-validity check: at every frequency, the instantaneous 3x3
    # Hermitian matrix of (self- and cross-) spectral densities must be
    # positive semi-definite (PSD) -- this is required for it to be a valid
    # spectral (cross-)density matrix of an actual physical noise process
    # (a negative eigenvalue would mean some linear combination of the noise
    # channels has "negative power", which is unphysical). `mat` below is
    # exactly the 3x3 matrix [[S11, S1_2, S1_12], [S1_2*, S22, S2_12],
    # [S1_12*, S2_12*, S1212]] the reconstruction/optimizer code assembles
    # elsewhere (e.g. the optimizer's SMat, see CLAUDE.md); checking its
    # smallest eigenvalue stays >= 0 across the band is a cheap way to catch a
    # bad choice of constants before they are ever used downstream.
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
    # This is a pure cross-check that chi_fid()'s numerical CONVENTION here
    # matches the one used to calibrate the (separately maintained) anchored
    # bland/featured models: plug in the anchored model's own S_11 constants
    # (copied by hand from noise/spectra.py, not imported -- see the module
    # docstring's explanation of why this script avoids importing spectra.py)
    # and confirm chi_fid(320) reproduces the 0.38 acceptance number recorded
    # in NOISE_MODEL_SPEC.md for that model. It says nothing about whether the
    # SHOWCASE constants above are right, only that the integral machinery
    # used to derive them is consistent with the rest of the codebase.
    s11_anch = lambda w: (1.067936e-04 * plaw(w, 0.7, 0.02)
                          + 8.622470e-06 * plaw(w, 1.2, 0.02))
    print(f"\n[convention check] anchored-model chi_FID(320) = "
          f"{chi_fid(s11_anch, 320.0):.3f}  (spec: 0.38)")

    # --- proxy ladder at Tg = 320 tau, min separation 8 tau -----------------------
    # Build every CDD_k order that still respects the minimum pulse separation
    # (k increases until pulses would have to be closer together than
    # MIN_SEP), plus a handful of CPMG-n trains at increasing pulse counts, and
    # score each one's first-order dephasing error against this landscape.
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
        # -- a quick stand-in for the real two-qubit gate infidelity, assuming
        # (i) each qubit's dephasing contributes independently and (ii) the
        # ZZ (coupler) channel's own contribution is negligible here (checked
        # separately, by hand, in earlier "probe" iterations -- not computed
        # by this loop). Good enough to RANK candidate sequences quickly; not
        # a substitute for the real optimizer's fidelity calculation.
        err = 1 - np.exp(-(c1 + c2))
        chi_table.append((tag, c1, c2, err))
        print(f"  {tag:16s} chi1 = {c1:.3e}  chi2 = {c2:.3e}  err ~ {err:.3e}")

    # NT parking proxy: best uniform train is an UPPER bound on NT (free
    # timings only help), so the gap floor is what matters:
    # -- CDD/CPMG use fixed, structured pulse-timing patterns, while NT is free
    # to place its pulses anywhere; NT can therefore never do WORSE than the
    # best member of these constrained families, only as well or better. So
    # the best (lowest-error) CDD/CPMG entry above is a pessimistic (upper)
    # bound on the real NT optimizer's error -- what actually limits how low
    # NT can go is the residual noise "floor" in whatever frequency gap it
    # would park a filter function in, which the next block estimates directly.
    best_cpmg = min((e for t_, c1, c2, e in chi_table if t_.startswith("CPMG")))
    best_cdd = min((e for t_, c1, c2, e in chi_table if t_.startswith("CDD")))
    print(f"\nbest CDD proxy error : {best_cdd:.3e}   (target >= 3e-3)")
    print(f"best CPMG proxy error: {best_cpmg:.3e}   (NT upper bound)")

    # where would NT park? scan gap centers
    # -- rough estimate of what chi a sequence could achieve if its ENTIRE
    # filter-function weight sat at a single candidate frequency wpark (a
    # crude proxy for "if NT could park perfectly at this frequency"); 0.9 is
    # just an empirical fudge factor from comparing this proxy to the real
    # chi_seq() calculation on past iterations, not a physical constant.
    print("\n# Gap survey (S_11 + S_22 at candidate parking frequencies):")
    for wpark in (0.075, 0.14, 0.16, 0.26, 0.28, 0.30, 0.32, 0.44, 0.55, 0.70):
        print(f"  w = {wpark:.3f}: S11+S22 = "
              f"{float(s_11(np.array([wpark]))[0] + s_22(np.array([wpark]))[0]):.3e}"
              f"  -> chi*320*0.9 ~ {0.9 * TG * float(s_11(np.array([wpark]))[0] + s_22(np.array([wpark]))[0]):.3e}")

    # tooth-signal feasibility: weakest interesting teeth at 64k shots
    # -- separate from the gate-error targets above, this checks that the
    # spectral FEATURES placed on specific comb teeth would actually be
    # measurable/reconstructable in a real (finite-shot-noise) QNS experiment,
    # not just present in the noise-free analytic model. `C` is a rough decay-
    # contrast proxy (how much coherence decay a tooth's noise power would
    # produce over an effective reconstruction time T_eff ~ 1600 tau); "64k
    # floor" is the empirically-observed statistical noise floor of a
    # 64000-shot QNS run (below which a real feature is indistinguishable from
    # shot noise) -- so a tooth's decay contrast needs to clear that floor for
    # its line to be trustworthy in the reconstructed spectra downstream.
    print("\n# Reconstruction-feasibility sketch (sweep T_eff ~ 1600 tau):")
    wk = TOOTH * np.arange(1, TRUNCATE + 1)
    s11k = s_11(wk)
    for kk in (1, 2, 4, 6, 8, 10, 12, 16, 20):
        C = 0.8 * s11k[kk - 1] * 1600
        print(f"  tooth {kk:2d} (w = {wk[kk - 1]:.3f}): S11 = {s11k[kk - 1]:.2e}, "
              f"decay C ~ {C:.2e} {'(>= 64k floor 4e-3)' if C >= 4e-3 else '(BELOW 64k floor)'}")


if __name__ == "__main__":
    main()
