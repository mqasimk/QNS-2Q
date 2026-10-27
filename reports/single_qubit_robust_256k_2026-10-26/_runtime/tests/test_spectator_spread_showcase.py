"""Integration regression for the spectator-state spread on the showcase noise.

Guards the paper's worst-case-spectator claim (App. embedding, Eq. worstcase) on the
ACTUAL reconstructed spectra and optimized idle, not random PSD matrices: build a real
N=3 (pair {0,1} + idling spectator {2}) overlap matrix from the demo's cross-spectra and
push it through the verified reduced_ptm_exact machinery of test_spectator_locality.py.

Asserts the structural facts that back the worst-case certificate -- the machinery and
the repo evaluator agree (convention), the overlap matrix is PSD, the worst spectator
state is a computational config, coherences are irrelevant -- and that the best--worst
spread sits at the few-percent level it was quantified at (a loose sanity band, not a
brittle exact-number check). Homogeneous-device assumption: the spectator couples like
the demo's own partner (the demo is 2-qubit and has no real third qubit). Idling
spectator => the two-body channel Z_lZ_k is filtered by the active qubit's train alone.

Skips cleanly if the showcase DraftRun data is absent.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QNS2Q_REGIME", "showcase")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "tsl", os.path.join(_HERE, "test_spectator_locality.py"))
tsl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tsl)

FOLDER = "DraftRun_NoSPAM_showcase_cap"
TG = 2560.0                                   # M=1 gate: no base-sequence folding


def _have_data():
    try:
        from qns2q.paths import project_root
    except Exception:
        return False
    opt = os.path.join(project_root(), FOLDER, "optimization_data_all_M_cap.npz")
    return os.path.exists(opt)


pytestmark = pytest.mark.skipif(not _have_data(),
                                reason="showcase DraftRun data not present")


def _nt_winner(opt, Tg):
    best = None
    for m in (int(x) for x in opt["M_values"]):
        gts = np.asarray(opt[f"M{m}_gate_times"], dtype=float)
        idx = np.where(np.isclose(gts, Tg))[0]
        if idx.size == 0:
            continue
        k = int(idx[0])
        inf = float(opt[f"M{m}_infs_opt"][k])
        seq = opt[f"M{m}_sequences_opt"][k]
        if seq is not None and (best is None or inf < best[0]):
            best = (inf, m, seq)
    return best


@pytest.fixture(scope="module")
def n3_problem():
    """Build the real N=3 overlap matrix and the reduced infidelities."""
    import jax.numpy as jnp
    from qns2q.control import idle as idmod
    from qns2q.paths import project_root

    opt = np.load(os.path.join(project_root(), FOLDER,
                  "optimization_data_all_M_cap.npz"), allow_pickle=True)
    win = _nt_winner(opt, TG)
    assert win is not None, "no NT winner at the chosen gate time"
    inf_nt, M, seq = win
    assert M == 1, "this regression fixes the M=1 gate to avoid folding subtleties"
    pt0, pt1 = jnp.asarray(seq[0]), jnp.asarray(seq[1])

    cfg = idmod.Config(fname=FOLDER, M=1, max_pulses=10**9, min_sep_factor=8.0)
    RMat, dt, nbs = idmod.prepare_time_domain_overlap(cfg.SMat, cfg.w, cfg.tau, TG, 1)
    ev = idmod.evaluate_overlap_folded
    idle = jnp.array([0., TG])
    tk01 = idmod.make_tk12(pt0, pt1)

    def ov(rc, fa, fb):
        return float(np.real(ev(fa, fb, RMat[rc[0], rc[1]], dt, nbs)))

    # bare pair (N=2) overlaps, for the convention cross-check
    I2 = np.zeros((3, 3))
    I2[0, 0] = ov((1, 1), pt0, pt0); I2[1, 1] = ov((2, 2), pt1, pt1)
    I2[2, 2] = ov((3, 3), tk01, tk01)
    I2[0, 1] = I2[1, 0] = ov((1, 2), pt0, pt1)
    I2[0, 2] = I2[2, 0] = ov((1, 3), pt0, tk01)
    I2[1, 2] = I2[2, 1] = ov((2, 3), pt1, tk01)
    I4 = np.zeros((4, 4), complex)
    I4[1, 1], I4[2, 2], I4[3, 3] = I2[0, 0], I2[1, 1], I2[2, 2]
    I4[1, 2] = I4[2, 1] = I2[0, 1]
    I4[1, 3] = I4[3, 1] = I2[0, 2]
    I4[2, 3] = I4[3, 2] = I2[1, 2]
    bare_repo = 1.0 - float(idmod.calculate_idling_fidelity(jnp.asarray(I4))) / 16.0
    # evaluate_overlap_folded returns the BARE overlap J = int(dw/2pi) S G; the exact
    # oracle (test_spectator_locality) uses the paper's 1/2-convention I = J/2
    # (<phi_a phi_b> = 2 I), so feed 0.5 * I into decay_matrix.
    W2 = tsl.decay_matrix(tsl.channels(2), 0.5 * I2, 2)
    bare_mach = 1.0 - np.trace(
        tsl.reduced_ptm_exact(W2, np.eye(1, dtype=complex), 2)).real / 16.0

    # N=3 overlap matrix (channels [Z0,Z1,Z2,Z01,Z02,Z12]); idling spectator + homogeneity.
    I3 = np.zeros((6, 6))

    def s(i, j, v):
        I3[i, j] = I3[j, i] = v
    s(0, 0, ov((1, 1), pt0, pt0)); s(1, 1, ov((2, 2), pt1, pt1)); s(2, 2, ov((1, 1), idle, idle))
    s(3, 3, ov((3, 3), tk01, tk01)); s(4, 4, ov((3, 3), pt0, pt0)); s(5, 5, ov((3, 3), pt1, pt1))
    s(0, 1, ov((1, 2), pt0, pt1)); s(0, 2, ov((1, 2), pt0, idle)); s(1, 2, ov((1, 2), pt1, idle))
    s(0, 3, ov((1, 3), pt0, tk01)); s(1, 3, ov((2, 3), pt1, tk01))
    s(0, 4, ov((1, 3), pt0, pt0)); s(2, 4, ov((2, 3), idle, pt0))     # q0 spectator-coupler driver
    s(1, 5, ov((1, 3), pt1, pt1)); s(2, 5, ov((2, 3), idle, pt1))     # q1 driver: aligned (worst geometry)

    W3 = tsl.decay_matrix(tsl.channels(3), 0.5 * I3, 3)   # bare J -> 1/2-convention I

    def infid(sig):
        return 1.0 - np.trace(tsl.reduced_ptm_exact(W3, np.asarray(sig, complex), 3)).real / 16.0

    zP = np.diag([1., 0.]); zM = np.diag([0., 1.]); mix = 0.5 * np.eye(2)
    coh = mix.copy().astype(complex); coh[0, 1] = coh[1, 0] = 0.3
    eq = 0.5 * (np.eye(2) + 0.36 * tsl.s1['X'] - 0.48 * tsl.s1['Y'])    # equatorial, <Z>=0
    return dict(inf_nt=inf_nt, bare_repo=bare_repo, bare_mach=bare_mach,
                mineig=float(np.linalg.eigvalsh(I3).min()),
                iP=infid(zP), iM=infid(zM), iMix=infid(mix), iCoh=infid(coh), iEq=infid(eq))


def test_convention_matches_repo_evaluator(n3_problem):
    """Machinery N=2 infidelity equals the repo's calculate_idling_fidelity, so the
    spread ratios sit on the same footing as the design numbers."""
    assert n3_problem["bare_mach"] == pytest.approx(n3_problem["bare_repo"], rel=1e-6)


def test_overlap_matrix_is_psd(n3_problem):
    assert n3_problem["mineig"] > -1e-12


def test_worst_state_is_config_on_real_noise(n3_problem):
    d = n3_problem
    worst = max(d["iP"], d["iM"])
    assert worst >= d["iEq"] - 1e-12          # equatorial never beats the worst config
    assert worst >= d["iMix"] - 1e-12         # mixed never beats the worst config
    assert d["iEq"] == pytest.approx(d["iMix"], abs=1e-12)   # equatorial == mixed (<Z>=0)


def test_coherence_irrelevant_on_real_noise(n3_problem):
    assert n3_problem["iCoh"] == pytest.approx(n3_problem["iMix"], abs=1e-12)


def test_spread_is_few_percent(n3_problem):
    """Best--worst spread is positive and at the few-percent level of the operational
    (spectator-present) infidelity -- a loose band, the quantified value is ~1.4%."""
    d = n3_problem
    spread = abs(d["iP"] - d["iM"])
    frac = spread / d["iMix"]
    assert 1e-4 < frac < 0.15
