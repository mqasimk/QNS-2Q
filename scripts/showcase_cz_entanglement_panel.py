"""SHOWCASE CZ entanglement-GENERATION panel: the entangler's counterpart to
the idle storage panel (scripts/showcase_storage_panel.py).

This is a pure READ-OUT. It loads the already-optimized CZ sequences and the
reconstructed/true spectra from a finished DraftRun and computes a NEW quantity
-- the entanglement the noisy CZ *generates* -- without touching the QNS
reconstruction or the NT optimization. Nothing here re-optimizes anything.

Physics
-------
The maximally entangling CZ (reduced to exp(-i pi/4 Z1 Z2) once the local Z
rotations are virtual) maps the product input |++> to

    |chi> = exp(-i pi/4 Z1 Z2)|++>
          = (1/sqrt2)[ e^{-i pi/4}|Phi+> + e^{+i pi/4}|Psi+> ],

a maximally entangled state (concurrence 1). Under the gate's residual
dephasing channel each computational-basis coherence |I><J| is damped by

    lambda_IJ = exp( -1/2 sum_{a,b in {1,2,12}} n_a n_b Re I[a,b] ),
    n_a = (z_a^I - z_a^J)/2  in {-1,0,1},

with I[a,b] the SAME folded overlap integrals the CZ optimizer uses
(control.cz.evaluate_overlap_folded; indices 1=Z1, 2=Z2, 3=Z1Z2). This is the
exact two-qubit-dephasing map; it reduces to the storage panel's bell_infs for
the Bell coherences and is consistent with that panel's Monte-Carlo check.

Two faces, mirroring the storage story:
  (a) GENERATED-PAIR QUALITY. 1 - concurrence of |chi> after the noisy CZ, for
      FID / best-CDD / best-NT. The NT entangler generates higher-fidelity
      entanglement, not merely a better average -- the both-gates-equal-footing
      point. (For the symmetric |chi> the first-order cross term cancels, so
      this panel is the self-spectrum + second-order-cross story.)
  (b) WHICH BELL CLASS, cross-conditioned. Used to make Phi+ vs Psi+, the SAME
      CZ produces them with split fidelity 1-F = (1 - exp(-(I11+I22 +/- 2 I12)/2))/2
      -- a FIRST-order inter-qubit-cross effect (sign of Re S_1_2) that the
      gate-averaged F_pro cancels and a single-qubit campaign cannot predict.
      Reported as concurrence via C = 2F - 1 (the manuscript footnote).

Validation gates (printed):
  * calculate_cz_fidelity(our I-matrix) vs the saved infs_opt/infs_known F_pro
    -> validates the overlap extraction;
  * lambda-formula reduces analytically to bell_infs (MC-validated in storage).

Usage:
    QNS2Q_REGIME=showcase PYTHONPATH=src ./venv/bin/python \
        scripts/showcase_cz_entanglement_panel.py [--out FILE] [--pdf FILE]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import jax.numpy as jnp

from qns2q.paths import project_root
from qns2q.control import idle as idmod
from qns2q.control import cz as czmod

FOLDER = "DraftRun_NoSPAM_showcase_cap"

# Z-eigenvalues on |00>,|01>,|10>,|11> for the three noise generators.
ZEIG = {1: np.array([+1., +1., -1., -1.]),   # Z1  (Z x I)
        2: np.array([+1., -1., +1., -1.]),   # Z2  (I x Z)
        3: np.array([+1., -1., -1., +1.])}   # Z12 (Z x Z)

# Ideal CZ output |chi> = exp(-i pi/4 Z1Z2)|++> in the |00>,|01>,|10>,|11> basis.
_CHI = 0.5 * np.array([np.exp(-1j * np.pi / 4), np.exp(+1j * np.pi / 4),
                       np.exp(+1j * np.pi / 4), np.exp(-1j * np.pi / 4)])


def cz_imatrix(cfg, pt1, pt2, T_seq, M, use_ideal):
    """Full complex 4x4 CZ overlap matrix via the folded evaluator that the CZ
    cost itself uses (control.cz._cost_folded), plus the two-body train pt12."""
    SMat = cfg.SMat_ideal if use_ideal else cfg.SMat
    w_grid = cfg.w_ideal if use_ideal else cfg.w
    RMat, dt, nbs = czmod.prepare_time_domain_overlap(SMat, w_grid, cfg.tau, T_seq, M)
    pt0 = jnp.array([0., T_seq])
    pt12 = czmod.make_tk12(jnp.asarray(pt1), jnp.asarray(pt2))
    pts = [pt0, jnp.asarray(pt1), jnp.asarray(pt2), pt12]
    I = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            I[i, j] = complex(czmod.evaluate_overlap_folded(pts[i], pts[j],
                                                            RMat[i, j], dt, nbs))
    return I, pt12


def fpro_cz(I, pt12, Jmax, M):
    """Average (process) CZ infidelity through the project's own evaluator."""
    diffs = np.diff(np.asarray(pt12))
    signs = (-1.0) ** np.arange(len(diffs))
    dc_12 = float(np.sum(diffs * signs))
    J = float(np.clip(np.pi * 0.25 / (M * dc_12), -Jmax, Jmax))
    fid = float(czmod.calculate_cz_fidelity(jnp.asarray(I), J, M, dc_12))
    return 1.0 - fid


def _coh_decay(I, a, b):
    """lambda_ab for the coherence between basis states a and b (real overlaps)."""
    n = {g: (ZEIG[g][a] - ZEIG[g][b]) / 2.0 for g in (1, 2, 3)}
    expo = 0.0
    for g1 in (1, 2, 3):
        for g2 in (1, 2, 3):
            expo += n[g1] * n[g2] * np.real(I[g1, g2])
    return np.exp(-0.5 * expo)


def generated_state(I):
    """Density matrix of the CZ-generated |chi> after the dephasing channel."""
    rho = np.outer(_CHI, _CHI.conj()).astype(complex)
    out = rho.copy()
    for a in range(4):
        for b in range(4):
            if a != b:
                out[a, b] *= _coh_decay(I, a, b)
    return out


def concurrence(rho):
    """Wootters concurrence of a 2-qubit density matrix."""
    Y = np.array([[0, 0, 0, -1], [0, 0, 1, 0],
                  [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=complex)  # sigma_y x sigma_y
    rho_t = Y @ rho.conj() @ Y
    ev = np.linalg.eigvals(rho @ rho_t)
    s = np.sort(np.sqrt(np.clip(np.real(ev), 0.0, None)))[::-1]
    return float(max(0.0, s[0] - s[1] - s[2] - s[3]))


def bell_gen_infs(I):
    """(1-F(Phi+), 1-F(Psi+), 1-F blind) for the CZ-generated Bell classes."""
    i11, i22, i12 = np.real(I[1, 1]), np.real(I[2, 2]), np.real(I[1, 2])
    inf_phi = 0.5 * (1 - np.exp(-0.5 * (i11 + i22 + 2 * i12)))
    inf_psi = 0.5 * (1 - np.exp(-0.5 * (i11 + i22 - 2 * i12)))
    inf_blind = 0.5 * (1 - np.exp(-0.5 * (i11 + i22)))   # single-qubit-blind: i12 -> 0
    return inf_phi, inf_psi, inf_blind


def _seq_pair(obj):
    """Normalize a saved sequence entry to (pt1, pt2) arrays."""
    pt1, pt2 = obj[0], obj[1]
    return np.asarray(pt1, dtype=float), np.asarray(pt2, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None, help="npz output")
    ap.add_argument('--pdf', default=None, help="figure output")
    args = ap.parse_args()

    cfg = idmod.Config(fname=FOLDER, M=1, max_pulses=10**9, min_sep_factor=8.0)
    dpath = os.path.join(project_root(), FOLDER, "plotting_data",
                         "plotting_data_cz_v2_cap.npz")
    d = np.load(dpath, allow_pickle=True)
    taxis = np.asarray(d['taxis'], dtype=float)
    M = int(d['M'])
    Jmax = float(np.pi * 0.25 / float(d['min_gate_time']))
    seqs_opt = d['sequences_opt']
    seqs_known = d['sequences_known']
    infs_opt = np.asarray(d['infs_opt'], dtype=float)
    infs_known = np.asarray(d['infs_known'], dtype=float)
    infs_nopulse = np.asarray(d['infs_nopulse'], dtype=float)

    print(f"Jmax={Jmax:.5f}  M={M}  gate times={taxis}")
    print("-" * 116)
    rows = []
    for k, Tg in enumerate(taxis):
        T_seq = Tg / M
        pt_fid = jnp.array([0., T_seq])
        pt1_nt, pt2_nt = _seq_pair(seqs_opt[k])
        pt1_cd, pt2_cd = _seq_pair(seqs_known[k])

        rec = dict(Tg=Tg)
        for tag, pt1, pt2, fpro_saved in (
                ("fid", pt_fid, pt_fid, infs_nopulse[k]),
                ("cdd", pt1_cd, pt2_cd, infs_known[k]),
                ("nt",  pt1_nt, pt2_nt, infs_opt[k])):
            I, pt12 = cz_imatrix(cfg, pt1, pt2, T_seq, M, use_ideal=True)
            rho = generated_state(I)
            C = concurrence(rho)
            F_chi = float(np.real(_CHI.conj() @ rho @ _CHI))
            fpro = fpro_cz(I, pt12, Jmax, M)
            inf_phi, inf_psi, inf_blind = bell_gen_infs(I)
            rec.update({
                f"{tag}_conc": C, f"{tag}_concloss": 1 - C,
                f"{tag}_statinf": 1 - F_chi, f"{tag}_fpro": fpro,
                f"{tag}_fpro_saved": fpro_saved,
                f"{tag}_phi": inf_phi, f"{tag}_psi": inf_psi,
                f"{tag}_blind": inf_blind,
            })
        rows.append(rec)
        # validation + headline print
        print(f"Tg={Tg:7.0f} | F_pro NT  ours {rec['nt_fpro']:.3e} vs saved "
              f"{rec['nt_fpro_saved']:.3e}  (CDD ours {rec['cdd_fpro']:.3e} vs "
              f"saved {rec['cdd_fpro_saved']:.3e})")
        print(f"           | 1-Concurrence  FID {rec['fid_concloss']:.3e}  "
              f"CDD {rec['cdd_concloss']:.3e}  NT {rec['nt_concloss']:.3e}  "
              f"(NT better {rec['cdd_concloss']/rec['nt_concloss']:.1f}x)")
        print(f"           | NT Bell-gen  Phi+ {rec['nt_phi']:.3e}  Psi+ "
              f"{rec['nt_psi']:.3e}  blind {rec['nt_blind']:.3e}  "
              f"split {max(rec['nt_phi'],rec['nt_psi'])/min(rec['nt_phi'],rec['nt_psi']):.2f}x")
        print("-" * 116)

    out = args.out or os.path.join(project_root(), FOLDER, "cz_entanglement_panel.npz")
    keys = rows[0].keys()
    np.savez(out, **{k: np.array([r[k] for r in rows]) for k in keys})
    print(f"Saved {out}")

    _plot(rows, taxis, args.pdf)


def _plot(rows, taxis, pdf_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def col(key):
        return np.array([r[key] for r in rows])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    # (a) generated-pair quality: 1 - concurrence, FID / CDD / NT
    axA.loglog(taxis, col("fid_concloss"), 's--', color="0.6", label="FID (bare)")
    axA.loglog(taxis, col("cdd_concloss"), 'o-', color="#c44", label="best CDD")
    axA.loglog(taxis, col("nt_concloss"),  'D-', color="#268", label="best NT")
    axA.set_xlabel(r"gate time $T_G\,/\,\tau$")
    axA.set_ylabel(r"$1-\mathcal{C}$  (generated $|\chi\rangle=$CZ$|{+}{+}\rangle$)")
    axA.set_title("(a) entanglement the CZ generates")
    axA.grid(True, which="both", alpha=0.25)
    axA.legend(frameon=False, fontsize=8)

    # (b) which Bell class, cross-conditioned (NT sequence). The split is a
    # modest first-order cross effect, so show it as a deviation from the
    # single-qubit-blind prediction (= what F_pro / a single-qubit campaign
    # sees), which is otherwise invisible on a shared log axis.
    phi_rel = col("nt_phi") / col("nt_blind")
    psi_rel = col("nt_psi") / col("nt_blind")
    axB.axhline(1.0, color="0.5", ls=":", lw=1.0)
    axB.fill_between(taxis, phi_rel, psi_rel, color="0.88", zorder=0)
    axB.plot(taxis, phi_rel, 'o-', color="#c44",
             label=r"make $\Phi^+$ (collective, $+2\,\mathrm{Re}\,S_{1,2}$)")
    axB.plot(taxis, psi_rel, 'D-', color="#268",
             label=r"make $\Psi^+$ (differential, $-2\,\mathrm{Re}\,S_{1,2}$)")
    axB.set_xscale("log")
    axB.set_xlabel(r"gate time $T_G\,/\,\tau$")
    axB.set_ylabel(r"gen. infidelity $\div$ single-qubit-blind")
    axB.set_title(r"(b) which Bell class, set by sign of $\mathrm{Re}\,S_{1,2}$")
    axB.grid(True, which="both", alpha=0.25)
    axB.annotate("single-qubit QNS / $F_{\\mathrm{pro}}$ blind to the split",
                 xy=(taxis[0], 1.0), xytext=(0, 22), textcoords="offset points",
                 fontsize=7.5, color="0.4")
    axB.legend(frameon=False, fontsize=8, loc="center right")

    from matplotlib.ticker import ScalarFormatter, NullLocator
    for ax in (axA, axB):
        ax.set_xticks(taxis)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())

    fig.tight_layout()
    out = pdf_path or os.path.join(project_root(), FOLDER, "cz_entanglement_panel.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
