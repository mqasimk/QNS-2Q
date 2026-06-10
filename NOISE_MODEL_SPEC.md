# Noise-model spec: experimentally-anchored two-qubit dephasing spectra

**Status: DRAFT — awaiting QK review, then Lorenza sign-off (one-pager to be distilled from this). No code implements this yet.**

Replaces both synthetic noise classes in `src/qns2q/noise/spectra.py` with spectra
composed of experimentally measured ingredients, per the 2026-06-10 decisions:

1. **Rebuild both classes** (`bland` → Class M, `featured` → Class F) for a fully
   provenanced figure set (paper tracker: `NOISE-MODEL-0610`, unblocks `CA-REPRO-NUMBERS`).
2. **S₁₂₁₂ calibrated at the gate operating point** (Dial-2013-style exchange noise),
   not at the symmetric-point floor measured by Yoneda 2023.

Every parameter below is tagged: **[M]** measured (with source), **[E]** extrapolated
from measurement (assumption to state in the paper), **[C]** modeling choice (free knob,
flagged for Lorenza).

---

## 1. Anchor dataset

| Source | What it pins |
|---|---|
| Yoneda et al., *Nat. Phys.* **19**, 1793 (2023); arXiv:2208.14150 — natural Si/SiGe pair, 100 nm apart, J = 1.1 MHz at symmetric point, \|Δ\| = 620 MHz ≫ J (Ising-ZZ regime, ν_Q^σ = ν_Q ± J/2) | The six-spectrum structure itself; auto-PSDs S_A: a = 1100 kHz²/Hz·(f/1Hz)^−1.21, S_B: 800·f^−1.14 + TLF Lorentzians (b = 105/129 kHz, τ₀ = 0.140/0.175 s); S_J: 0.36 kHz²/Hz·f^−1.37 (≈3×10⁻⁴ of S_A at 1 Hz); c_AB up to **0.70 ± 0.06** (in-phase ≳80 mHz, anti-phase ≲40 mHz); c_AJ ≈ **+0.8** (in-phase); c_BJ ≈ **0.3–0.6 anti-phase**; qubit B carries extra uncorrelated nuclear noise; two-local-electric-field susceptibility model reproduces all six spectra |
| Rojas-Arias et al., *npj QI* (2025); arXiv:2408.13707 — natural Si, 1 Hz–1 MHz over 3 devices | High-frequency auto-PSD shape: charge-noise dominated above ~10 kHz with exponents **f^−0.4 (qubit L) / f^−0.7 (qubit R)**; hyperfine-dominated f^−1 below ~0.3 Hz; T₂* = 1.03 µs, T₂^CPMG(64π) = 190 µs; **no nuclear Larmor peaks** in natural-Si CPMG band |
| Rojas-Arias et al., *PRApplied* **20**, 054024 (2023) — purified ²⁸Si (800 ppm) | Purified-device T₂* = **6.1/6.9 µs**; correlation ~0.1 at 200 nm; TLF Lorentzian (knee 0.01 Hz); spatial decay ∝ d⁻⁵ |
| Malinowski et al., *Nat. Nanotech.* **12**, 16 (2017); arXiv:1601.06677 — GaAs S-T₀ | The featured-class lines: narrow noise bands at the **differences of nuclear Larmor frequencies** of ⁶⁹Ga/⁷¹Ga/⁷⁵As — two nearly degenerate + one at twice their frequency, positions ∝ B (0.2–1 T explored); at 300 mT: ≈0.83/0.87/1.70 MHz; steep power-law background (model β = 3, T_SD = 600 µs, N = 7×10⁵, δB = 1.1 mT); notch-filtered T₂ = 0.87 ± 0.13 ms |
| Dial et al., *PRL* **110**, 146804 (2013) — GaAs S-T₀ exchange | Exchange/charge noise **at operating J**: S_ε(f) = 8×10⁻¹⁶ V²/Hz·(1 Hz/f)^0.7 over **50 kHz–1 MHz** (0.2 nV/√Hz at 1 MHz); echo decay exp[−(τ/T₂)^{β+1}], **β ≈ 0.7 at every ε**; Q_echo = J·T₂^echo/2π > 600 up to J ≈ 350 MHz; dJ/dε ∝ J (exponential J(ε)) ⇒ scale-invariant relative J-noise |
| Supporting: Tarucha group arXiv:2603.03051 (5-qubit array, 2026); Donnelly/Simmons arXiv:2405.03763 (correlation 0.5→0.1 over 75→300 nm) | Corroborate charge-noise correlation range & tunability |

Cross-platform convergence worth quoting in the paper: in-band charge-noise exponents
cluster at **γ ≈ 0.4–0.7** in both Si (npj 2025) and GaAs (Dial 2013).

---

## 2. Calibration philosophy

Work in dimensionless τ-units as now (`_TAU_SI = 25 ns` retained **[C]** — keeps the
legacy anchor; the π/2 times of 65.5/103.5 ns in Yoneda 2023 mean the instantaneous-pulse
idealization is a stated approximation either way).

- **Shapes, exponents, coherences, signs: matched to measurement** (the [M] entries).
- **Absolute amplitude scale: set by the T₂*/τ target**, not by unit-converting kHz²/Hz
  (avoids one-sided/two-sided and angular-frequency convention traps; conversions only
  appear in the paper as "for a representative τ = 25 ns this corresponds to …").
- **Headline calibration: purified-²⁸Si coherence, natural-Si correlation structure [C]:**
  target T₂*(1), T₂*(2) ≈ **260τ** (6.5 µs at the anchor, PRApplied purified devices;
  Yoneda 2018's 20 µs ≈ 800τ noted as the optimistic end). Bare gate at the 320τ sweet
  spot then sits at ≈1.2 T₂* — FID baseline fails without saturating, consistent with
  current regenerated dynamics (FID ≈ 0.15 at 320τ).
- Band check: comb harmonics ω̃_k = 2πk/160, k = 1…20 → ω̃ ∈ [0.039, 0.785]
  (0.25–5 MHz at the anchor). The npj/Dial measured window (10 kHz–1 MHz) covers the
  lower ~half of the band; the upper half extrapolates the same power law **[E]**.

---

## 3. Class M — monotonic (replaces `bland`)

Pure Tarucha-Si composition. All six spectra derive from the component construction in §5.

Self-spectra (electrical + local nuclear, no lines):

- `S_11(ω̃) = A_1·(ω̃² + ω̃_ir²)^(−γ₁/2) + N_1(ω̃)` with **γ₁ = 0.7 [M]** (npj qubit R)
- `S_22(ω̃) = A_2·(ω̃² + ω̃_ir²)^(−γ₂/2) + N_2(ω̃)` with **γ₂ = 0.4 [M]** (npj qubit L)
- `S_1212(ω̃) = A_12·(ω̃² + ω̃_ir²)^(−γ_J/2)` with **γ_J = 0.7 [M]** (Dial β; Yoneda S_J exponent 1.37 is the sub-Hz value — in-band we use the operating-point GaAs measurement per decision 2)

Components:

- **IR cutoff ω̃_ir [C, constrained]:** regularizes S(0) for trajectory synthesis and
  the FID-slope DC protocol. Start at ω̃_ir = 0.02 (half the first harmonic); final value
  fixed by the DC linear-regime gate (§6-ii) — same analysis that weakened the old bland
  model (n_shots floor + ≥95% linearity over the fit window).
- **A_1, A_2:** set by the T₂* = 260τ target per qubit. Qubit asymmetry (γ₁ ≠ γ₂,
  A_1 ≠ A_2) is itself a measured feature [M].
- **N_l(ω̃):** local (uncorrelated) nuclear component, smooth steeper power law,
  weighted heavier on qubit 2 [M — Yoneda's qubit B carried excess uncorrelated nuclear
  noise]. In Class M keep it small (~10–20% of in-band power [C]); it exists mainly so
  the coherences come out partial rather than perfect.
- **A_12 (decision 2, the gate-operating-point choice):** set so the in-band ratio
  S_1212/√(S_11·S_22) ≈ **0.1 [C, range 0.05–0.15]**. Rationale: Dial's scale-invariant
  relative exchange noise with Q_echo ~ 600 puts coupler noise 1–2 orders below qubit
  noise *at operating J* — far above Yoneda's symmetric-point floor (3×10⁻⁴) but still
  subdominant. Acceptance test below verifies the ZZ channel is relevant-but-not-dominant.
  This is the single most consequential free knob — flag prominently to Lorenza.
- Slow TLFs (b, τ₀ of Yoneda Fig. 1c) and the sub-Hz 1/f^1.2 content live ~8 decades
  below the band; they are subsumed in the quasi-static/IR plateau, not modeled as
  separate in-band terms [M→C, state in paper].

Cross-spectra: generated by the §5 construction with targets
**c₁₂(in-band) = +0.7, c₁,₁₂ = +0.8, c₂,₁₂ = −0.5** [M at ≲Hz; E carried to band — the
in-phase branch of the measured crossover, justified because the in-band noise is the
charge component (npj attribution) and charge is the correlated species].
Phases: real ± structure from susceptibilities [M] plus a small causal lag
**δt̃ ≈ 1–2 τ on e_B [C]** to produce Im{S_ab} at the 10–30% level in the upper band —
replaces the current unphysical 11.4τ pure delay; no cross-PSD data exists above ~Hz to
constrain this, state as model choice that exercises the Im reconstruction.

## 4. Class F — featured (replaces `featured`)

**Class M background (identical parameters) + the nuclear-difference triplet** [M, GaAs]:

- Three Gaussian lines on the *single-qubit* channels only (nuclear ⇒ local field noise;
  J-noise is electrical [M — Yoneda 2023's central conclusion], so **no lines on S_1212**).
- Placement at effective B = 600 mT (inside Malinowski's 0.2–1 T range) [C]:
  centers ω̃₀ ∈ {0.261, 0.273, 0.534} (1.66/1.74/3.40 MHz at the anchor), preserving the
  measured fingerprint — **two nearly degenerate lines + one at twice the frequency,
  positions ∝ B** [M]. The near-degenerate pair is closer than the comb spacing (0.012 <
  0.039) and will merge at reconstruction resolution — fine, and worth one caption sentence.
- Widths: σ̃ ≈ 0.02 (≈ half comb spacing) [C, constrained by reconstructability — must be
  sampled by ≥1–2 harmonics; physical lines are narrower, state as resolution-limited].
- Heights: ×10–30 over local background [C] with different weights on qubits 1 and 2
  (e.g. full triplet on qubit 2, reduced on qubit 1), tuned so DD-passband alignment is
  punished — the NT-advantage regime the paper's §V features.
- **Free physical signature:** because the lines are local and uncorrelated, the
  coherence c₁₂(ω̃) *dips at the line frequencies* — the cross-spectra are not a scaled
  copy of the self-spectra. This directly feeds Lorenza's "do I need all 6 spectra?"
  (the answer is visible in the model, not just asserted).

Honesty note for the paper: in Malinowski's system the narrow-band nuclear noise couples
to the S-T₀ splitting quadratically via transverse Overhauser components; we transpose
the measured *spectral shape class* (power-law background + narrow MHz lines, the kind
DD passbands collide with) into our linear-dephasing model, we do not claim to model
their microscopics. Cite alongside: Bluhm 2011 (echo revivals), Sung 2019 (engineered
peaked spectra as validation targets — already in aps_v2.bib).

## 5. Synthesis: two correlated local fields + local nuclear processes

Replaces the single-shared-draw construction in `trajectories.make_noise_mat_arr`
(which enforces coherence ≡ 1 — the thing we're removing). Mirrors the validated
Yoneda-2023 susceptibility model. Sign constraint that forces this structure: a single
fully-shared source gives sign(c₁₂)·sign(c₁,₁₂)·sign(c₂,₁₂) > 0, but the measured
pattern is (+, +, −) — impossible with one source, reproducible with two partially
correlated local fields [M].

```
e_A(t) = c_E·g₀(t)       + √(1−c_E²)·g_A(t)          # local field at qubit 1
e_B(t) = c_E·g₀(t − δt̃) + √(1−c_E²)·g_B(t)          # local field at qubit 2
ζ₁  = χ₁·e_A + h₁·n₁(t)
ζ₂  = χ₂·e_B + h₂·n₂(t)
ζ₁₂ = χ_J·(a·e_A − b·e_B)                            # J couples to the field *difference*
```

g₀, g_A, g_B: independent Gaussian draws with the electrical spectral shape;
n₁, n₂: independent draws with the nuclear shape (incl. Class-F lines).
J as difference-coupling is physical (exchange responds to the interdot potential
tilt/barrier) and yields the measured sign pattern: with c_E ≈ 0.75, a = 1, b ≈ 0.9 →
c₁,₁₂ > 0, c₂,₁₂ < 0, c₁₂ ≈ +0.7 ✓ (exact values solved numerically to hit the targets).
PSD-matrix positivity is automatic (sum of rank-1 components). All six analytic spectra
derive from the same components, so the public API (`S_11 … S_2_12`), the QNS forward
model, the reconstruction overlays, and the optimizer `SMat` stay mutually consistent.

Code-touch list: `noise/spectra.py` (component-based rewrite, both regime branches —
env keys `bland`/`featured` kept for plumbing stability); `trajectories.make_noise_mat_arr`
(4 base draws + mixing matrix; remove γ = T/14, T/28 lags); `gamma`-parameter plumbing in
configs/reconstruct overlays (cross-spectrum phase now comes from the model, not a
per-call lag argument); tests — keep `test_tau_invariance`, add (i) PSD-matrix min-eigenvalue
≥ 0 over the band, (ii) synthesized-trajectory cross-PSD vs analytic targets (coherence
and sign), (iii) T₂* regression per channel.

Known interaction to re-examine: the SPAM-**robust** D̂± estimator derivation assumes
S_1,12 = S_2,12 = 0. The current coherence-1 model violates this maximally; the new model
violates it at the measured level (c ≈ 0.8/0.5) — quantify the induced bias in the robust
arm when re-running the SPAM comparison (feeds the "why characterize SPAM" story:
robust ⇒ 4 spectra + an assumption; mitigated ⇒ all 6).

## 6. Acceptance gates (before any figure regeneration)

i.   T₂* per channel within target (1,2: 230–290τ; report FID + CDD₁ echo times too).
ii.  DC protocol linear-regime check passes at final (A_l, ω̃_ir) — redo the old
     bland-weakening analysis (signal ≥ 1.5× the n_shots floor at t_max, ≥95% linearity).
iii. Medium-stats QNS run (`--medium`): all six spectra + DC points reconstruct within
     error bars in both classes.
iv.  NT-vs-CDD probe at T_G ∈ {64τ, 320τ, 640τ} on Class F: NT ≤ CDD everywhere with a
     clear margin at the sweet spot (thesis survives); Class M sanity: CDD ~ NT (DD
     near-optimal for monotonic noise).
v.   ZZ-relevance check on the A_12 knob: re-run iv with S_1212 → 0 to confirm the ZZ
     channel matters at the chosen ratio (ties into the ask-2 subset experiment).

Then: full `CA-REPRO-NUMBERS` regeneration (reconstruction figs on Class M, gate figs on
both classes, SPAM arms re-run), captions + `tab:fidelity_summary` re-transcribed.

## 7. Paper touch-points

- `\cite{YY}` at `main_v9.tex:792` → {Yoneda2023, Malinowski2017} (+ optionally Bylander2011
  for cross-platform breadth).
- New bib entries: Yoneda2023 (Nat. Phys. 19, 1793), RojasArias2023 (PRApplied 20, 054024),
  RojasArias2025 (npj QI, DOI 10.1038/s41534-025-01150-6), Malinowski2017 (Nat. Nano 12, 16),
  Dial2013 (PRL 110, 146804); optional arXiv:2603.03051, arXiv:2405.03763.
- §V Methods numerics-setup paragraph: τ anchor, T₂*/τ, J_max·τ = 0.05 (→ 318 kHz;
  slower than Noiri-2022 fast gates by design — DD-protected slow-gate regime, say so),
  and the [M]/[E]/[C] provenance sentence. This is also the landing zone for Lorenza's
  ask 1 ("how much dephasing").
- Ask 2 gains the measured punchline (c up to 0.7/0.8 in a real device) + the line-dip
  coherence signature; ask 3 re-runs on the new model.

## 8. Knobs explicitly reserved for Lorenza's sign-off

1. A_12 ratio (S₁₂₁₂ strength at operating point): 0.05–0.15 proposed.
2. Line placement (B_eff = 600 mT proposed) and line-height factor (×10–30).
3. In-band coherence level carried from the ≲Hz measurements (0.7/0.8/−0.5 proposed).
4. The causal lag δt̃ (Im-part generator): 1–2τ proposed, 0 = all-real fallback.
5. Headline T₂* calibration: purified (260τ) vs natural (40τ) — purified proposed.
