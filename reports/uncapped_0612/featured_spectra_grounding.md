# Featured spectra in spin qubits: measurement grounding (2026-06-12)

Purpose: ground the paper's "featured" noise class in published measurements, and
identify platforms where **featured in-band spectra**, **fast pulses (small τ)**, and
a **high-fidelity operating regime** coexist. Compiled for the 6/12 meeting
(Addendum thread); lean follow-up to the killed deep-research workflow.

## Per-platform inventory of measured spectral structure

| Platform | Measured feature (in/near band) | Numbers | Pulse speed (τ anchor) | Fidelity regime | Source |
|---|---|---|---|---|---|
| **GaAs S-T₀** | Nuclear Larmor-**difference** triplet (⁶⁹Ga/⁷¹Ga/⁷⁵As); echo collapse-revival | ≈0.83/0.87/1.70 MHz at 300 mT, positions ∝ B (0.2–1 T); strong enough to dominate echo decay | Exchange pulses ~ns (√SWAP ≈ 180 ps demonstrated, Petta 2005); **τ ≈ 1 ns credible** | 99.5% closed-loop 1Q (Cerfontaine/Bluhm 2020) | Malinowski 2017 Nat. Nanotech. (current model anchor); Bluhm 2011 Nat. Phys. (revivals); Dial 2013 PRL (exchange noise f^−0.7) |
| **Ge hole spins (planar)** | **⁷³Ge nuclear Larmor line measured directly in CPMG noise spectrum**, on top of 1/f charge noise | γ(⁷³Ge) = 1.485 MHz/T → line at 30 kHz @ 20 mT, **B-tunable to ~1.5 MHz @ 1 T**; S₀,HF = 2.5×10⁶ Hz²/Hz; width 9–17 kHz; charge noise 1.9 µeV/√Hz @ 1 Hz, S_E ∝ B² | Rabi 20–150 MHz electrically tunable (>100 MHz demonstrated) → **π ≈ 3–25 ns, τ ≈ 5 ns credible** (sweet-spot paper ran 1 MHz Rabi at 20 mT — B trades speed vs line position vs charge coupling) | 99.94% 1Q at sweet spot; T₂* = 17.6 µs, T₂^DD = 1.3 ms | Hendrickx et al. Nat. Mater. 2024 (sweet spot + line); Hendrickx et al. Nature 2020 (fast 2Q logic, >100 MHz Rabi) |
| **²⁸Si/SiGe exchange-only (enriched!)** | **⁷³Ge Larmor + first-harmonic peaks + quadrupolar structure from the SiGe *barrier***, superimposed on ²⁹Si dipole-dipole 1/f — measured by exchange-echo noise spectroscopy **in isotopically-enriched wells** | Peaks at f_L(⁷³Ge) and 2f_L; B- and epitaxy- (well-width-) dependent; diminish at higher B | Exchange (voltage) pulses ~ns | HRL 6-qubit/2Q EO fidelities ~99.9% era | HRL, arXiv:2009.08079 (PRApplied); arXiv:2208.11784 (full-permutation DD) |
| **Si/SiGe S-T₀ (exchange channel)** | Charge-noise spectrum over ~12 decades with **changing spectral exponent (knees)**; individual TLF Lorentzians resolvable and even *stabilizable* | Colored across full range; sensor-dot transport agrees with qubit DD spectroscopy (512π CPMG) | Exchange pulses ~ns | — (spectroscopy device) | Connors et al. Nat. Commun. 13, 940 (2022); arXiv:2407.05439 (single TLF) |
| **Natural Si/SiGe (single spin)** | **No intrinsic in-band peaks**: smooth f^−1 → f^−1.4 below 0.3 Hz (hyperfine), f^−0.4…−0.7 charge above ~10 kHz; one 3.6 kHz peak = electronics artifact | (our current anchor set) | EDSR π ≈ 50–200 ns (τ = 25 ns is already optimistic here) | 99.9% era (Yoneda 2018) | Rojas-Arias npj QI 2025; Yoneda Nat. Phys. 2023 |
| **²⁸Si purified (single spin)** | Smooth; TLF knee at 0.01 Hz (far below band) | T₂* = 6.1–6.9 µs | EDSR, as above | 99.9%+ | Rojas-Arias PRApplied 2023; Struck npj QI 2020 |

## The three load-bearing conclusions

1. **"Featured" is experimentally real, on three independent mechanisms**: nuclear
   Larmor-difference lines (GaAs, our current anchor), direct nuclear Larmor lines
   (⁷³Ge in Ge holes — *measured in the CPMG band*), and charge TLF knees /
   exponent changes (Si/SiGe exchange channel). The synthetic-spectra criticism
   does not apply to the featured class per se — only to amplitudes/positions not
   tied to a device.

2. **Featured survives purification — via the barrier and via charge.** The HRL
   result is decisive: isotopically-enriched ²⁸Si wells still show ⁷³Ge Larmor and
   quadrupolar peaks because the Ge sits in the SiGe barrier, which no Si
   purification touches. Charge TLF structure (Connors) is isotope-independent.
   So "high-fidelity material ⇒ bland spectra" is FALSE as a general claim; the
   correct statement is that the *hyperfine 1/f background* purifies away while
   line/TLF structure can persist — which *raises* feature contrast over the
   background it sits on, up to the charge floor.

3. **The platform that has all three at once is Ge holes** (with GaAs S-T₀ as the
   harsh-regime featured anchor we already use): measured in-band ⁷³Ge line with
   B-tunable position (30 kHz–1.5 MHz over 20 mT–1 T), Rabi 20–150 MHz (τ ≈ 5 ns
   credible), 99.94% demonstrated 1Q fidelity, T₂* = 17.6 µs. One honest tension
   to state: the demonstrated sweet spot sits at low B (slow Rabi, line at 30 kHz);
   fast-Rabi operation at higher B moves the line in-band but raises charge
   coupling (S_E ∝ B²). B is the knob that trades τ, line position, and charge
   noise against each other — a *design surface*, which is exactly the story the
   paper's τ-anchored, dimensionless framework is built to explore.

## Mapping to the model's dimensionless axes

Model regime ≡ (T₂*/τ, ω_feature·τ). Current featured class: T₂*/τ = 800,
lines at ω·τ = 0.26/0.27/0.53.

- GaAs S-T₀ @ τ = 1 ns: T₂*/τ ~ 100–2000 (DNP/notch dependent); Malinowski lines
  at ω·τ ≈ 0.005–0.011 (Tg ≳ 600τ sees them) — *the current dimensionless model
  is closest to this platform*.
- Ge holes @ τ = 5 ns, B = 0.5–1 T: T₂*/τ ~ 700–3500 (17.6 µs is low-B; expect
  lower at high B); ⁷³Ge line at ω·τ ≈ 0.023–0.047 → in-band for Tg ≳ 140–270τ.
- ²⁸Si/SiGe EO @ τ = 5 ns (exchange pulses): T₂*/τ ~ 1200+; ⁷³Ge barrier peaks
  in-band; charge knees from the Connors-type exchange spectrum.

## Recommended next step (post-meeting)

Build the high-fidelity featured regime as **"Class P+" = purified/quiet background
(T₂*/τ ≈ 3000–5000) + measured surviving structure** (⁷³Ge-type line with
B-scaled position + Connors-type TLF knee + Dial/symmetric-point S₁₂₁₂ as
Lorenza's knob), anchored per the table above, and re-run the uncapped pipeline.
This is the regime where bare ~1e-3 / NT ~1e-4 *and* a featured margin are
simultaneously honest.
