"""Two-qubit Quantum Noise Spectroscopy (QNS): experiment simulation, spectral
reconstruction, and noise-tailored gate optimization for the companion paper
"Noise-tailored two-qubit gates from spectral reconstruction".

Subpackages, in pipeline order (CLAUDE.md has the full picture, DEPENDENCY_MAP.md
the exact import graph):

    noise         ground-truth noise power spectral densities (the model)
    model         simulation engine: noise trajectories, Hamiltonian, propagators
    characterize  Stages 1-2: QNS experiments -> spectral reconstruction
    control       Stage 3: CZ / idling gate optimization against those spectra
    viz           Stage 4 plotting helpers

``paths`` (this package's top level) is the regime/run-folder resolver every
other module goes through.
"""
