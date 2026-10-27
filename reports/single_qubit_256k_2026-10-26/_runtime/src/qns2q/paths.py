"""Central resolver for noise-regime selection and run-folder paths.

This module carries no physics of its own, but it is the switchboard that decides
*which* physics the rest of the pipeline sees, and *where* every stage's inputs and
outputs live on disk. QNS-2Q ships three synthetic two-qubit dephasing-noise power
spectral density (PSD) models -- ``bland`` (smooth, monotonically decaying spectrum,
"Class M" in NOISE_MODEL_SPEC.md), ``featured`` (bland + narrow resonance lines,
e.g. GaAs nuclear-difference peaks, "Class F"), and ``showcase`` (the hand-tuned
"engineered trap" spectrum used to generate the eight published-paper figures) --
whose actual formulas live in ``qns2q.noise.spectra``. Exactly one of the three is
"live" for a given Python process, selected by a single environment variable,
``QNS2Q_REGIME``, read here and nowhere else. This is what lets a student
regenerate the *entire* pipeline (QNS experiment simulation -> spectral
reconstruction -> CZ/idle gate optimization -> figures) against a different noise
model just by re-exporting one shell variable before running a script -- no source
edits, and no manually renaming/hunting for output folders (see CLAUDE.md,
"Noise regime").

Pipeline role: this file sits underneath BOTH pipeline arms -- ``characterize/``
(QNS experiments + spectral reconstruction) and ``control/`` (CZ/idle gate
optimization) -- as well as ``noise/spectra.py`` (the actual PSD formulas) and most
of ``scripts/*.py``. Concretely, ``noise/spectra.py`` calls ``current_regime()``
once at *import time* (module-level code, not inside a function) to bake the
matching PSD constants into its module attributes. Practical consequence for
anyone experimenting interactively: setting ``os.environ["QNS2Q_REGIME"]`` *after*
``qns2q.noise.spectra`` has already been imported in that Python process has no
effect -- the regime is latched in at first import, so change the environment
variable (e.g. in the shell, before launching Python) rather than mutating it at
runtime.

Inputs/outputs: this module reads only the ``QNS2Q_REGIME`` environment variable
(plus, to find the repo root, this file's own on-disk location) and writes nothing
itself -- every function below returns a plain string or filesystem path. Other
modules use ``run_folder()``/``run_path()`` to find the run folder a previous
pipeline stage already wrote (``{results,params,specs}.npz``, etc.) and to decide
where the *current* stage should write its own output.

Because dozens of other files do ``from qns2q.paths import run_folder, ...`` (see
``characterize/*.py``, ``control/*.py``, ``noise/spectra.py``, ``viz/cz_pulse_plot.py``,
most ``scripts/run_*.py``, and several ``tests/``), none of the public names below
(``current_regime``, ``run_folder``, ``project_root``, ``run_path``,
``VALID_REGIMES``, ``DEFAULT_REGIME``) may be renamed, removed, or have their call
signature changed without breaking those call sites.

The active regime is read from the ``QNS2Q_REGIME`` environment variable so a whole
pipeline can be switched without editing any source:

    QNS2Q_REGIME=bland    PYTHONPATH=src python -m qns2q.control.idle
    QNS2Q_REGIME=featured PYTHONPATH=src python -m qns2q.control.idle   # default

Run folders follow the canonical scheme
``DraftRun_{NoSPAM|SPAM}_{bland|featured|showcase}`` (optionally with a SPAM
``protocol`` suffix, e.g. ``..._robust``); see ``run_folder()`` below.
"""

import os

# The three noise-model regimes this codebase supports (see the module docstring
# above and qns2q.noise.spectra for what each one actually looks like as a PSD).
# "featured" is the default because it is the regime most pipeline stages were
# originally exercised against (see CLAUDE.md, "Noise regime": default 'featured').
VALID_REGIMES = ("bland", "featured", "showcase")
DEFAULT_REGIME = "featured"


def current_regime():
    """Return the active noise regime from ``QNS2Q_REGIME`` (default 'featured').

    Reading ``os.environ`` with ``.get(..., DEFAULT_REGIME)`` is how a value gets
    into a Python program from the shell without any command-line flag: whoever
    launched the process can set ``QNS2Q_REGIME=bland`` beforehand and every
    downstream call to this function (and everything that depends on it, e.g.
    ``qns2q.noise.spectra``) picks it up automatically. The ``.strip().lower()``
    normalizes stray whitespace/capitalization (e.g. an accidentally-quoted
    ``" Bland"`` from a shell script) so it still matches. The explicit
    ``ValueError`` below is a deliberate fail-fast check: a typo'd regime name
    (e.g. ``QNS2Q_REGIME=featur``) would otherwise silently fall through to
    whatever the calling code does with an unrecognized string, likely surfacing
    as a confusing crash or silently-wrong physics deep inside the pipeline rather
    than an immediate, readable error at start-up.
    """
    regime = os.environ.get("QNS2Q_REGIME", DEFAULT_REGIME).strip().lower()
    if regime not in VALID_REGIMES:
        raise ValueError(
            f"QNS2Q_REGIME must be one of {VALID_REGIMES}, got {regime!r}"
        )
    return regime


def run_folder(regime=None, spam=False, protocol=None):
    """Canonical run-folder *name*, e.g. ``DraftRun_NoSPAM_bland``.

    This is a plain string (folder name only, not an absolute path -- see
    ``run_path()`` below for that). ``regime`` defaults to whatever
    ``current_regime()`` currently resolves to, so most callers omit it and just
    let the environment variable decide. ``spam`` picks between the two top-level
    pipeline variants: the default no-SPAM runs (state-preparation-and-measurement
    errors turned off) versus the SPAM-error runs used to test the mitigation
    protocols (see CLAUDE.md, "SPAM pipeline"). ``protocol`` (e.g. 'raw' |
    'mitigated' | 'robust') appends a suffix for the SPAM-pipeline runs:
    ``DraftRun_SPAM_featured_robust``; it names which SPAM-error-handling strategy
    produced that particular run's data. Default None keeps the canonical
    two-part names unchanged.
    """
    regime = regime or current_regime()
    tag = "SPAM" if spam else "NoSPAM"
    base = f"DraftRun_{tag}_{regime}"
    return base if protocol is None else f"{base}_{protocol}"


def project_root():
    """Repo root, resolved from this file's location (src/qns2q/paths.py -> repo root).

    Climbing three directories up from this file (``qns2q/`` -> ``src/`` -> repo
    root) via ``os.path.abspath(__file__)`` -- rather than, say, assuming the repo
    root is the current working directory -- is what makes every pipeline stage
    "CWD-independent" (CLAUDE.md, "Running the Pipeline"): a script can be
    launched from any directory and its data paths still resolve correctly,
    because they are anchored to where this source file lives on disk, not to
    wherever the shell happened to ``cd`` to before invoking Python.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_path(regime=None, spam=False, protocol=None):
    """Absolute path to the canonical run folder for the (regime, spam) selection.

    Just ``project_root()`` joined with ``run_folder(...)`` -- a convenience so
    callers that need to actually open files (e.g. ``np.load(...)``) get a full,
    unambiguous path rather than having to remember to join the two themselves.
    """
    return os.path.join(project_root(), run_folder(regime, spam, protocol))
