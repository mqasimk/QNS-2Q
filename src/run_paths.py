"""Central resolver for noise-regime selection and run-folder paths.

Single source of truth for *which* noise model and *which* output folder a run
uses, replacing the per-script hardcoded ``DraftRun_NoSPAM_*`` strings (several of
which were stale or pointed at empty/nonexistent folders).

The active regime is read from the ``QNS2Q_REGIME`` environment variable so a whole
pipeline can be switched without editing any source:

    QNS2Q_REGIME=bland    python id_optimize.py
    QNS2Q_REGIME=featured python id_optimize.py   # default

Run folders follow the canonical scheme ``DraftRun_{NoSPAM|SPAM}_{bland|featured}``.
"""

import os

VALID_REGIMES = ("bland", "featured")
DEFAULT_REGIME = "featured"


def current_regime():
    """Return the active noise regime from ``QNS2Q_REGIME`` (default 'featured')."""
    regime = os.environ.get("QNS2Q_REGIME", DEFAULT_REGIME).strip().lower()
    if regime not in VALID_REGIMES:
        raise ValueError(
            f"QNS2Q_REGIME must be one of {VALID_REGIMES}, got {regime!r}"
        )
    return regime


def run_folder(regime=None, spam=False):
    """Canonical run-folder *name*, e.g. ``DraftRun_NoSPAM_bland``."""
    regime = regime or current_regime()
    tag = "SPAM" if spam else "NoSPAM"
    return f"DraftRun_{tag}_{regime}"


def project_root():
    """Repo root (parent of ``src/``), resolved from this file's location."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_path(regime=None, spam=False):
    """Absolute path to the canonical run folder for the (regime, spam) selection."""
    return os.path.join(project_root(), run_folder(regime, spam))
