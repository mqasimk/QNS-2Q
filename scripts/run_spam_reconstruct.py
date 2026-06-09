"""Stage 2 of the SPAM pipeline: reconstruct spectra from a SPAM-protocol run.

Usage (from the repo root, regime via QNS2Q_REGIME):

    python scripts/run_spam_reconstruct.py [raw|mitigated|robust]

Reads ``DraftRun_SPAM_<regime>_<protocol>/{results,params}.npz`` and writes
``specs.npz`` plus the reconstruction figures into the same folder. The
SPAM-robust branch (M-regression of the swept harmonic observables, S_l_12
flagged inaccessible) is selected automatically from the run's params.npz.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from qns2q.characterize.reconstruct import main
from qns2q.paths import run_folder

if __name__ == "__main__":
    protocol = sys.argv[1] if len(sys.argv) > 1 else 'mitigated'
    if protocol not in ('raw', 'mitigated', 'robust', 'reference'):
        raise SystemExit(f"Unknown protocol {protocol!r}; "
                         "expected raw|mitigated|robust|reference")
    main(data_folder=run_folder(spam=True, protocol=protocol))
