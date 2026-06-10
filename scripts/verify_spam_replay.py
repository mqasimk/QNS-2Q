"""Equivalence check for the record/replay SPAM pipeline (run on demand, ~5 min GPU).

Runs legacy raw + mitigated arms (reduced config, strong SPAM), records a
reference-arm phase dataset, replays both arms from it, and compares results:
means must match to float precision (the replay applies the identical diagonal
propagators); bootstrap-based errors may drift ~10% (np.random stream position
differs). Re-run after any change to trajectories/observables/experiments.

    QNS2Q_REGIME=featured python scripts/verify_spam_replay.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from run_spam_experiments import build_config
from qns2q.characterize.experiments import main
from qns2q.paths import project_root


def run(protocol, fname, strong=False, **mainkw):
    cfg = build_config(protocol, reduced=True, strong=strong)
    cfg.fname = fname
    main(cfg, **mainkw)
    return np.load(os.path.join(project_root(), fname, 'results.npz'))


def compare(tag, legacy, replay):
    worst_mean, worst_err = 0.0, 0.0
    for k in legacy.files:
        a, b = np.asarray(legacy[k]), np.asarray(replay[k])
        assert a.shape == b.shape, f"{tag} {k}: shape {a.shape} vs {b.shape}"
        rel = float(np.max(np.abs(a - b)/np.maximum(np.abs(a), 1e-12)))
        if k.endswith('_err'):
            worst_err = max(worst_err, rel)
        else:
            worst_mean = max(worst_mean, rel)
    print(f"  {tag}: worst rel diff -- means {worst_mean:.2e}, errs {worst_err:.2e}")
    return worst_mean, worst_err


if __name__ == "__main__":
    phases = os.path.join(project_root(), 'ReplayCheck_ref', 'phases.npz')
    legacy_raw = run('raw', 'ReplayCheck_raw_legacy', strong=True)
    legacy_mit = run('mitigated', 'ReplayCheck_mit_legacy', strong=True)
    run('reference', 'ReplayCheck_ref', record_to=phases)
    replay_raw = run('raw', 'ReplayCheck_raw_replay', strong=True, replay_from=phases)
    replay_mit = run('mitigated', 'ReplayCheck_mit_replay', strong=True, replay_from=phases)
    m1, e1 = compare("raw   legacy vs replay", legacy_raw, replay_raw)
    m2, e2 = compare("mitig legacy vs replay", legacy_mit, replay_mit)
    ok = m1 < 1e-8 and m2 < 1e-8 and e1 < 0.25 and e2 < 0.25
    print("EQUIVALENCE:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)
