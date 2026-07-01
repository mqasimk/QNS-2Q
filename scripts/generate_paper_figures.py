"""Single entry point: generate every figure used by the companion paper.

The companion manuscript ("Noise-tailored two-qubit gates from spectral
reconstruction", Khan/Norris/Viola, ``~/Noise_Tailored_2Q_Gates/main_v10.tex``)
uses eight figures, all from the ``showcase`` noise regime. FIGURE_PROVENANCE.md
is the authoritative figure -> data -> command map; this script mechanizes it
into one command instead of several, so a new student doesn't have to piece
the pipeline together by hand. If you only want to regenerate ONE figure (or
the exact command diverges from what's below after a future change), go read
FIGURE_PROVENANCE.md directly -- it is the source of truth, this script is a
convenience wrapper around it.

Two modes:

  (default) assemble-only    Rebuild all 8 figures from the .npz summary data
                              already committed to the repo (DraftRun_*/ run
                              folders). Fast: a few minutes, no GPU-heavy
                              simulation. This is what most students should
                              run, and it's a good way to sanity-check that
                              your environment (pyproject.toml / requirements.txt)
                              reproduces the published figures byte-for-byte
                              before you touch any physics.

  --full-regen                Also re-derive every underlying .npz from
                              scratch first: Stage 1 capture (the 256k-shot
                              NoSPAM run), all four SPAM arms, both gate
                              optimizers (main run + SPAM arms + the
                              knowledge-ladder ablation rungs), the margin
                              bands, the storage panel, and the design-number
                              harvest -- THEN assemble the figures. This wraps
                              scripts/run_capture_arm.py followed by
                              scripts/run_carrier_battery_0616.sh; read that
                              shell script's header for the per-stage
                              parallelism knobs (GATE_JOBS/SPAM_JOBS/AUX_JOBS)
                              and expected GPU-memory sharing. Expensive: on a
                              single consumer GPU this is a multi-hour run, not
                              a "grab coffee" run -- it re-derives the paper's
                              entire showcase dataset, not just one figure.

Usage (from the repo root, with the venv active):

    python scripts/generate_paper_figures.py                # assemble only
    python scripts/generate_paper_figures.py --full-regen    # regenerate everything first
    python scripts/generate_paper_figures.py --dry-run       # print the commands, run nothing
"""
import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP_FOLDER = "DraftRun_NoSPAM_showcase_cap"
SHOWCASE_FIGS_DIR = "reports/showcase_0613/figs"

# Each step: (human-readable label, argv, extra env vars beyond the QNS2Q_REGIME
# and PYTHONPATH set for every step below).
FULL_REGEN_PREFIX_STEPS = [
    ("Stage 1: NoSPAM capture arm (256k shots per estimator)",
     [sys.executable, "scripts/run_capture_arm.py"], {}),
    ("Stages 1-3 battery: SPAM arms, knowledge-ladder rungs, both gate "
     "optimizers, margin bands, storage panel, design-number harvest "
     "(the long pole -- see scripts/run_carrier_battery_0616.sh for details)",
     ["bash", "scripts/run_carrier_battery_0616.sh"], {}),
]

ASSEMBLE_STEPS = [
    ("Six showcase panels (model, reconstruction, SPAM arms, design, "
     "storage, gates)",
     [sys.executable, "scripts/report_showcase_figs.py"],
     {"SHOWCASE_FIGS_DIR": SHOWCASE_FIGS_DIR}),
    # This step synthesizes noise trajectories out to a fairly large M and is
    # the most GPU-memory-hungry of the three (see run_carrier_battery_0616.sh's
    # comment on this exact script): give it a larger fraction of GPU memory
    # than JAX's default, and don't run anything else on the GPU at the same time.
    ("C_1_0_MT_vs_M.pdf (standalone single-qubit SPAM-robustness figure)",
     [sys.executable, "scripts/run_single_qubit.py"],
     {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9"}),
    ("showcase_pulse_sequences.pdf (best-known vs best-NT CZ pulse comparison)",
     [sys.executable, "scripts/run_cz_pulse_plot.py",
      "--folder", CAP_FOLDER, "--tag", "_cap"], {}),
]


def run_step(label, argv, extra_env, dry_run):
    """Run one step; return True on success, False on failure. Never raises
    (a subprocess failure is reported and swallowed) so the caller can decide
    per-step whether a failure should stop the batch or just get logged and
    skipped -- see the two call sites in main() for why that differs between
    the full-regen prefix (each step's output feeds the next) and the
    assemble steps (each reads independent, already-on-disk data, so one
    failing -- e.g. a GPU too small for run_single_qubit.py's memory needs --
    shouldn't cost you the other seven figures)."""
    env = dict(os.environ)
    env["QNS2Q_REGIME"] = "showcase"
    src_path = os.path.join(REPO_ROOT, "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    env.update(extra_env)

    env_prefix = "".join(f"{k}={v} " for k, v in extra_env.items())
    print(f"\n=== {label} ===")
    print(f"    $ QNS2Q_REGIME=showcase {env_prefix}{' '.join(argv)}")
    if dry_run:
        return True
    t0 = time.time()
    result = subprocess.run(argv, cwd=REPO_ROOT, env=env)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"    ({elapsed:.0f}s)")
        return True
    print(f"    FAILED (exit {result.returncode}, {elapsed:.0f}s)")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--full-regen", action="store_true",
        help="Re-derive every showcase .npz from scratch (Stage 1 capture + "
             "the full run_carrier_battery_0616.sh battery) before assembling "
             "figures. Expensive: hours on a single GPU. Without this flag, "
             "figures are assembled from the .npz data already committed to "
             "the repo (minutes).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would run, without running them.")
    args = parser.parse_args()

    if args.full_regen and not args.dry_run:
        print("--full-regen overwrites the committed showcase .npz run-folder "
              "data and can take several hours on a single GPU.")
        print("Ctrl-C now to abort; starting in 5s...")
        time.sleep(5)

    if args.full_regen:
        # Sequential dependency (capture -> battery): stop immediately if a
        # prefix step fails, since every later step reads its output.
        for label, argv, extra_env in FULL_REGEN_PREFIX_STEPS:
            if not run_step(label, argv, extra_env, args.dry_run):
                sys.exit(f"\n{label!r} failed -- aborting before the assemble "
                         f"steps, since they depend on its output.")

    # Independent steps: run all of them and report every failure at the end,
    # rather than letting one figure's failure hide the others.
    failed = [label for label, argv, extra_env in ASSEMBLE_STEPS
              if not run_step(label, argv, extra_env, args.dry_run)]

    if args.dry_run:
        print("\nDry run complete -- nothing was executed.")
        return
    if failed:
        sys.exit("\nFailed: " + "; ".join(failed) +
                  "\nThe other figures above still succeeded. Re-run just the "
                  "failed one(s) once you've addressed the cause (e.g. a "
                  "too-small GPU for run_single_qubit.py -- see its docstring).")
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
