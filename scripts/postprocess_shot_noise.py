"""Reprocess saved showcase phases with analytic finite-measurement errors.

Run with the repository venv. No simulation, random sampling, gate optimization
or figure generation is performed. Scientific reconstruction code is exported
from --pin; original data are read-only. All products go below --output.

    venv/bin/python scripts/postprocess_shot_noise.py \
        --output reports/decision30_analytic_shots_2026-10-25
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

# Keep deterministic postprocessing on CPU, with modest temporary arrays.
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['QNS2Q_REGIME'] = 'showcase'
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'src/qns2q/characterize/shot_noise.py'
SPEC = importlib.util.spec_from_file_location('_saved_shot_noise', MODULE_PATH)
STAT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAT
SPEC.loader.exec_module(STAT)
RUNS = ('DraftRun_NoSPAM_showcase_cap', 'DraftRun_SPAM_showcase_reference',
        'DraftRun_SPAM_showcase_raw', 'DraftRun_SPAM_showcase_mitigated')
SPECTRA = ('S11', 'S22', 'S1212', 'S12', 'S112', 'S212')


def json_default(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=json_default) + '\n')


def pinned_bytes(pin, path):
    return subprocess.check_output(['git', 'show', f'{pin}:{path}'], cwd=ROOT)


def load_npz(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def phase_moments(cache_run, params, output):
    path = ROOT / cache_run / 'phases.npz'
    target = output / f'{cache_run}_phase_moments.npz'
    plan = list(STAT.acquisition_plan(params))
    count = sum(a.width for a in plan)
    expected_lengths = np.concatenate([np.full(a.width, a.repetitions * int(params['t_grain']))
                                       for a in plan])
    with np.load(path, allow_pickle=False) as data:
        for key in ('T', 'M', 't_grain', 'truncate', 'w_grain', 'n_shots', 'midpoint', 'synth_wmax'):
            if not np.array_equal(data[f'meta_{key}'], params[key]):
                raise ValueError(f'Cache configuration mismatch: {cache_run}, {key}')
        if int(data['n_calls']) != count or not np.array_equal(data['t_lens'], expected_lengths):
            raise ValueError(f'Cache call-order mismatch: {cache_run}')
        if target.exists():
            with np.load(target, allow_pickle=False) as previous:
                if int(previous['source_bytes']) != path.stat().st_size:
                    raise ValueError('Cached phase moments have a different source size.')
                return previous['transfers'].copy()
        transfers = np.empty((count, 4, 4), complex)
        content_hash = hashlib.sha256()
        for i in range(count):
            phases = data[f'call_{i:04d}']
            if phases.shape != (int(params['n_shots']), 3):
                raise ValueError(f'Wrong phase shape in call {i}')
            content_hash.update(np.ascontiguousarray(phases).tobytes())
            transfers[i] = STAT.phase_transfer(phases)
            if (i + 1) % 100 == 0 or i + 1 == count:
                print(f'[saved phases] {cache_run}: {i+1}/{count}', flush=True)
        np.savez(target, transfers=transfers, t_lens=expected_lengths,
                 source_path=str(path), source_bytes=path.stat().st_size,
                 sha256_ordered_phase_arrays=content_hash.hexdigest())
    return transfers


def process_observables(run, params, original, transfers, shots, destination):
    plan = list(STAT.acquisition_plan(params))
    revised = {k: v.copy() for k, v in original.items()}
    results = []
    differences = []
    for acquisition in plan:
        subset = transfers[acquisition.first_call:acquisition.first_call+acquisition.width]
        result = STAT.analyze_acquisition(acquisition, subset, params, shots)
        old = float(original[acquisition.name][acquisition.index])
        difference = abs(result['value'] - old)
        if difference > 1e-10 * (1 + abs(old)):
            raise ValueError(f'Saved-value recovery failed: {run}, {acquisition}: {difference}')
        differences.append(difference)
        revised[acquisition.name + '_err'][acquisition.index] = result['stderr']
        results.append(result)
    revised.update(uncertainty_model=np.array('analytic_multinomial_independent_settings_delta2'),
                   physical_shots_per_setting=np.array(shots))
    np.savez(destination / 'results.npz', **revised)
    np.savez(destination / 'observable_statistics.npz',
             coefficient=np.array([a.name for a in plan]), index=np.array([a.index for a in plan]),
             dc=np.array([a.dc for a in plan]), first_cache_call=np.array([a.first_call for a in plan]),
             **{key: np.array([r[key] for r in results]) for key in results[0]})
    nonlinear = [dict(coefficient=a.name, index=a.index, dc=a.dc,
                      contrast_snr=r['minimum_contrast_snr'],
                      bias_in_stderr=r['second_order_bias_in_stderr'],
                      second_order_stderr_fraction=r['second_order_stderr_fraction'],
                      quadrature_variance_discrepancy=r['gaussian_quadrature_relative_variance_discrepancy'])
                 for a, r in zip(plan, results)
                 if (r['minimum_contrast_snr'] < 10 or r['second_order_bias_in_stderr'] > .1
                     or r['second_order_stderr_fraction'] > .1)]
    report = dict(coefficient_points=len(plan), physical_measurement_settings=4*len(plan),
                  total_physical_shots=4*len(plan)*shots,
                  maximum_saved_value_discrepancy=max(differences), nonlinear_flags=nonlinear,
                  estimator_approximation='second-order Taylor; exact multinomial central moments',
                  maximum_quadrature_variance_discrepancy=max(r['gaussian_quadrature_relative_variance_discrepancy'] for r in results),
                  original_means_preserved=True)
    write_json(destination / 'observable_summary.json', report)
    print(f'[observables] {run}: {len(plan)} coefficients; max recovery error '
          f'{max(differences):.3g}; {len(nonlinear)} nonlinear diagnostic flags', flush=True)
    return report


def prepare_runtime(pin, output):
    runtime = output / '_pinned_runtime'
    if not (runtime / 'src/qns2q/characterize/reconstruct.py').exists():
        raw = subprocess.check_output(['git', 'archive', pin, 'src',
                                       'tests/conftest.py', 'tests/test_spectral_inversion.py',
                                       'tests/test_systematics.py', 'tests/test_spam.py'], cwd=ROOT)
        runtime.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            archive.extractall(runtime, filter='data')
    # Guard against accidentally running a different version after resuming.
    for relative in ('characterize/reconstruct.py', 'characterize/inversion.py',
                     'characterize/systematics.py', 'model/trajectories.py', 'noise/spectra.py'):
        path = 'src/qns2q/' + relative
        if (runtime / path).read_bytes() != pinned_bytes(pin, path):
            raise ValueError(f'Pinned runtime drift: {path}')
    sys.path.insert(0, str(runtime / 'src'))
    import jax
    jax.config.update('jax_enable_x64', True)
    from qns2q.characterize import reconstruct, inversion, systematics
    from qns2q.model import trajectories

    def forbidden(*args, **kwargs):
        raise RuntimeError('Simulation/trajectory entry points are disabled during saved-data postprocessing.')

    for name in ('make_noise_mat_arr', 'make_noise_mat', 'make_noise_traj',
                 'make_channel_trajs', 'solver_prop', 'solver_phase_coeffs',
                 'solver_phase_coeffs_fast', 'fast_solver'):
        if hasattr(trajectories, name):
            setattr(trajectories, name, forbidden)

    # Filter vectors depend on timing, not spectra. Reuse identical pinned
    # quadratures across baseline/revised fits and SPAM arms without new noise.
    original_filter = systematics._ff_grid
    cache = {}
    filter_checks = dict(cached_filters=0, maximum_probe_discrepancy=0.0,
                         method='segmented discrete trapezoid, checked against pinned quadrature')

    def cached_filter(toggle, tb_j, wgrid, sign, chunk=4000):
        arrays = [np.asarray(a) for a in (toggle, tb_j, wgrid)]
        key = (sign,) + tuple((a.shape, a.dtype.str, hashlib.sha256(a.tobytes()).digest()) for a in arrays)
        if key not in cache:
            try:
                value = STAT.piecewise_trapezoid_filter(*arrays, sign)
            except ValueError:
                value = original_filter(toggle, tb_j, wgrid, sign, chunk=min(chunk, 128))
            probe = np.unique([0, len(wgrid)//2, len(wgrid)-1])
            reference = original_filter(toggle, tb_j, np.asarray(wgrid)[probe], sign, chunk=3)
            if not np.allclose(value[probe],reference,rtol=2e-9,atol=2e-9):
                raise ValueError('Segmented quadrature disagrees with pinned Fourier sums.')
            filter_checks['cached_filters'] += 1
            filter_checks['maximum_probe_discrepancy'] = max(
                filter_checks['maximum_probe_discrepancy'],float(np.max(np.abs(value[probe]-reference))))
            cache[key] = value
        return cache[key]

    systematics._ff_grid = cached_filter
    return reconstruct, inversion, filter_checks


def reconstruct_run(module, inversion, folder):
    config = module.SpectraReconConfig(data_folder=str(folder))
    recon = module.SpectraReconstructor(config)
    recon.load_observables()
    fits = []
    original_fit = inversion._ramsey_fit_dc
    import inspect
    signature = inspect.signature(original_fit)

    def record_fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        t, c = np.asarray(a['t']), np.asarray(a['C'])
        order = np.argsort(t)
        t, c = t[order], c[order]
        e = None if a['obs_err'] is None else np.asarray(a['obs_err'])[order]
        lower = a['c_min'] if e is None else 2*e
        measurable = (c > lower) & (c < a['c_max'])
        if measurable.sum() < 2:
            measurable = c < a['c_max']
        indices = np.flatnonzero(measurable)
        selected = indices
        if len(indices) >= 2:
            late = t[indices] >= .4*t[indices].max()
            if late.sum() >= 3:
                selected = indices[late]
        fits.append(dict(factor=a['factor'], value=result[0], stderr=result[1],
                         reliable=result[2], selected_indices=selected,
                         times=t, coefficient=c, coefficient_stderr=e))
        return result

    inversion._ramsey_fit_dc = record_fit
    try:
        recon.reconstruct()
    finally:
        inversion._ramsey_fit_dc = original_fit
    np.savez(folder / 'raw_spectra.npz', wk=recon.wk,
             **{key: value for key, value in recon.reconstructed_spectra.items()},
             **recon.reconstructed_spectra_err)
    write_json(folder / 'dc_fits.json', dict(zip(SPECTRA, fits)))
    recon.unfold_comb_bias()
    recon.add_systematic_errors()
    recon.save_reconstructed_spectra()
    with np.load(folder / 'specs.npz', allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def compare_spectra(original, revised):
    rows = []
    for key in SPECTRA:
        for part in (('real', 'imag') if np.iscomplexobj(revised[key]) else ('real',)):
            component = np.real if part == 'real' else np.imag
            old_error = component(original[key+'_err'])[1:]
            new_error = component(revised[key+'_err'])[1:]
            ratio = new_error / np.maximum(old_error, 1e-300)
            rows.append(dict(spectrum=key, component=part,
                             median_statistical_error_ratio=float(np.median(ratio)),
                             max_statistical_error_ratio=float(ratio.max()),
                             max_harmonic_point_change=float(np.max(np.abs(component(revised[key]-original[key])[1:]))),
                             old_dc=float(component(original[key])[0]), new_dc=float(component(revised[key])[0]),
                             old_dc_reliable=bool(original[key+'_dc_ok']), new_dc_reliable=bool(revised[key+'_dc_ok'])))
    return rows


def write_report(output):
    """Summarize the separate, completed postprocessing products for review."""
    provenance=json.loads((output/'provenance.json').read_text())
    summaries=json.loads((output/'observable_summaries.json').read_text())
    comparisons=json.loads((output/'spectral_comparison.json').read_text())
    baseline=json.loads((output/'baseline_validation.json').read_text())
    filters=json.loads((output/'filter_validation.json').read_text())
    overall={}
    lines=['Decision 30: analytic finite-measurement postprocessing',
           'Scientific code/data pin: '+provenance['pin'],
           'Physical measurements per observable/preparation/readout setting: '+str(provenance['shots_per_setting']),
           '',
           'Completed for capture, reference, raw and mitigated. SPAM-robust is excluded.',
           'Original run data, manuscript source and paper figures were not replaced.',
           'No trajectories, noise realizations or detector outcomes were generated; no gate optimization was run.',
           '',
           'Statistical model:',
           '  The saved ensemble-averaged readout probabilities define independent multinomial acquisitions.',
           '  Different observable and preparation settings, including repeated settings in different coefficient families, receive independent shot budgets.',
           '  Readout inversion and preparation correction use the saved exact SPAM calibration.',
           '  The physical covariance replaces the trajectory-only Monte Carlo covariance; it is not added to it.',
           '  Second-order Taylor propagation uses exact second/third/fourth multinomial moments.',
           '  Nonlinear finite-shot biases are saved separately; the probability-averaged coefficient central values are retained.',
           '  Gaussian quadrature is only a deterministic check of log-Taylor truncation, not a coverage test.',
           '',
           'Reconstruction:',
           '  The pinned harmonic inversion, adaptive uncertainty-weighted DC fits, two-step comb-bias correction and existing residual-error prescription were rerun.',
           '  The inherited correction uncertainty uses the pinned simultaneous-one-sigma-shift prescription. This pass does not validate it as a full joint covariance treatment.',
           '  Harmonic correlations across derived spectra and their use in later gate uncertainty bands have not been newly validated.',
           '  Inherited line-prior/model assumptions and later review decisions remain open.',
           '',
           'Validation and comparison:']
    for run in RUNS:
        summary=summaries[run]
        with np.load(output/run/'specs.npz') as updated, np.load(output/run/'baseline/pinned_specs.npz') as previous:
            ratios=[]
            for key in SPECTRA:
                for component in ((np.real,np.imag) if np.iscomplexobj(updated[key]) else (np.real,)):
                    ratios.extend((component(updated[key+'_err'])[1:]/component(previous[key+'_err'])[1:]).tolist())
            overall[run]=dict(real_harmonic_components=len(ratios),
                              median_statistical_error_multiplier=float(np.median(ratios)))
        lines += ['',run,
                  '  Median statistical SE multiplier over all %d real harmonic components: %.5g.' %
                  (overall[run]['real_harmonic_components'],overall[run]['median_statistical_error_multiplier']),
                  '  Recovered coefficient points: %d; maximum discrepancy %.5g.' % (summary['coefficient_points'],summary['maximum_saved_value_discrepancy']),
                  '  Physical acquisition settings: %d; assumed total measurements: %d (calibration excluded).' % (summary['physical_measurement_settings'],summary['total_physical_shots']),
                  '  Settings needing more than a first-order diagnostic: %d; second-order terms are included.' % len(summary['nonlinear_flags']),
                  '  Maximum Gaussian-quadrature/Taylor variance discrepancy: %.5g.' % summary['maximum_quadrature_variance_discrepancy'],
                  '  Maximum baseline error scaled to its channel harmonic scale: %.5g.' % max(v['scaled'] for v in baseline[run].values())]
        for row in (r for r in comparisons if r['run']==run):
            lines.append('  %s %s: median statistical SE multiplier %.4g; maximum %.4g.' %
                         (row['spectrum'],row['component'],row['median_statistical_error_ratio'],row['max_statistical_error_ratio']))
        old_fits=json.loads((output/run/'baseline/dc_fits.json').read_text())
        new_fits=json.loads((output/run/'dc_fits.json').read_text())
        for key in SPECTRA:
            old,new=old_fits[key],new_fits[key]
            lines.append('  DC %s: fit %.8g +/- %.5g, reliable=%s, indices=%s -> %.8g +/- %.5g, reliable=%s, indices=%s.' %
                         (key,old['value'],old['stderr'],old['reliable'],old['selected_indices'],
                          new['value'],new['stderr'],new['reliable'],new['selected_indices']))
    lines += ['',
              'Fourier quadratures: %d cached filters; maximum discrepancy against pinned quadrature probes %.5g.' %
              (filters['cached_filters'],filters['maximum_probe_discrepancy']),
              '',
              'Interpretation limits:',
              '  These are predicted finite-shot uncertainties around probability-averaged means, not a newly sampled experimental data set.',
              '  A changed DC reliability flag reflects the existing uncertainty-dependent detection rule; it does not by itself establish a more linear physical regime.',
              '  Expected finite-shot nonlinear bias can matter at nearly ideal contrasts; inspect observable_statistics.npz before making precision/coverage claims.',
              '  No robust-arm conclusions or updated gate predictions/optimizations are supplied.',
              '',
              'Outputs in each run folder:',
              '  params.npz: unchanged pinned configuration; results.npz: original means with revised physical-shot SEs.',
              '  observable_statistics.npz: primitive probabilities/weights, derivatives, first/second-order errors, finite-shot bias and curvature diagnostics.',
              '  raw_spectra.npz: inversion before comb-bias correction; specs.npz: revised pinned Stage-2 result.',
              '  dc_fits.json: actual DC fit indices, errors and reliability flags.',
              '  baseline/: unchanged pinned inputs and independently reproduced old-uncertainty reconstruction.',
              'Root provenance, checksums and spectral_comparison.csv record assumptions and changes.']
    (output/'summary.txt').write_text('\n'.join(lines)+'\n')
    write_json(output/'summary_statistics.json',overall)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pin', default='4241fbf')
    parser.add_argument('--shots', type=int, default=256000)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--moments-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.shots <= 0:
        parser.error('--shots must be positive')
    if output == ROOT or any(output.is_relative_to(path.resolve()) for path in ROOT.glob('DraftRun_*') if path.is_dir()):
        parser.error('Output must be separate from the original run folders.')
    if output.exists() and any(output.iterdir()) and not (output/'provenance.json').exists():
        parser.error('An existing nonempty output needs this postprocessor\'s provenance.json.')
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git','rev-parse',args.pin],cwd=ROOT,text=True).strip()
    provenance = dict(pin=commit, shots_per_setting=args.shots, runs=RUNS,
                      uncertainty_model='analytic_multinomial_independent_settings_delta2',
                      calibration='saved exact calibration; no calibration-shot uncertainty',
                      systematics='pinned unfolding and residual-error prescription retained',
                      noise_generation=False, measurement_outcome_sampling=False,
                      optimization=False, figure_generation=False)
    manifest = output / 'provenance.json'
    if manifest.exists() and json.loads(manifest.read_text()) != json.loads(json.dumps(provenance)):
        previous=json.loads(manifest.read_text())
        equivalent=dict(previous,uncertainty_model=provenance['uncertainty_model'])
        if equivalent != json.loads(json.dumps(provenance)):
            raise ValueError('Existing output has a different provenance or shot budget.')
        write_json(output/'previous_provenance.json',previous)
    write_json(manifest, provenance)
    originals, parameters, original_hashes = {}, {}, {}
    for run in RUNS:
        destination = output / run
        destination.mkdir(exist_ok=True)
        baseline = destination / 'baseline'
        baseline.mkdir(exist_ok=True)
        for name in ('params.npz', 'results.npz', 'specs.npz'):
            data = pinned_bytes(args.pin, f'{run}/{name}')
            if (ROOT / run / name).read_bytes() != data:
                raise ValueError(f'Original data differs from pin: {run}/{name}')
            original_hashes[f'{run}/{name}'] = hashlib.sha256(data).hexdigest()
            (baseline / ('pinned_specs.npz' if name == 'specs.npz' else name)).write_bytes(data)
            if name == 'params.npz':
                parameters[run] = load_npz(data)
                (destination / name).write_bytes(data)
            elif name == 'results.npz':
                originals[run] = load_npz(data)
    moment_paths = {RUNS[0]: RUNS[0], **{run: RUNS[1] for run in RUNS[1:]}}
    summaries = {}
    for cache_run in RUNS[:2]:
        transfers = phase_moments(cache_run, parameters[cache_run], output)
        for run in RUNS:
            if moment_paths[run] != cache_run:
                continue
            for key in ('T','M','t_grain','truncate','w_grain','n_shots','midpoint','synth_wmax'):
                if not np.array_equal(parameters[cache_run][key], parameters[run][key]):
                    raise ValueError(f'Arm/cache mismatch: {run}/{key}')
            summaries[run] = process_observables(run, parameters[run], originals[run],
                                               transfers, args.shots, output/run)
    write_json(output/'observable_summaries.json', summaries)
    if not args.moments_only:
        module, inversion, filter_checks = prepare_runtime(args.pin, output)
        comparison = []
        baseline_validation = {}
        for run in RUNS:
            folder = output/run
            print(f'[reconstruction] {run}: reproduce pinned baseline', flush=True)
            baseline_path=folder/'baseline/specs.npz'
            baseline = (load_npz(baseline_path.read_bytes()) if baseline_path.exists()
                        else reconstruct_run(module, inversion, folder/'baseline'))
            expected = load_npz((folder/'baseline/pinned_specs.npz').read_bytes())
            differences = {}
            for key in expected:
                if key in baseline and np.asarray(expected[key]).dtype.kind in 'fci':
                    delta=float(np.nanmax(np.abs(expected[key]-baseline[key])))
                    scale=float(np.nanmax(np.abs(expected[key][1:]))) if expected[key].ndim else float(abs(expected[key]))
                    differences[key] = dict(maximum_absolute=delta,harmonic_scale=scale,
                                             scaled=delta/max(scale,1e-300))
                    if not np.allclose(expected[key], baseline[key], rtol=1e-5,
                                       atol=max(5e-13,1e-5*scale), equal_nan=True):
                        write_json(folder/'baseline_validation.json', differences)
                        raise ValueError(f'Pinned reconstruction did not reproduce {run}/{key}: {differences[key]}')
                elif key in baseline and expected[key].dtype.kind=='b' and not np.array_equal(expected[key],baseline[key]):
                    raise ValueError(f'Baseline reliability flag differs: {run}/{key}')
            baseline_validation[run] = differences
            print(f'[reconstruction] {run}: analytic physical-shot uncertainties', flush=True)
            revised = reconstruct_run(module, inversion, folder)
            comparison.extend(dict(run=run, **row) for row in compare_spectra(expected, revised))
        write_json(output/'baseline_validation.json', baseline_validation)
        write_json(output/'filter_validation.json', filter_checks)
        write_json(output/'spectral_comparison.json', comparison)
        with (output/'spectral_comparison.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
    for path, digest in original_hashes.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Original data changed during postprocessing: {path}')
    write_json(output/'original_data_hashes.json', original_hashes)
    if not args.moments_only:
        write_report(output)
    print(f'[complete] {output}', flush=True)


if __name__ == '__main__':
    main()
