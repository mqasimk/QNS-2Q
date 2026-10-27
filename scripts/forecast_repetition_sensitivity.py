"""Deterministic repetition forecast; no trajectories or outcome samples.

Uses the pinned showcase PSD, ideal library pulse timing, Gaussian dephasing,
and the saved exact SPAM calibration. Forecasts are separate from paper data.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['QNS2Q_REGIME'] = 'showcase'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import postprocess_shot_noise as saved
STAT = saved.STAT
BASE = ROOT / 'reports/decision30_analytic_shots_2026-10-25'
OUT = ROOT / 'reports/repetition_forecast_2026-10-26'
T = 160.0
SHOTS = 256000
MS = np.unique(np.r_[10, 16, 24, 32, 48, 64, 96, 128, 192, 256,
                     384, 512, 768, 1024, 1536, 2048, 3072, 4096,
                     6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536,
                     98304, 131072, 196608, 262144,
                     np.round(np.geomspace(10, 262144, 88)).astype(int)])
CROSS = {
    'S12': ('C_12_12_MT_1', 'C_12_12_MT_2'),
    'S112': ('C_1_2_MT_1', 'C_1_2_MT_2'),
    'S212': ('C_2_1_MT_1', 'C_2_1_MT_2'),
}
PULSES = {
    'CPMG': [0, .25, .75, 1], 'CDD3': [0, .125, .375, .5, .625, .875, 1],
    'CDD1': [0, .5, 1], 'CDD1-1/2': [0, .25, .5, .75, 1],
    'CDD1-1/4': np.linspace(0, 1, 9), 'FID': [0, 1],
}


def geometric_sum(x, count):
    x = (np.asarray(x) + np.pi) % (2*np.pi) - np.pi
    return (count * np.exp(.5j*(count-1)*x)
            * np.sinc(count*x/(2*np.pi)) / np.sinc(x/(2*np.pi)))


def cycle_segments(pulse):
    edges = np.unique(np.concatenate([PULSES[p] for p in pulse]))
    mid = (edges[:-1]+edges[1:])/2
    ys = [(-1.)**(np.searchsorted(PULSES[p], mid, side='right')-1) for p in pulse]
    return edges, np.array([*ys, ys[0]*ys[1]])


def ideal_block_filters(pulse, harmonic, w):
    edges, ys = cycle_segments(pulse)
    ct = T/harmonic
    duration = np.diff(edges)*ct
    mid = (edges[:-1]+edges[1:])*ct/2
    intervals = (duration[:, None]*np.sinc(duration[:, None]*w/(2*np.pi))
                 * np.exp(1j*mid[:, None]*w))
    return (ys @ intervals)*geometric_sum(w*ct, harmonic)


def spectrum_matrix(spectra, w):
    s = np.zeros((3, 3, len(w)), complex)
    for a, key in enumerate(('S11', 'S22', 'S1212')):
        s[a, a] = np.asarray(spectra[key](w))
    for a, b, key in ((0, 1, 'S12'), (0, 2, 'S112'), (1, 2, 'S212')):
        s[a, b] = np.asarray(spectra[key](w))
        s[b, a] = s[a, b].conj()
    return s


def covariance_sweep(smat, w, filters, bins, bands, repetitions=MS):
    """Fourier block-covariances give every M with no long time grid.

    Var(sum of M blocks) = M r0 + 2 sum_p (M-p) r_p. Lags
    beyond bins/2 are omitted; doubling bins checks that approximation.
    """
    dw = w[1]-w[0]
    count = bins//2
    lag = np.arange(1, count)
    covariance = np.zeros((len(repetitions), 3, 3))
    tail = []
    for a in range(3):
        for b in range(a, 3):
            h = np.real(smat[a, b]*filters[a]*filters[b].conj())
            folded = h[:-1].reshape(bands, bins).sum(axis=0)
            folded[0] += .5*(h[-1]-h[0])
            r = np.fft.fft(folded).real[:count]*dw/np.pi
            c0 = np.r_[0., np.cumsum(r[1:])]
            c1 = np.r_[0., np.cumsum(lag*r[1:])]
            p = np.minimum(np.asarray(repetitions)-1, count-1)
            val = repetitions*r[0]+2*(repetitions*c0[p]-c1[p])
            covariance[:, a, b] = covariance[:, b, a] = val
            tail.append(float(np.max(np.abs(r[-32:]))))
    return covariance, max(tail)


def gaussian_transfer(covariance):
    differences = STAT.DIAGONAL_BASIS[:, :, None]-STAT.DIAGONAL_BASIS[:, None, :]
    decay = np.einsum('aij,ab,bij->ij', differences, covariance, differences)/8
    if decay.min() < -1e-9:
        raise ValueError('Nonpositive Gaussian phase covariance')
    return np.exp(-np.maximum(decay, 0))


def discrete_filters(pulse, harmonic, repetitions, params, w, trajectories):
    """Exactly the pinned tiled-toggle trapezoid, without an M-long array."""
    n = int(params['t_grain'])
    dt = repetitions*T/(repetitions*n-1)
    y = np.asarray(trajectories.make_y(np.linspace(0,T,n), list(pulse),
                                     ctime=T/harmonic, m=1))
    result = []
    for a in range(3):
        v = y[a,a]
        weighted = STAT.piecewise_trapezoid_filter(v, np.arange(n)*dt, w)
        unweighted = weighted+.5*dt*(v[0]+v[-1]*np.exp(1j*w*(n-1)*dt))
        full = (unweighted*geometric_sum(w*n*dt, repetitions)
                -.5*dt*(v[0]+v[-1]*np.exp(1j*w*(repetitions*n-1)*dt)))
        result.append(full)
    return np.asarray(result)


def covariance_discrete(smat, filters, dw):
    return np.array([[np.sum(np.real(smat[a,b]*filters[a]*filters[b].conj()))*dw/np.pi
                      for b in range(3)] for a in range(3)])


def inverse_maps(inversion, params):
    captured = []
    original = inversion.solve_inverse
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(np.real(result[1]))
        return result
    inversion.solve_inverse = capture
    try:
        for name in ('recon_S_1_2', 'recon_S_1_12', 'recon_S_2_12'):
            getattr(inversion,name)([np.zeros(20),np.zeros(20)],
                                    c_times=params['c_times'],m=1,T=T,diagnostics=False)
    finally:
        inversion.solve_inverse = original
    pinned={key: captured[2*i:2*i+2] for i,key in enumerate(CROSS)}
    # Match the continuum ideal controls used by the forecast. The pinned
    # numerical time quadrature leaves tiny off-diagonal entries (~1e-8 in
    # inverse rows); multiplying these by an unmeasurable log estimator at
    # high M produces a spurious global error blow-up. Exact pulse filters
    # recover the harmonic selection rules instead of extending that artifact.
    recipes={
        'S12':[(('CPMG','CPMG'),0,1),(('CDD3','CPMG'),0,1)],
        'S112':[(('CPMG','FID'),0,2),(('CPMG','CDD3'),0,1)],
        'S212':[(('FID','CPMG'),1,2),(('CDD3','CPMG'),1,0)],
    }
    result={}; checks={}; wk=2*np.pi*np.arange(1,21)/T
    for key,recipe in recipes.items():
        result[key]=[]
        for part,(pulse,a,b) in enumerate(recipe):
            kernel=[]
            for k in range(1,21):
                ff=ideal_block_filters(pulse,k,wk)
                product=(2/T)*ff[a]*ff[b].conj()
                kernel.append(product.real if part==0 else product.imag)
            kernel=np.array(kernel)
            kernel[abs(kernel)<1e-12*np.max(abs(kernel))]=0
            inverse=np.linalg.inv(kernel)
            result[key].append(inverse)
            checks[key+('_real' if part==0 else '_imag')]=float(
                np.max(abs(inverse-pinned[key][part]))/np.max(abs(inverse)))
    saved.write_json(OUT/'inverse_validation.json',checks)
    return result


def acquisition(name, k):
    _, kind, qubit = next(t for t in STAT.HARMONIC_EXPERIMENTS if t[0]==name)
    return STAT.Acquisition(name,kind,qubit,k,0,1,False)


def analyze(covariance, name, k, params):
    transfer = gaussian_transfer(covariance)
    try:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            r = STAT.analyze_acquisition(acquisition(name,k),[transfer]*4,params,SHOTS)
        valid = (np.isfinite(r['stderr']) and r['minimum_contrast_snr']>=10
                 and r['gaussian_quadrature_relative_variance_discrepancy']<.01)
        r['valid'] = bool(valid)
        return r
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
        return dict(value=np.nan,stderr=np.inf,valid=False,means=np.full(4,np.nan),
                    variances=np.full(4,np.nan),minimum_contrast_snr=0.,
                    gaussian_quadrature_relative_variance_discrepancy=np.inf)


def main():
    OUT.mkdir(exist_ok=True)
    _, inversion, _ = saved.prepare_runtime('4241fbf', OUT)
    from qns2q.characterize import systematics
    from qns2q.model import trajectories
    spectra = systematics.analytic_spectra()
    params = {arm: saved.load_npz((BASE/run/'params.npz').read_bytes())
              for arm,run in zip(('capture','reference','raw','mitigated'),saved.RUNS)}
    maps = inverse_maps(inversion,params['capture'])
    np.savez(OUT/'inverse_maps.npz',**{f'{key}_{part}':v[j] for key,v in maps.items()
                                    for j,part in enumerate(('real','imag'))})
    names = [name for pair in CROSS.values() for name in pair]
    pulse_pairs = sorted(set(tuple(systematics._FWD_OBS[n][1]) for n in names))
    bands, bins = 40, 2048
    w = np.linspace(0,2*np.pi/T*bands,bands*bins+1)
    smat = spectrum_matrix(spectra,w)
    cache = OUT/'phase_covariances.npz'
    cached = cache.exists() and np.array_equal(np.load(cache)['MS'],MS)
    if cached:
        z=np.load(cache); covariances=z['covariances']; tails=z['tails']
    else:
        covariances=np.empty((len(pulse_pairs),20,len(MS),3,3)); tails=[]
        for j,pulse in enumerate(pulse_pairs):
            for k in range(1,21):
                f=ideal_block_filters(pulse,k,w)
                covariances[j,k-1],tail=covariance_sweep(smat,w,f,bins,bands)
                tails.append(tail)
            print('Covariances:',pulse,flush=True)
        np.savez(cache,covariances=covariances,tails=tails,MS=MS,pulse_pairs=pulse_pairs)
    lookup={pulse:i for i,pulse in enumerate(pulse_pairs)}
    # Validate continuum integration/lag truncation at low, middle and high harmonics.
    validations=[]
    for pulse in pulse_pairs:
        for k in (1,8,13,20):
            for fine_bands,fine_bins,label in ((40,4096,'double_resolution'),(80,2048,'double_bandwidth')):
                wf=np.linspace(0,2*np.pi/T*fine_bands,fine_bands*fine_bins+1)
                sf=spectrum_matrix(spectra,wf)
                cf,_=covariance_sweep(sf,wf,ideal_block_filters(pulse,k,wf),fine_bins,fine_bands)
                coarse=covariances[lookup[pulse],k-1]
                scale=np.sqrt(np.diagonal(cf,axis1=1,axis2=2)[:,:,None]*
                              np.diagonal(cf,axis1=1,axis2=2)[:,None,:])
                readout_differences=[]
                for name in names:
                    if tuple(systematics._FWD_OBS[name][1])!=pulse: continue
                    for mi in np.flatnonzero(np.isin(MS,[16,128,1024,8192,32768,131072])):
                        rc=analyze(coarse[mi],name,k-1,params['capture'])
                        rf=analyze(cf[mi],name,k-1,params['capture'])
                        if rc['valid'] and rf['valid']:
                            readout_differences.append(abs(rc['stderr']-rf['stderr'])/rf['stderr'])
                validations.append(dict(pulse=pulse,harmonic=k,check=label,
                    maximum_valid_coefficient_SE_change=max(readout_differences,default=None),
                    maximum_relative_to_variance=float(np.max(np.abs(cf-coarse)/np.maximum(scale,1e-20))),
                    maximum_relative_diagonal=float(np.max(np.abs(np.diagonal(cf-coarse,axis1=1,axis2=2))/
                                                           np.maximum(np.diagonal(cf,axis1=1,axis2=2),1e-20)))))
    saved.write_json(OUT/'integration_validation.json',validations)
    # Baseline: exact finite discrete model vs stored noise-ensemble averages.
    baseline=[]; filter_checks=[]
    for arm in ('capture','reference'):
        p=params[arm]; m=int(p['M']); grain=int(p['w_grain']); dw=float(p['wmax'])/grain
        wd=(np.arange(2*grain)+.5)*dw; sd=spectrum_matrix(spectra,wd)
        run=saved.RUNS[0 if arm=='capture' else 1]
        old=np.load(BASE/run/'baseline/results.npz')
        physical=np.load(BASE/run/'observable_statistics.npz')
        for name in names:
            pulse=tuple(systematics._FWD_OBS[name][1])
            for k in range(1,21):
                fd=discrete_filters(pulse,k,m,p,wd,trajectories)
                cd=covariance_discrete(sd,fd,dw)
                r=analyze(cd,name,k-1,p)
                i=np.flatnonzero((physical['coefficient']==name)&(physical['index']==k-1))[0]
                baseline.append(dict(arm=arm,coefficient=name,harmonic=k,
                    prediction=r['value'],saved_mean=float(old[name][k-1]),
                    difference_over_saved_MC_SE=float((r['value']-old[name][k-1])/old[name+'_err'][k-1]),
                    predicted_physical_SE=r['stderr'],saved_physical_SE=float(physical['stderr'][i])))
                if name==names[0] and k==13:
                    tv=np.linspace(0,m*T,m*int(p['t_grain']))
                    yy=trajectories.make_y(np.linspace(0,T,int(p['t_grain'])),list(pulse),ctime=T/k,m=m)
                    probe=[0,grain,2*grain-1]
                    direct=np.asarray(systematics._ff_grid(np.asarray(yy[0,0]),tv,wd[probe],1,chunk=3))
                    filter_checks.append(float(np.max(abs(fd[0,probe]-direct))))
        print('Baseline checked:',arm,flush=True)
    saved.write_json(OUT/'baseline_validation.json',baseline)
    saved.write_json(OUT/'filter_validation.json',dict(maximum_direct_filter_error=max(filter_checks)))
    # Forecast no-SPAM and exactly mitigated acquisition; raw is biased and is
    # intentionally not treated as a detection of the physical cross spectrum.
    rows=[]; primitive=[]; measurements={}
    for arm in ('capture','mitigated'):
        p=params[arm]
        all_coefficients=[]; all_errors=[]; all_valid=[]
        for mi,m in enumerate(MS):
            coef={}; err={}; valid={}; contrast={}
            for name in names:
                pulse=tuple(systematics._FWD_OBS[name][1]); j=lookup[pulse]
                rr=[analyze(covariances[j,k,mi],name,k,p) for k in range(20)]
                coef[name]=np.array([r['value'] for r in rr])
                err[name]=np.array([r['stderr'] for r in rr])
                valid[name]=np.array([r['valid'] for r in rr])
                contrast[name]=np.array([r['minimum_contrast_snr'] for r in rr])
                if name=='C_12_12_MT_1':
                    r=rr[12]
                    primitive.append(dict(arm=arm,M=int(m),harmonic=13,
                        YY=float(r['means'][1]),YY_SE=float(np.sqrt(r['variances'][1])),
                        YY_SNR=float(abs(r['means'][1])/np.sqrt(r['variances'][1])),
                        coefficient=r['value'],coefficient_SE=r['stderr'],valid=r['valid']))
            for key,pair in CROSS.items():
                truth=np.asarray(spectra[key](2*np.pi*np.arange(1,21)/T))
                for part_i,part in enumerate(('real','imag')):
                    name=pair[part_i]; a=maps[key][part_i]/m
                    # Negate imaginary branch as in the pinned inverse.
                    center=(1 if part_i==0 else -1)*a@coef[name]
                    component=truth.real if part_i==0 else truth.imag
                    # Conservative: every acquisition with appreciable influence
                    # must retain a well-resolved log contrast.
                    influential=abs(a)>1e-7*np.max(abs(a),axis=1)[:,None]
                    ok=np.all(~influential|valid[name][None,:],axis=1)
                    safe_error=np.where(np.isfinite(err[name]),err[name]**2,1e200)
                    se=np.sqrt((a*a)@safe_error)
                    for k in range(20):
                        rows.append(dict(arm=arm,M=int(m),spectrum=key,component=part,
                            harmonic=k+1,truth=float(component[k]),forecast_mean=float(center[k]),
                            statistical_SE=float(se[k]),SNR=float(abs(component[k])/se[k]) if ok[k] else 0.,
                            valid=bool(ok[k]),comb_bias=float(center[k]-component[k])))
            if m in (16,128,1024,8192,16384): print('Forecast:',arm,int(m),flush=True)
            all_coefficients.append([coef[n] for n in names])
            all_errors.append([err[n] for n in names])
            all_valid.append([valid[n] for n in names])
        measurements[arm]=dict(coefficients=np.array(all_coefficients),
                               errors=np.array(all_errors),valid=np.array(all_valid))
    np.savez(OUT/'coefficient_sensitivity.npz',MS=MS,names=names,
             **{arm+'_'+k:v for arm,d in measurements.items() for k,v in d.items()})
    adaptive=[]
    for arm,d in measurements.items():
        scaled_error=d['errors']/MS[:,None,None]
        scaled_error=np.where(d['valid'],scaled_error,np.inf)
        for end,limit in enumerate(MS):
            choice=np.argmin(scaled_error[:end+1],axis=0)
            for key,pair in CROSS.items():
                truth=np.asarray(spectra[key](2*np.pi*np.arange(1,21)/T))
                for part_i,part in enumerate(('real','imag')):
                    ni=names.index(pair[part_i]); pick=choice[ni]; selected_M=MS[pick]
                    selected_error=scaled_error[pick,ni,np.arange(20)]
                    selected_coefficient=d['coefficients'][pick,ni,np.arange(20)]/selected_M
                    a=maps[key][part_i]
                    se=np.sqrt((a*a)@np.where(np.isfinite(selected_error),selected_error**2,1e200))
                    center=(1 if part_i==0 else -1)*a@selected_coefficient
                    component=truth.real if part_i==0 else truth.imag
                    for k in range(20):
                        adaptive.append(dict(arm=arm,maximum_M=int(limit),spectrum=key,
                            component=part,harmonic=k+1,truth=float(component[k]),
                            statistical_SE=float(se[k]),SNR=float(abs(component[k])/se[k]),
                            forecast_mean=float(center[k]),comb_bias_in_SE=float((center[k]-component[k])/se[k]),
                            M_for_same_harmonic=int(selected_M[k])))
    with (OUT/'adaptive_sensitivity.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(adaptive[0]));writer.writeheader();writer.writerows(adaptive)
    saved.write_json(OUT/'adaptive_schedule.json',{
        arm:{name:MS[np.argmin(np.where(d['valid'],d['errors']/MS[:,None,None],np.inf),axis=0)[ni]].tolist()
             for ni,name in enumerate(names)} for arm,d in measurements.items()})
    for filename,data in (('spectral_sensitivity.csv',rows),('harmonic13_readout.csv',primitive)):
        with (OUT/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    saved.write_json(OUT/'provenance.json',dict(pin='4241fbf',shots_per_setting=SHOTS,
        sequence_block_time=T,repetitions=MS,model='Gaussian pinned showcase PSD; ideal periodic library pulses',
        covariance='finite repetition Fourier block-covariance sum',frequency_cutoff=float(w[-1]),
        lag_cutoff=bins//2-1,calibration='exact saved calibration; no calibration uncertainty',
        uncertainty='second-order multinomial propagation; independent settings; exact ideal-filter counterpart of pinned cross-spectrum inverse',
        excludes='SPAM robust, raw-arm detection, DC fits, comb-bias correction, gate optimization, experimental control errors',
        simulations_run=False,measurement_outcomes_sampled=False,
        input_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                      for run in saved.RUNS for path in [BASE/run/'params.npz']}))
    summarize(rows,primitive,validations,baseline,adaptive)


def summarize(rows,primitive,validations,baseline,adaptive):
    lines=['Repetition sensitivity forecast, 256,000 independent measurements per setting',
           'Pinned Gaussian showcase model; no new trajectories or outcome draws.',
           'Statistical sensitivity only; significance/coverage and systematic corrections are not certified.', '']
    best=[]
    for arm in ('capture','mitigated'):
        lines.append(arm)
        for key in CROSS:
            for part in ('real','imag'):
                selected=[r for r in rows if r['arm']==arm and r['spectrum']==key and r['component']==part and r['harmonic']==13]
                b=max(selected,key=lambda r:r['SNR']); old=next(r for r in selected if r['M']==(16 if arm=='capture' else 10))
                first3=next((r['M'] for r in selected if r['SNR']>=3),None)
                first5=next((r['M'] for r in selected if r['SNR']>=5),None)
                best.append(dict(arm=arm,spectrum=key,component=part,harmonic=13,
                                 baseline_SNR=old['SNR'],first_grid_M_3=first3,first_grid_M_5=first5,
                                 best_grid_M=b['M'],maximum_SNR=b['SNR']))
                lines.append(f"  harmonic 13 {key} {part}: baseline SNR {old['SNR']:.4g}; first M>=3: {first3}; first M>=5: {first5}; best {b['SNR']:.4g} at M={b['M']}")
    lines+=['','Adaptive M per acquisition:']
    adaptive_summary=[]
    for arm in ('capture','mitigated'):
        for key in CROSS:
            for part in ('real','imag'):
                selected=[r for r in adaptive if r['arm']==arm and r['spectrum']==key and r['component']==part and r['harmonic']==13]
                b=selected[-1]
                first3=next((r['maximum_M'] for r in selected if r['SNR']>=3),None)
                first5=next((r['maximum_M'] for r in selected if r['SNR']>=5),None)
                adaptive_summary.append(dict(arm=arm,spectrum=key,component=part,harmonic=13,
                    first_grid_M_3=first3,first_grid_M_5=first5,maximum_SNR=b['SNR'],
                    selected_M=b['M_for_same_harmonic'],comb_bias_in_SE=b['comb_bias_in_SE']))
                lines.append(f"  {arm} {key} {part}: first max M>=3: {first3}; >=5: {first5}; best SNR {b['SNR']:.4g}; own M {b['M_for_same_harmonic']}; comb bias/SE {b['comb_bias_in_SE']:.4g}")
    saved.write_json(OUT/'adaptive_harmonic13_summary.json',adaptive_summary)
    lines+=['','Maximum normalized covariance difference, grid doubled: '+str(max(r['maximum_relative_to_variance'] for r in validations if r['check']=='double_resolution')),
            'Maximum normalized covariance difference, bandwidth doubled: '+str(max(r['maximum_relative_to_variance'] for r in validations if r['check']=='double_bandwidth')),
            'Baseline largest difference in old MC SE units: '+str(max(abs(r['difference_over_saved_MC_SE']) for r in baseline)),
            '', 'Higher M increases experiment duration at fixed shots; no fixed-wall-time claim.',
            'Large-M results use a refined continuous spectrum, not the coarse discrete simulation frequency grid.',
            'Ideal instantaneous pulses omit pulse errors, relaxation, and drift.']
    saved.write_json(OUT/'harmonic13_summary.json',best)
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
