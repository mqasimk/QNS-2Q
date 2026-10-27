"""Uniform PSD scaling forecast from existing finite-M covariance calculations."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import forecast_repetition_sensitivity as f

OUT=f.ROOT/'reports/spectral_scale_forecast_2026-10-26'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(f.OUT/'_pinned_runtime/src'))
import jax
jax.config.update('jax_enable_x64',True)
from qns2q.characterize.systematics import analytic_spectra, _FWD_OBS

def main():
    spectra=analytic_spectra()
    data=np.load(f.OUT/'phase_covariances.npz')
    pairs=[tuple(x) for x in data['pulse_pairs']]
    ms=data['MS']
    inverse=np.load(f.OUT/'inverse_maps.npz')
    params={arm:f.saved.load_npz((f.BASE/run/'params.npz').read_bytes())
            for arm,run in [('capture',f.saved.RUNS[0]),('mitigated',f.saved.RUNS[3])]}
    scales=np.unique(np.r_[1,2,4,8,10,16,32,48,64,96,100,128,192,256,384,512,700,1024,
                          np.geomspace(1,10000,121)])
    rows=[]
    for arm,p in params.items():
        for m in (10,16,32,64,128,256,512,768,1024):
            mi=np.flatnonzero(ms==m)[0]
            for spectrum,names in f.CROSS.items():
                truth=complex(spectra[spectrum](np.array([13*2*np.pi/f.T]))[0])
                for part_index,part in enumerate(('real','imag')):
                    name=names[part_index]
                    pulse=tuple(_FWD_OBS[name][1])
                    covariance=data['covariances'][pairs.index(pulse),12,mi]
                    a=inverse[spectrum+'_'+part][12]/m
                    assert np.count_nonzero(a)==1 and a[12]!=0
                    value=truth.real if part=='real' else truth.imag
                    for scale in scales:
                        r=f.analyze(covariance*scale,name,12,p)
                        center=(1 if part_index==0 else -1)*a[12]*r['value']
                        se=abs(a[12])*r['stderr']
                        bias=center-scale*value
                        snr=abs(scale*value)/se if r['valid'] else 0.
                        guarded=(abs(scale*value)-abs(bias))/se if r['valid'] else 0.
                        rows.append(dict(arm=arm,M=m,scale=float(scale),spectrum=spectrum,
                            component=part,truth=scale*value,forecast_mean=center,SE=se,
                            SNR=snr,bias_in_SE=bias/se,bias_adjusted_score=guarded,
                            contrast_valid=r['valid']))
            print('Complete:',arm,m,flush=True)
    with (OUT/'sensitivity.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    thresholds=[]
    for arm in params:
        for m in (10,16,32,64,128,256,512,768,1024):
            for spectrum in f.CROSS:
                for part in ('real','imag'):
                    selected=[r for r in rows if r['arm']==arm and r['M']==m and r['spectrum']==spectrum and r['component']==part]
                    first=next((r for r in selected if r['bias_adjusted_score']>=3),None)
                    best=max(selected,key=lambda r:r['bias_adjusted_score'])
                    thresholds.append(dict(arm=arm,M=m,spectrum=spectrum,component=part,
                        first_grid_scale_3=None if first is None else first['scale'],
                        best_scale=best['scale'],maximum_bias_adjusted_score=best['bias_adjusted_score']))
    (OUT/'thresholds.json').write_text(json.dumps(thresholds,indent=2)+'\n')
    lines=['Uniform PSD-scaling sensitivity forecast at harmonic 13',
           'All six spectral functions are multiplied by the same positive alpha.',
           'Gaussian phase covariance scales exactly by alpha at fixed M; no trajectories or outcomes sampled.',
           'Original model parameters, run files, manuscript, and figures are unchanged.',
           '256,000 measurements per setting. Ideal pulse controls and exact SPAM calibration.',
           'Criterion: |alpha S| > 3 SE + |modeled raw comb-inversion bias|.',
           'These are planning forecasts, not coverage/significance validation.',
           '']
    for r in thresholds:
        if r['spectrum']=='S12':
            lines.append('{arm} M={M} {component} S12: first scale {first_grid_scale_3}; best score {maximum_bias_adjusted_score:.4g} at scale {best_scale:.4g}'.format(**r))
    lines+=['','Fixed alpha*M = 6144 comparison (no SPAM; harmonic 13 Re S12):']
    for m,alpha in [(16,384),(32,192),(64,96),(128,48)]:
        r=next(r for r in rows if r['arm']=='capture' and r['M']==m and r['scale']==alpha and r['spectrum']=='S12' and r['component']=='real')
        lines.append(f"  M={m}, alpha={alpha}: SNR={r['SNR']:.5g}, bias/SE={r['bias_in_SE']:.5g}, guarded score={r['bias_adjusted_score']:.5g}")
    lines+=['','Interpretation:',
            'Uniform scaling preserves spectral shape, relative cross-correlations, and PSD-matrix positivity.',
            'In the comb regime covariance is approximately proportional to alpha*M. Finite-M filtering means this is not an exact equivalence.',
            'At fixed M, relative comb-inversion bias does not shrink with alpha. Short-M leakage and cancellation can defeat the apparent statistical gain.',
            'Uniform scaling also strengthens the peaks and increases decoherence in all gate experiments. Gate fidelities and optimal controls were not recalculated here.',
            'An admissible global model change must also check the strong-feature settings; this study tests harmonic 13 only.',
            'Increasing only selected cross-spectra is not covered and can violate spectral-matrix positivity.',
            'SPAM robust remains excluded. No adoption of a rescaled manuscript model is implied.']
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    (OUT/'provenance.json').write_text(json.dumps(dict(pin='4241fbf',
        source_covariances=str(f.OUT/'phase_covariances.npz'),shots_per_setting=f.SHOTS,
        operation='all self- and cross-spectra scaled uniformly by alpha; phase covariance scaled by alpha',
        trajectories_generated=False,outcomes_sampled=False,gate_optimization=False,
        spectrum_parameters_changed=False),indent=2)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__': main()
