"""Targeted follow-up at user-selected M of order 100; no simulations."""
import json
import sys
import numpy as np
import forecast_repetition_sensitivity as f

sys.path.insert(0,str(f.OUT/'_pinned_runtime/src'))
import jax
jax.config.update('jax_enable_x64',True)
from qns2q.characterize.systematics import analytic_spectra, _FWD_OBS

out=f.ROOT/'reports/spectral_scale_forecast_2026-10-26'
spectra=analytic_spectra()
ms=np.array([1,2,4,8,10,16,24,32,48,64,96,100,128])
w=np.linspace(0,np.pi/2,40*2048+1)
smat=f.spectrum_matrix(spectra,w)
inverse=np.load(f.OUT/'inverse_maps.npz')
params={arm:f.saved.load_npz((f.BASE/run/'params.npz').read_bytes())
        for arm,run in [('capture',f.saved.RUNS[0]),('mitigated',f.saved.RUNS[3])]}
rows=[]
for spectrum,names in f.CROSS.items():
    truth=complex(spectra[spectrum](np.array([13*2*np.pi/f.T]))[0])
    for part_index,part in enumerate(('real','imag')):
        name=names[part_index]
        pulse=tuple(_FWD_OBS[name][1])
        cov,_=f.covariance_sweep(smat,w,f.ideal_block_filters(pulse,13,w),2048,40,ms)
        value=truth.real if part=='real' else truth.imag
        for m,alpha in [(100,80),(128,60),(100,100),(128,80)]:
            a=inverse[spectrum+'_'+part][12,12]/m
            for arm,p in params.items():
                r=f.analyze(alpha*cov[np.flatnonzero(ms==m)[0]],name,12,p)
                center=(1 if part_index==0 else -1)*a*r['value']
                se=abs(a)*r['stderr'];bias=center-alpha*value
                rows.append(dict(arm=arm,M=m,scale=alpha,spectrum=spectrum,component=part,
                    statistical_SNR=abs(alpha*value)/se,bias_in_SE=bias/se,
                    bias_adjusted_score=(abs(alpha*value)-abs(bias))/se,
                    valid=r['valid']))
strong=[]
for k in range(1,21):
    cov,_=f.covariance_sweep(smat,w,f.ideal_block_filters(('CPMG','CPMG'),k,w),2048,40,ms)
    for target,alpha in [(100,80),(128,60),(100,100),(128,80)]:
        for arm,p in params.items():
            results=[(m,f.analyze(alpha*c,'C_12_0_MT_1',k-1,p)) for m,c in zip(ms,cov) if m<=target]
            acceptable=[int(m) for m,r in results if r['valid']]
            r=results[-1][1]
            strong.append(dict(harmonic=k,arm=arm,scale=alpha,target_M=target,
                contrast_SNR_at_target=r['minimum_contrast_snr'],valid_at_target=r['valid'],
                largest_tested_valid_M=max(acceptable,default=None)))
report=dict(criterion='|scaled S| > 3 SE + |modeled raw comb bias|',shots_per_setting=256000,
    harmonic13=rows,CPMG_joint_coherence_audit=strong,
    limits='Same pinned Gaussian finite-band model, ideal pulses, exact calibration; no gate-performance or full-suite validation')
(out/'target_M100.json').write_text(json.dumps(report,indent=2)+'\n')
for r in rows:
    print(r)
print('Strong-feature coherence, harmonic 9:')
for r in strong:
    if r['harmonic']==9: print(r)
