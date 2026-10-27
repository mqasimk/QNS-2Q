"""Summarize and plot the saved deterministic repetition forecast."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/repetition_forecast_2026-10-26'
rows=list(csv.DictReader((OUT/'adaptive_sensitivity.csv').open()))
labels={'S12':r'S_{1,2}','S112':r'S_{1,12}','S212':r'S_{2,12}'}
thresholds=[]
for key in labels:
    for part in ('real','imag'):
        for arm in ('capture','mitigated'):
            data=[r for r in rows if r['arm']==arm and r['spectrum']==key
                  and r['component']==part and r['harmonic']=='13']
            def first(target,bias):
                r=next((r for r in data if float(r['SNR'])-(abs(float(r['comb_bias_in_SE'])) if bias else 0)>=target),None)
                return None if r is None else int(r['maximum_M'])
            thresholds.append(dict(spectrum=key,component=part,arm=arm,
                first_M_statistical_3=first(3,False),first_M_with_bias_3=first(3,True),
                first_M_statistical_5=first(5,False),first_M_with_bias_5=first(5,True)))
(OUT/'decision_thresholds.json').write_text(json.dumps(thresholds,indent=2)+'\n')
with (OUT/'decision_thresholds.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(thresholds[0]));writer.writeheader();writer.writerows(thresholds)

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                    'font.family':'DejaVu Sans','savefig.facecolor':'white'})
fig,axes=plt.subplots(2,3,figsize=(13,7.5),sharex=True,sharey=True)
for j,key in enumerate(labels):
    for i,part in enumerate(('real','imag')):
        ax=axes[i,j]
        for arm,color,label in [('capture','#176BB0','No SPAM'),('mitigated','#D76B24','SPAM mitigated')]:
            data=[r for r in rows if r['arm']==arm and r['spectrum']==key
                  and r['component']==part and r['harmonic']=='13']
            ms=np.array([int(r['maximum_M']) for r in data])
            # A conservative bias allowance; this is not a calibrated coverage claim.
            score=np.array([float(r['SNR'])-abs(float(r['comb_bias_in_SE'])) for r in data])
            ax.plot(ms,np.maximum(score,1e-5),color=color,lw=2.2,label=label)
        ax.axhline(3,color='#343E4A',ls='--',lw=1.1)
        ax.text(1.15e5,3.6,'3× SE',color='#343E4A',fontsize=9)
        ax.axvspan(10,16,color='#C8CDD4',alpha=.25)
        ax.set(xscale='log',yscale='log',xlim=(10,262144),ylim=(.003,220),
               title=('$\\mathrm{Re}\\,' if part=='real' else '$\\mathrm{Im}\\,')+labels[key]+'$')
        ax.grid(which='major',alpha=.15)
        if i==1: ax.set_xlabel('Maximum repetitions per setting, M')
        if j==0: ax.set_ylabel('Bias-adjusted signal / SE')
axes[0,0].legend(loc='lower right',frameon=False,fontsize=9)
fig.suptitle('Repetition forecast at harmonic 13: 256,000 measurements per setting',fontsize=16,x=.52,y=.98)
fig.text(.5,.025,'Each acquisition may choose M up to the plotted limit. Signal score = (|input spectrum| − |modeled comb bias|) / statistical SE.\n'
         'Pinned Gaussian model; ideal pulses; exact SPAM calibration. Gray strip: original M=10–16. No new simulations.',
         ha='center',fontsize=9,color='#4D5763')
fig.tight_layout(rect=(0,.085,1,.94))
fig.savefig(OUT/'repetition_sensitivity.png',dpi=180)
plt.close(fig)

lines=['Repetition forecast: decision notes',
       'Question: can more sequence repetitions resolve weak correlations at 256k physical shots, without new simulations?',
       '',
       'Yes for many components in the specified ideal Gaussian model. A uniform modest increase is not enough.',
       'The strongest line settings should retain shorter sequences; each acquisition can use its own M.',
       'All results use 256,000 independent physical measurements per observable/preparation/readout setting.',
       '',
       'Harmonic 13 (omega*tau = 13*2*pi/160 = %.8f): first tested M satisfying |S| > 3*SE + |comb bias|.'%(13*2*np.pi/160),
       'These are grid-bracketed planning thresholds, not experimentally validated detection significances.']
for row in thresholds:
    lines.append('  {arm:9s} {spectrum:4s} {component:4s}: M={first_M_with_bias_3}; statistical-only threshold M={first_M_statistical_3}'.format(**row))
lines+=['','Number of the 120 cross-spectrum components satisfying the same criterion:']
for arm in ('capture','mitigated'):
    for limit in (16,128,1024,4096,8192,16384,262144):
        subset=[r for r in rows if r['arm']==arm and r['maximum_M']==str(limit)]
        count=sum(float(r['SNR'])-abs(float(r['comb_bias_in_SE']))>=3 for r in subset)
        lines.append(f'  {arm:9s} M <= {limit:6d}: {count}/120')
validation=json.loads((OUT/'integration_validation.json').read_text())
grid=max(r['maximum_valid_coefficient_SE_change'] or 0 for r in validation if r['check']=='double_resolution')
band=max(r['maximum_valid_coefficient_SE_change'] or 0 for r in validation if r['check']=='double_bandwidth' and r['harmonic']==13)
lines+=['','Method and validation:',
        '  Scientific input: commit 4241fbf, showcase PSD and saved SPAM parameters.',
        '  The saved phase moments alone cannot determine the response of a different sequence.',
        '  Gaussian phase covariance is integrated deterministically with finite-M periodic filters.',
        '  The covariance-to-readout and second-order multinomial propagation are the prior saved-data implementation.',
        '  The exact ideal-control harmonic inverse is used; the pinned quadrature inverse differs by at most 0.172% in max-entry normalized error.',
        '  The original coarse-grid, sampled-toggle model was evaluated separately at M=10/16: 240 cross coefficients agree with saved values within 3.77 original Monte Carlo SE.',
        f'  Doubling the frequency/lag resolution changes the checked valid coefficient SEs by at most {grid:.3g} relative.',
        f'  Doubling the modeled bandwidth changes checked harmonic-13 coefficient SEs by up to {100*band:.2f}%; use rounded planning thresholds.',
        '',
        'Limits and interpretation:',
        '  This is a model-dependent forecast, not new experimental data, a calibrated coverage test, or revised paper results.',
        '  The main scan preserves the original noise bandwidth, 0 <= omega*tau <= pi/2, while replacing its coarse frequency discretization by a converged continuum integral.',
        '  Pulse timings are ideal continuous library sequences; original sampled-toggle and coarse-grid artifacts are not extrapolated.',
        '  M is the number of T=160*tau blocks. Raising M from 16 to 6144 lengthens that setting by 384x at fixed shot count.',
        '  Pulse imperfections, relaxation, calibration error, and experimental drift are absent; large M must be checked against those limits.',
        '  Bias allowance uses the known input model: finite-M plus harmonic-truncation bias of the raw linear inverse. No DC refit or existing nonlinear comb-bias correction was run.',
        '  The choice of M is informed by the known showcase model. A blind experiment would need a pilot/adaptive procedure and its acquisition cost.',
        '  SPAM robust is excluded. Raw biased reconstructions are not treated as detections. No gate optimization or figure replacement.',
        '',
        'Outputs: repetition_sensitivity.png; decision_thresholds.csv; adaptive_sensitivity.csv; spectral_sensitivity.csv;',
        '         coefficient_sensitivity.npz; phase_covariances.npz; baseline/integration/inverse validation JSON; provenance.json.']
(OUT/'decision_notes.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:31]))
