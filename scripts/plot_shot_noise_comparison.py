"""Plot existing versus analytic-shot-noise reconstructions on matched axes.

This reads the separate decision-30 products and pinned original summaries.
It does not reconstruct spectra or run any simulation. The SPAM-robust arm is
excluded. PNGs are ready for an inline preview or a comparison PDF.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['QNS2Q_REGIME'] = 'showcase'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, FixedLocator, FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'capture': 'DraftRun_NoSPAM_showcase_cap',
    'reference': 'DraftRun_SPAM_showcase_reference',
    'raw': 'DraftRun_SPAM_showcase_raw',
    'mitigated': 'DraftRun_SPAM_showcase_mitigated',
}
STYLE = {
    'capture': ('#0072B2', 'o', 'No-SPAM capture'),
    'reference': ('#0072B2', 'o', 'Reference'),
    'raw': ('#CC79A7', 'v', 'Raw SPAM'),
    'mitigated': ('#D55E00', '^', 'SPAM-mitigated'),
}
PANELS = [
    ('S11', np.real, r'$S_{1,1}$'), ('S22', np.real, r'$S_{2,2}$'),
    ('S1212', np.real, r'$S_{12,12}$'),
    ('S12', np.real, r'$\mathrm{Re}\,S_{1,2}$'),
    ('S112', np.real, r'$\mathrm{Re}\,S_{1,12}$'),
    ('S212', np.real, r'$\mathrm{Re}\,S_{2,12}$'),
    ('S12', np.imag, r'$\mathrm{Im}\,S_{1,2}$'),
    ('S112', np.imag, r'$\mathrm{Im}\,S_{1,12}$'),
    ('S212', np.imag, r'$\mathrm{Im}\,S_{2,12}$'),
]


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def signed_ticks(limits, width):
    """Keep decade ticks apart in transformed space, including near zero."""
    transformed=np.arcsinh(np.asarray(limits)/width)
    minimum_spacing=.115*(transformed[1]-transformed[0])
    ticks=[0.0] if limits[0] <= 0 <= limits[1] else []
    exponents=range(int(np.floor(np.log10(width)))-2,
                    int(np.ceil(np.log10(max(abs(limits)))))+1)
    for value in sorted([sign*10.0**exponent for sign in [-1,1] for exponent in exponents],key=abs):
        if not limits[0] < value < limits[1]:
            continue
        u=np.arcsinh(value/width)
        if all(abs(u-np.arcsinh(existing/width))>=minimum_spacing for existing in ticks):
            ticks.append(value)
    return sorted(ticks)


def decade_label(value, position):
    if value==0:
        return '0'
    exponent=int(round(np.log10(abs(value))))
    return ('$-10^{%d}$' if value<0 else '$10^{%d}$') % exponent


def make_comparison(arms, title, filename, original, updated, truth, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white', 'text.usetex': False})
    fig = plt.figure(figsize=(16, 9.2), facecolor='white')
    grid = fig.add_gridspec(3, 7, width_ratios=[1,1,1,.3,1,1,1],
                            left=.067, right=.982, bottom=.12, top=.805,
                            hspace=.40, wspace=.48)
    fig.suptitle(title, y=.98, fontsize=18, fontweight='semibold', color='#15283b')
    fig.text(.5,.935,'256,000 measurements per observable/preparation setting; exact saved SPAM calibration',
             ha='center', fontsize=10.5, color='#465668')
    fig.text(.285,.842,'EXISTING RECONSTRUCTION',ha='center',fontsize=12,
             fontweight='bold',color='#465668')
    fig.text(.765,.842,'WITH ANALYTIC MEASUREMENT SHOT NOISE',ha='center',fontsize=12,
             fontweight='bold',color='#465668')
    handles=[Line2D([0],[0],color='#1c242d',linestyle='--',lw=1.2,label='Input model')]
    for arm in arms:
        color,marker,label=STYLE[arm]
        handles.append(Line2D([0],[0],color=color,marker=marker,linestyle='none',
                              markersize=5,label=label))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.916),
               ncol=len(handles),frameon=False,fontsize=10,columnspacing=2.0)
    w=np.asarray(original[arms[0]]['wk'])
    fine=np.sort(np.unique(np.r_[np.linspace(0,w.max()*1.02,2800),w]))
    offsets=np.linspace(-.16,.16,len(arms))*(w[1]-w[0]) if len(arms)>1 else [0]
    validation=[]
    for p,(key,component,label) in enumerate(PANELS):
        row,col=divmod(p,3)
        model=component(np.asarray(truth[key](fine)))
        reference_harmonics=np.abs(component(np.asarray(truth[key](w[1:]))))
        nonzero=reference_harmonics[reference_harmonics>1e-20]
        linear_width=max(float(np.median(nonzero))/8 if len(nonzero) else 1e-10,1e-12)
        lows=[model.min(),0.0]; highs=[model.max(),0.0]
        for data in (original,updated):
            for arm in arms:
                values=component(data[arm][key]); error=np.abs(component(data[arm][key+'_errtot']))
                lows.append(float(np.min(values-error))); highs.append(float(np.max(values+error)))
        low,high=min(lows),max(highs)
        transformed=np.arcsinh(np.array([low,high])/linear_width)
        span=max(float(np.diff(transformed)[0]),1)
        limits=linear_width*np.sinh(transformed+np.array([-.075,.075])*span)
        pair=[]
        for side,data in enumerate((original,updated)):
            ax=fig.add_subplot(grid[row,col+(4 if side else 0)])
            ax.plot(fine,model,'--',color='#1c242d',lw=1.05,zorder=2)
            ax.axhline(0,color='#aab3bd',lw=.65,zorder=0)
            for offset,arm in zip(offsets,arms):
                color,marker,_=STYLE[arm]
                values=component(data[arm][key]); error=np.abs(component(data[arm][key+'_errtot']))
                x=np.asarray(data[arm]['wk'])+offset
                ax.errorbar(x,values,yerr=error,fmt='none',ecolor=color,
                            elinewidth=.85,capsize=1.8,alpha=.78,zorder=3)
                ax.scatter(x[1:],values[1:],s=14,marker=marker,facecolors=color,
                           edgecolors=color,linewidths=.5,zorder=4)
                ax.scatter(x[:1],values[:1],s=22,marker=marker,
                           facecolors=color if bool(data[arm][key+'_dc_ok']) else 'white',
                           edgecolors=color,linewidths=.9,zorder=5)
            ax.set_yscale('asinh',linear_width=linear_width)
            ax.set_ylim(*limits);ax.set_xlim(-.022,w.max()*1.035)
            ax.yaxis.set_major_locator(FixedLocator(signed_ticks(limits,linear_width)))
            ax.yaxis.set_major_formatter(FuncFormatter(decade_label))
            ax.set_title(label,pad=6)
            ax.grid(axis='y',alpha=.2,lw=.5)
            ax.yaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticks([0,.2,.4,.6,.8])
            if row==2: ax.set_xlabel(r'$\omega\tau$')
            else: ax.tick_params(labelbottom=False)
            if col==0: ax.set_ylabel(r'$S(\omega)/\tau$')
            pair.append(ax)
        assert np.array_equal(pair[0].get_ylim(),pair[1].get_ylim())
        assert np.array_equal(pair[0].get_xlim(),pair[1].get_xlim())
        validation.append(dict(spectrum=key,component='real' if component is np.real else 'imag',
                               y_limits=limits.tolist(),asinh_linear_width=linear_width))
    fig.text(.5,.063,'Bars: +/-1 quoted total uncertainty (statistical + systematic), matching the showcase figure convention.',
             ha='center',fontsize=10,color='#24384b')
    fig.text(.5,.035,'Paired panels use identical signed axes. Open DC markers indicate fits flagged undetermined. SPAM-robust is excluded.',
             ha='center',fontsize=9,color='#526172')
    path=output/(filename+'.png')
    fig.savefig(path,dpi=220)
    plt.close(fig)
    return dict(image=path.name,title=title,panels=validation)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,default=ROOT/'reports/decision30_analytic_shots_2026-10-25')
    args=parser.parse_args()
    analysis=args.analysis.resolve();output=analysis/'plots';output.mkdir(exist_ok=True)
    checks=json.loads((analysis/'original_data_hashes.json').read_text())
    for path,digest in checks.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
            raise ValueError('Original data changed: '+path)
    sys.path.insert(0,str(analysis/'_pinned_runtime/src'))
    import jax
    jax.config.update('jax_enable_x64',True)
    from qns2q.noise.spectra import S_11,S_22,S_1212,S_1_2,S_1_12,S_2_12
    truth=dict(zip(('S11','S22','S1212','S12','S112','S212'),
                   (S_11,S_22,S_1212,S_1_2,S_1_12,S_2_12)))
    original={arm:read_npz(analysis/run/'baseline/pinned_specs.npz') for arm,run in RUNS.items()}
    updated={arm:read_npz(analysis/run/'specs.npz') for arm,run in RUNS.items()}
    outputs=[]
    for arm,title in [('capture','Capture reconstruction'),
                      ('reference','SPAM-free reference'),('raw','Raw SPAM reconstruction'),
                      ('mitigated','SPAM-mitigated reconstruction')]:
        outputs.append(make_comparison([arm],title,arm+'_comparison',original,updated,truth,output))
    outputs.insert(1,make_comparison(['reference','raw','mitigated'],
                                   'SPAM comparison: reference, raw and mitigated',
                                   'spam_comparison',original,updated,truth,output))
    (output/'plot_manifest.json').write_text(json.dumps(dict(
        uncertainty_multiplier=1,error_field='*_errtot',spatial_offsets='SPAM arms only, for visibility',
        axes='paired asinh axes with common limits; negative bounds retained',
        original='pinned original summaries, redrawn on matching signed axes',
        revised='analytic multinomial measurement errors, second-order propagation',outputs=outputs),indent=2)+'\n')
    print('Created '+str(len(outputs))+' comparison images in '+str(output))


if __name__=='__main__':
    main()
