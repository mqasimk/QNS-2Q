"""Independent deterministic checks for the repetition forecast."""
import importlib.util
from pathlib import Path
import numpy as np
import numpy.testing as npt
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/forecast_repetition_sensitivity.py'
spec=importlib.util.spec_from_file_location('forecast',path)
f=importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


@pytest.mark.parametrize('count',[1,3,16,137])
def test_repetition_factor_matches_explicit_sum(count):
    x=np.array([0,1e-12,.123,np.pi,2*np.pi,6*np.pi+1e-10])
    reference=np.exp(1j*np.arange(count)[:,None]*x).sum(axis=0)
    npt.assert_allclose(f.geometric_sum(x,count),reference,atol=2e-10,rtol=1e-11)


def test_gaussian_channel_matches_two_qubit_yy_closed_form():
    covariance=np.array([[.3,.07,.01],[.07,.2,-.02],[.01,-.02,.4]])
    transfer=f.gaussian_transfer(covariance)
    params=dict(a_sp=np.ones(2),c=np.zeros(2),CM=np.eye(4))
    p=f.STAT.readout_probabilities(transfer,params,'pp','yy')
    actual=f.STAT.OUTCOME_SIGNS[0]@p
    expected=np.exp(-(.3+.2)/2)*np.sinh(.07)
    npt.assert_allclose(actual,expected,atol=1e-15)
    assert np.linalg.eigvalsh(transfer).min()>-1e-14


def test_block_covariance_sum_matches_full_filter_quadrature():
    bins,bands=1024,4
    w=np.linspace(0,bands*2*np.pi/f.T,bins*bands+1)
    filt=f.ideal_block_filters(('CPMG','CDD1-1/4'),3,w)
    smat=np.zeros((3,3,len(w)),complex)
    for a in range(3): smat[a,a]=(a+1)/(1+(13*w)**2)
    smat[0,1]=smat[1,0]=.1/(1+(7*w)**2)
    repetitions=np.array([1,2,10,32,75])
    actual,_=f.covariance_sweep(smat,w,filt,bins,bands,repetitions)
    for i,m in enumerate(repetitions):
        full=filt*f.geometric_sum(w*f.T,m)
        reference=np.array([[np.trapezoid(np.real(smat[a,b]*full[a]*full[b].conj()),w)/np.pi
                             for b in range(3)] for a in range(3)])
        npt.assert_allclose(actual[i],reference,rtol=1e-10,atol=1e-10)


def test_pulse_cycle_filter_matches_direct_midpoint_integration():
    w=np.array([0,.01,.13,.4,.83])
    k=3
    pulse=('CDD3','CPMG')
    # Directly integrate the piecewise toggles across all three cycles.
    n=960000
    t=(np.arange(n)+.5)*f.T/n
    phase=(t/(f.T/k))%1
    y1=(-1.)**(np.searchsorted(f.PULSES[pulse[0]],phase,side='right')-1)
    y2=(-1.)**(np.searchsorted(f.PULSES[pulse[1]],phase,side='right')-1)
    direct=np.array([np.sum(y[:,None]*np.exp(1j*t[:,None]*w),axis=0)*f.T/n
                     for y in (y1,y2,y1*y2)])
    npt.assert_allclose(f.ideal_block_filters(pulse,k,w),direct,rtol=1e-7,atol=2e-8)
