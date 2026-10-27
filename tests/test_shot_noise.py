"""Deterministic checks of measurement statistics, mitigation and derivatives."""
import numpy as np
import numpy.testing as npt
import pytest

from qns2q.characterize.shot_noise import (
    Acquisition, acquisition_plan, analyze_acquisition, coefficient_derivatives,
    coefficient_maps, multinomial_covariance, phase_transfer,
    quadratic_estimator_moments, piecewise_trapezoid_filter,
)


def test_pauli_variance_includes_projection_with_no_classical_variance():
    p = np.array([.8, .2, 0, 0])
    signs = np.array([1, -1, -1, 1])
    covariance = multinomial_covariance(p, 100)
    npt.assert_allclose(signs @ covariance @ signs, (1 - .6**2) / 100)
    npt.assert_allclose(covariance.sum(axis=0), 0, atol=1e-17)
    assert np.linalg.eigvalsh(covariance).min() > -1e-16


def test_total_variance_identity_for_different_conditional_probabilities():
    conditional = np.array([[.8,.2,0,0], [.3,.7,0,0], [.5,.5,0,0]])
    p = conditional.mean(axis=0)
    within = np.mean([multinomial_covariance(q, 1) for q in conditional], axis=0)
    deviations = conditional - p
    between = deviations.T @ deviations / len(conditional)
    npt.assert_allclose(multinomial_covariance(p, 1), within + between, atol=1e-16)


@pytest.mark.parametrize('kind', ['sum', 'diff'])
@pytest.mark.parametrize('qubit', [0, 1, 2])
def test_estimator_derivatives_against_independent_finite_differences(kind, qubit):
    x = np.array([.78, .21, .12, -.08]) if not qubit else np.array([.78,.11,.69,-.07])
    div = np.array([.99,.98])
    value, gradient, hessian = coefficient_derivatives(x, kind, qubit, div)
    maps = coefficient_maps(qubit, div)
    signs = [1, 1 if kind == 'sum' else -1]

    def direct(z):
        return sum(-sign * .25 * np.log(np.sum((matrix @ z)**2))
                   for sign, matrix in zip(signs, maps))

    npt.assert_allclose(value, direct(x), atol=1e-15)
    step = 1e-5
    fd_gradient = np.zeros(4)
    fd_hessian = np.zeros((4,4))
    for i in range(4):
        ei = np.eye(4)[i] * step
        fd_gradient[i] = (direct(x+ei)-direct(x-ei))/(2*step)
        for j in range(4):
            ej = np.eye(4)[j] * step
            fd_hessian[i,j] = (direct(x+ei+ej)-direct(x+ei-ej)
                               -direct(x-ei+ej)+direct(x-ei-ej))/(4*step**2)
    npt.assert_allclose(gradient, fd_gradient, rtol=1e-7, atol=1e-8)
    npt.assert_allclose(hessian, fd_hessian, rtol=1e-4, atol=1e-6)


def test_mitigation_amplifies_physical_readout_noise_and_recovers_the_mean():
    base = dict(spam_protocol=np.array('none'), a_sp=np.ones(2), c=np.zeros(2,complex),
                c_prep=np.zeros(2,complex), CM=np.eye(4), spMit=np.array(False))
    # A deterministic ensemble: no new random noise realizations.
    phases = np.array([[.1,.03,.01],[-.1,-.03,-.01],[.2,-.02,.03],[-.2,.02,-.03]])
    transfers = [phase_transfer(phases)] * 4
    acquisition = Acquisition('test','sum',0,0,0,1,False)
    ideal = analyze_acquisition(acquisition,transfers,base,256000)
    confusion1 = np.array([[.96,.04],[.04,.96]])
    cm = np.kron(confusion1, confusion1)
    mitigated = dict(base, spam_protocol=np.array('mitigated'), CM=cm, est_cm=cm,
                     spMit=np.array(True), a_sp_div=np.ones(2))
    corrected = analyze_acquisition(acquisition,transfers,mitigated,256000)
    npt.assert_allclose(corrected['value'],ideal['value'],atol=1e-14)
    assert corrected['stderr'] > ideal['stderr']


def test_plan_matches_pinned_cache_lengths_and_rejects_robust():
    for repetitions, calls in [(16,1104),(10,960)]:
        plan=list(acquisition_plan(dict(M=repetitions,truncate=20,spam_protocol='none')))
        assert sum(a.width for a in plan)==calls
        assert plan[-1].first_call+plan[-1].width==calls
        assert len(plan)==240+10*repetitions
    with pytest.raises(ValueError, match='robust'):
        list(acquisition_plan(dict(M=10,truncate=20,spam_protocol='robust')))


@pytest.mark.parametrize('shots',[0,-1,1.5])
def test_invalid_shot_budgets_are_rejected(shots):
    with pytest.raises(ValueError):
        multinomial_covariance(np.ones(4)/4,shots)


def test_quadratic_moments_match_exact_binomial_enumeration():
    from math import comb
    shots = 4
    probabilities = np.array([[.8,.2,0,0],[.3,.7,0,0],[1,0,0,0],[1,0,0,0]])
    weights = np.array([1,-1,1,-1])
    means = probabilities @ weights
    gradient = np.array([.4,-.2,.1,.3])
    hessian = np.array([[.5,.2,0,0],[.2,-.1,0,0],[0,0,0,0],[0,0,0,0]])
    _, variance, bias = quadratic_estimator_moments(probabilities,weights,shots,gradient,hessian)
    values, masses = [], []
    for k in range(shots+1):
        for l in range(shots+1):
            delta = np.array([2*k/shots-1,2*l/shots-1,1,1])-means
            values.append(gradient@delta+.5*delta@hessian@delta)
            masses.append(comb(shots,k)*.8**k*.2**(shots-k)
                          *comb(shots,l)*.3**l*.7**(shots-l))
    values,masses = np.asarray(values),np.asarray(masses)
    mean = masses@values
    npt.assert_allclose(bias,mean,atol=1e-15)
    npt.assert_allclose(variance,masses@(values-mean)**2,atol=1e-15)


@pytest.mark.parametrize('sign',[-1,1])
def test_segmented_filter_matches_direct_trapezoids(sign):
    times = np.linspace(.13,16.73,1601)
    toggle = np.ones(len(times)); toggle[200:431]=-1; toggle[712:950]=0
    dt=times[1]-times[0]
    frequencies=np.array([0,.00001,.13,1.7,2*np.pi/dt])
    actual=piecewise_trapezoid_filter(toggle,times,frequencies,sign)
    expected=np.trapezoid(toggle*np.exp(1j*sign*frequencies[:,None]*times),times,axis=1)
    npt.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)


def test_postprocessor_refuses_original_run_output():
    from pathlib import Path
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    original=root/'DraftRun_SPAM_showcase_robust'
    result=subprocess.run([sys.executable,str(root/'scripts/postprocess_shot_noise.py'),
                           '--output',str(original)],capture_output=True,text=True)
    assert result.returncode != 0
    assert 'separate from the original run folders' in result.stderr
