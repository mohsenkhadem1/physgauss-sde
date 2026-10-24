import numpy as np
import pytest,subprocess,json
from experiments import diagnose_surface_cdf_v01 as d
from experiments.campaign_surface_conditional_amplitude_v01 import ConditionalAmplitudeReference
from scipy.special import logsumexp

def test_selected_prefix_row_and_allowlist(tmp_path):
    f=tmp_path/'toy.npz';a=np.arange(240,dtype=float).reshape(10,24);n=np.arange(24,dtype=float)
    np.savez_compressed(f,matching_linear=a,prefix_noise=n,forbidden_future=np.ones(100))
    assert np.array_equal(d.selected_npy_row(f,'matching_linear.npy',4),a[4])
    assert np.array_equal(d.selected_npy_row(f,'prefix_noise.npy'),n)
    with pytest.raises(ValueError,match='allowlist'):d.selected_npy_row(f,'forbidden_future.npy')
    with pytest.raises(ValueError,match='row required'):d.selected_npy_row(f,'matching_linear.npy',20)

def test_complement_and_raw_CDF_closure_against_exact_uniform_logprior():
    def zero(z,labels):return np.zeros((len(z),len(labels),1))
    e=ConditionalAmplitudeReference(None,[('prefix',np.zeros(1))],['prefix'],log_ratio_predict=zero)
    full=float(logsumexp(e._nodes(64,32)[3]))
    for dim in(0,1):
        for u in(.03125,.3125,.75,.96875):
            low=float(logsumexp(e._nodes(64,32,(dim,u))[3]));high=d.log_complement(e,dim,u,64,32)
            assert abs(np.exp(low-full)-u)<1e-12
            assert abs(np.exp(high-full)-(1-u))<1e-12
            assert abs(np.logaddexp(low,high)-full)<1e-12

def test_actual_guard_command():
    x=subprocess.run(d.command('0'*64,guard_only=True),cwd=d.ROOT,capture_output=True,text=True,timeout=20)
    assert x.returncode==0,x.stderr
    assert json.loads(x.stdout)['guard_passed']
