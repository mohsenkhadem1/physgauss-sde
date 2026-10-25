"""CPU tensor controls only: never allocates or queries any CUDA device."""
import numpy as np
import pytest
from experiments import campaign_grouped_linear_gpu_v01 as GPU
from experiments import campaign_grouped_linear_decision_v01 as CPU

def bank():return CPU.GroupedLinearBank([.2,.3,.5],[0,0,1],[[.1,.2,.4],[.15,.3,.5],[.2,.4,.8]],[[.1,.4,.5],[.5,.3,.2],[.2,.6,.2]],[[1.,-.4],[.3,.6],[-.6,1.]],np.tile([[1.,2.,-.3],[.5,-1.,1.],[2.,.4,-.2]],(1,16)))

def test_vectorized_tensor_allatoms_conditional_and_full_outcome_parity():
    b=bank();engine=GPU.TensorGroupedControl(b,outcome_chunk=4);y=np.array([-.4,0.,.35,1.])
    for action in range(2):
        a,v=engine.outcome_risk(action,y,.1);c,w=CPU.outcome_risk(b,action,y,.1,row_chunk=1);np.testing.assert_allclose(a,c,atol=3e-14);np.testing.assert_allclose(v,w,atol=1e-15,rtol=1e-13)
        got=engine.expected_risk(action,.1,order=8);ref=CPU.expected_risk(b,action,.1,order=8,outcome_chunk=4)
        for key in('expected_target_variance','prior_target_variance','predictive_mass','omitted_risk_bound'):assert got[key]==pytest.approx(ref[key],rel=2e-13,abs=1e-15)

def test_zero_mass_and_narrow_amplitudes_no_variance_cancellation():
    b=CPU.GroupedLinearBank([1.,0.],[0,1],[[1.,1.+1e-9],[2.,3.]],[[.5,.5],[.3,.7]],[[1.],[100.]],[[2.],[2.]])
    e=GPU.TensorGroupedControl(b);lp,v=e.outcome_risk(0,np.array([1.,1.2]),.1);_,ref=CPU.outcome_risk(b,0,np.array([1.,1.2]),.1);np.testing.assert_allclose(v,ref,atol=1e-26,rtol=1e-6);assert np.all(v>=0)

def test_forbidden_unpinned_or_multiple_devices_refused_before_torch_cuda(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',GPU.FORBIDDEN_UUID)
    with pytest.raises(ValueError,match='GPU3 forbidden'):GPU.Float64GroupedGPU(bank(),physical_UUID=GPU.FORBIDDEN_UUID)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',','.join(GPU.ALLOWED_UUIDS[:2]))
    with pytest.raises(ValueError,match='Exactly one'):GPU.Float64GroupedGPU(bank(),physical_UUID=GPU.ALLOWED_UUIDS[0])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',GPU.ALLOWED_UUIDS[0]);monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG','bad')
    with pytest.raises(ValueError,match='cuBLAS'):GPU.pin_config(GPU.ALLOWED_UUIDS[0])

def test_allocation_preflight_and_batch_limits_before_vectorization():
    b=bank()
    with pytest.raises(MemoryError):GPU.TensorGroupedControl(b,allocated_cap_bytes=1)
    e=GPU.TensorGroupedControl(b,outcome_chunk=2)
    with pytest.raises(ValueError,match='Bounded'):e.outcome_risk(0,np.zeros(3),.1)
