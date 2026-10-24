"""Opt-in GPU controls; default tests never initialize a GPU."""
import os
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.campaign_surface_p2_inference_v01 import LabelledModal
from experiments.campaign_surface_gpu_modal_v01 import Float64GPUModal,FORBIDDEN_GPU_UUID


def test_forbidden_device_refused_before_cuda_initialization():
    with pytest.raises(ValueError):Float64GPUModal(None,[],physical_UUID=FORBIDDEN_GPU_UUID,allowed_UUIDs=[FORBIDDEN_GPU_UUID])


def fake_modal():
    rng=np.random.default_rng(48945);m=2976;q=24;r=10
    return SimpleNamespace(eigenvalues=rng.uniform(0,1,m),coefficients=rng.normal(size=(q*r,m)),output_shape=(q,r))


@pytest.mark.skipif(os.environ.get('SURFACE_GPU_CONTROL_UUID')is None,reason='Explicit GPU control UUID required')
def test_float64_selected_modes_independent_likelihood_and_batch_guard():
    UUID=os.environ['SURFACE_GPU_CONTROL_UUID'];labels=[f'load{i}'for i in range(10)];modal=fake_modal()
    cpu=LabelledModal(modal,labels);gpu=Float64GPUModal(modal,labels,physical_UUID=UUID,allowed_UUIDs=[UUID])
    rng=np.random.default_rng(48946);unit=rng.uniform(size=(1024,2));selected=['load8','load0','load8']
    a=cpu.select(selected)(unit);b=gpu.select(selected)(unit)
    np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-13)
    y=cpu.select(['load0'])(np.array([[.4,.6]]))[0,0];obs=[('load0',y),('load0',y+1e-5)]
    np.testing.assert_allclose(cpu.loglikelihood(obs)(unit),gpu.loglikelihood(obs)(unit),rtol=2e-10,atol=1e-5)
    with pytest.raises(ValueError):gpu.select(['load0'])(np.zeros((4097,2)))
    assert gpu.torch.cuda.max_memory_allocated()<512*1024**2
