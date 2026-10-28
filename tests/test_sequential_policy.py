import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve
from scripts.phystwin.sequential_policy import Experiment,BOUNDS,field_indices
from scripts.phystwin.active_support_policy import Mixture

def fixture():
 axes=[np.linspace(*b,4) for b in BOUNDS];theta=np.stack(np.meshgrid(*axes,indexing='ij'),-1);rng=np.random.default_rng(12);linear=rng.normal(size=(3,270))*1e-4
 data=(theta-theta.mean((0,1,2)))@linear
 interp=[RegularGridInterpolator(axes,data+i*1e-5,method='cubic',solver=spsolve) for i in range(3)]
 return Experiment(interp,5,np.zeros(270),np.eye(270)*1e-12,.00015,device='cpu')

def test_exact_three_parameter_tensor_mean_matches_dense():
 ex=fixture();m=ex.model([1,4]);rng=np.random.default_rng(7);y=rng.normal(size=(3,len(ex.observed([1,4]))))*.0001
 structured=m.infer(y);basis=m.tensor_basis;m.tensor_basis=None;dense=m.infer(y);m.tensor_basis=basis
 for key in ['mean','sd','log_parameter_quantiles','layout_probability']:assert np.allclose(structured[key],dense[key],rtol=1e-9,atol=1e-12)
 assert np.allclose(structured['layout_probability'].sum(1),1)
 assert np.all(np.diff(structured['log_parameter_quantiles'],axis=2)>=0)

def test_history_is_canonical_and_no_probe_repetition():
 ex=fixture();assert ex.model([4,1]) is ex.model([1,4]);y=np.zeros(27)
 first,r=ex.choose([],y,16);second,r=ex.choose([first],y,16)
 assert first!=second and np.isinf(r[first])
 assert len(field_indices())==270 and len(np.unique(field_indices()))==270
 assert np.isfinite(ex.predict([first,second],y)['mean']).all()

def test_shared_field_implementation_matches_original_dense_mixture():
 ex=fixture();history=[0,3];obs=ex.observed(history);m=ex.model(history)
 old=Mixture(ex.fields,ex.prior,ex.bias,ex.C,obs,ex.target,0.,'cpu');y=np.ones((2,len(obs)))*1e-4
 assert np.allclose(m.weights(y),old.weights(y),rtol=1e-10,atol=1e-13)
 assert np.allclose(m.risk(y),old.risk(y),rtol=1e-9,atol=1e-15)
 expected=old.weights(y)@old.M+old.tensor(y)@old.gain.T
 assert np.allclose(m.infer(y)['mean'],expected.numpy(),rtol=1e-9,atol=1e-12)
