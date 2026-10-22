"""Manufactured CPU-only HVP checks; no cohort/trajectory arrays or GPU calls."""
import os,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.benchmark_p2_warp_rhs import ENV_SITE
sys.path.append(str(ENV_SITE))
import warp as wp
from experiments.campaign_p2_warp_bulk_hvp import WarpBulkHVP
from experiments.campaign_p2_streaming import StreamingTwoLevelP2
from experiments.campaign_p2_exact_quadrature import ExactBulkP2Geometry
from experiments.campaign_p2_keast_quadrature import KeastBulkP2Geometry

@pytest.mark.parametrize('geometry',[KeastBulkP2Geometry,ExactBulkP2Geometry])
def test_fullHVP_streaming_parity_FD_symmetry_and_zero(geometry):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert len(os.sched_getaffinity(0))==2
 torch.set_num_threads(2);wp.config.kernel_cache_dir=str(Path(__file__).resolve().parents[1]/'runs/operator-dev/p2-warp-bulk-hvp-v01/warp-cpu-cache')
 m=geometry((1,1,1),young=4500,pattern='axial_pull');ref=StreamingTwoLevelP2(m,quadrature_chunk_size=7)
 obj=WarpBulkHVP(elements=m.elements,linear_gradients=m.linear_gradients,bary=m.bary,weights=m.weights,volumes=m.volumes,node_count=len(m.rest),young=m.young,poisson=m.poisson,device='cpu')
 X=m.rest;u=np.column_stack([.04*X[:,0]+.3*X[:,1]*X[:,2],-.02*X[:,1]+.1*X[:,0]*X[:,2],.03*X[:,2]+.2*X[:,0]*X[:,1]])
 rng=np.random.default_rng(19);v=rng.normal(size=X.shape);v/=np.linalg.norm(v);w=rng.normal(size=X.shape);w/=np.linalg.norm(w)
 t=lambda x:torch.tensor(x,dtype=torch.float64);cache=ref.evaluate(t(u));actual=obj.tangent(u[m.elements],v);expected=ref.tangent(cache,t(v)).numpy()
 np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=2e-12)
 eps=2e-7;cp=ref.evaluate(t(u+eps*v));cm=ref.evaluate(t(u-eps*v));fd=(cp['gradient']-cm['gradient']).numpy()/(2*eps)
 np.testing.assert_allclose(actual,fd,atol=2e-6,rtol=2e-8)
 ed=(float(cp['energy'])-2*float(cache['energy'])+float(cm['energy']))/eps**2
 assert abs(ed-np.sum(actual*v))<2e-3
 Hw=obj.tangent(u[m.elements],w);assert abs(np.sum(w*actual)-np.sum(v*Hw))<2e-11
 np.testing.assert_array_equal(obj.tangent(u[m.elements],np.zeros_like(v)),0.)
 # Stored geometry has no Q*E*10*3 shape-gradient array.
 assert set(obj.data)=={'elements','linear_gradients','bary','weights','volumes'}
 assert obj.geometry_bytes==sum(x.nbytes for x in [m.elements,m.linear_gradients,m.bary,m.weights,m.volumes])
 print({'rule':m.order if hasattr(m,'order') else str(geometry),'Q':len(m.bary),'HVP_L2_error_N':float(np.linalg.norm(actual-expected)),'FD_L2_error_N':float(np.linalg.norm(actual-fd)),'symmetry_error':abs(np.sum(w*actual)-np.sum(v*Hw)),'geometry_bytes':obj.geometry_bytes})


def test_explicit_FP64_shapes_and_material_contract():
 m=KeastBulkP2Geometry((1,1,1),young=4500,pattern='axial_pull');args=dict(elements=m.elements,linear_gradients=m.linear_gradients,bary=m.bary,weights=m.weights,volumes=m.volumes,node_count=len(m.rest),young=m.young,poisson=m.poisson,device='cpu');obj=WarpBulkHVP(**args);u=np.zeros((len(m.elements),10,3));v=np.zeros_like(m.rest)
 with pytest.raises(ValueError,match='float64'):obj.tangent(u.astype(np.float32),v)
 with pytest.raises(ValueError,match='direction'):obj.tangent(u,v[:-1])
 with pytest.raises(ValueError):WarpBulkHVP(**{**args,'young':0})
 with pytest.raises(ValueError):WarpBulkHVP(**{**args,'weights':-m.weights})
