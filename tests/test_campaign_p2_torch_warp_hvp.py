"""CPU interop/linear-direction controls only; no GPU/cohort arrays."""
import os,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.benchmark_p2_warp_rhs import ENV_SITE
sys.path.append(str(ENV_SITE))
import warp as wp
from experiments.campaign_p2_torch_warp_hvp import TorchWarpBulkHVPMixin
from experiments.campaign_p2_streaming import StreamingTwoLevelP2
from experiments.campaign_p2_keast_quadrature import KeastBulkP2Geometry
from experiments.campaign_p2_geometry import P2Geometry

class Fast(TorchWarpBulkHVPMixin,StreamingTwoLevelP2):pass

@pytest.mark.parametrize('geometry',[P2Geometry,KeastBulkP2Geometry])
def test_CPU_alias_storage_HVP_and_same_CG_direction_original_residual(geometry):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))==2
 torch.set_num_threads(2);wp.config.kernel_cache_dir=str(Path(__file__).resolve().parents[1]/'runs/operator-dev/p2-torch-warp-hvp-v01/warp-cpu-cache')
 m=geometry((1,1,1),young=4500,pattern='axial_pull');a=Fast(m);b=StreamingTwoLevelP2(m)
 a.initialize_warp_hvp()
 for name,t in a._warp_hvp_tensors.items():assert a._warp_hvp_geometry[name].ptr==t.data_ptr()
 X=m.rest;u=torch.tensor(np.column_stack([.04*X[:,0]+.3*X[:,1]*X[:,2],-.02*X[:,1]+.1*X[:,0]*X[:,2],.03*X[:,2]+.2*X[:,0]*X[:,1]]));u[a.mask]=0
 ca=a.evaluate(u);cb=b.evaluate(u);rng=np.random.default_rng(2026102204);v=torch.tensor(rng.normal(size=X.shape));actual=a.tangent(ca,v);expected=b.tangent(cb,v)
 torch.testing.assert_close(actual,expected,atol=2e-10,rtol=2e-12)
 saved=actual.clone();a.tangent(ca,2*v);torch.testing.assert_close(actual,saved,atol=0,rtol=0)
 np.testing.assert_array_equal(a.tangent(ca,torch.zeros_like(v)).numpy(),0.)
 rhs=torch.tensor(rng.normal(size=X.shape)).masked_fill(a.mask,0.);factor=1000.
 x,k=a.cg(ca,rhs,factor,rtol=1e-11,atol=1e-12);y,j=b.cg(cb,rhs,factor,rtol=1e-11,atol=1e-12)
 torch.testing.assert_close(x,y,atol=2e-10,rtol=2e-9)
 oldR=(b.tangent(cb,x)+factor*b.mass(x)-rhs).masked_fill(b.mask,0.)
 assert float(torch.linalg.vector_norm(oldR))<=2e-10
 assert a.evaluate.__func__ is b.evaluate.__func__ and a.mass.__func__ is b.mass.__func__ and a.nodal_blocks.__func__ is b.nodal_blocks.__func__
 assert a.make_preconditioner.__func__ is b.make_preconditioner.__func__ and a.solve.__func__ is b.solve.__func__
 print({'Q':len(m.bary),'HVP_L2_difference_N':float(torch.linalg.vector_norm(actual-expected)),'CG_max_difference_m':float(abs(x-y).max()),'independent_old_HVP_true_R_N':float(torch.linalg.vector_norm(oldR)),'candidate_CG':k,'original_CG':j})
 with pytest.raises(ValueError):a.tangent(ca,v.float())
