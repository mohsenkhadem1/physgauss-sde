import os,sys
from pathlib import Path
import numpy as np
import torch
from experiments.benchmark_p2_warp_rhs import ENV_SITE
sys.path.append(str(ENV_SITE))
import warp as wp
from experiments.run_p2_torch_warp_hvp_benchmark_v02 import gates,shared_operator_cache
from experiments.campaign_p2_torch_warp_hvp import TorchWarpBulkHVPMixin
from experiments.campaign_p2_streaming import StreamingTwoLevelP2
from experiments.campaign_p2_keast_quadrature import KeastBulkP2Geometry

def test_original_cache_evaluated_ONCE_and_alias_shared_CPU_HVP():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))==2
 torch.set_num_threads(2);wp.config.kernel_cache_dir=str(Path(__file__).resolve().parents[1]/'runs/operator-dev/p2-torch-warp-hvp-benchmark-v02/warp-cpu-cache')
 class Fast(TorchWarpBulkHVPMixin,StreamingTwoLevelP2):pass
 m=KeastBulkP2Geometry((1,1,1),young=4500);old=StreamingTwoLevelP2(m);new=Fast(m);u=torch.tensor(.01*m.rest);oc,nc=shared_operator_cache(old,u);assert oc is nc and oc['local_u'].data_ptr()==nc['local_u'].data_ptr()
 v=torch.tensor(np.random.default_rng(5).normal(size=m.rest.shape));torch.testing.assert_close(new.tangent(nc,v),old.tangent(oc,v),atol=2e-10,rtol=2e-12)
 class Spy:
  calls=0
  def evaluate(self,u):self.calls+=1;return {'local_u':u}
 spy=Spy();a,b=shared_operator_cache(spy,u);assert spy.calls==1 and a is b
 r=dict(old_HVP_true_R_N=1e-11,original_candidate_free_R_N=1e-11,maximum_constraint_m=0.,minimum_J=.9,positive_Bernstein=True,maximum_candidate_vs_exact80_m=1e-9)
 assert gates(r)
 for key,value in [('old_HVP_true_R_N',1.1e-10),('original_candidate_free_R_N',1.1e-10),('maximum_constraint_m',1e-20),('minimum_J',.1),('positive_Bernstein',False),('maximum_candidate_vs_exact80_m',1.1e-8)]:assert not gates({**r,key:value})
