"""CPU-only controlledconstructor/compatiblefield E/g comparison, no responses."""
import os,sys
from pathlib import Path
import numpy as np
from experiments.benchmark_p2_warp_rhs import ENV_SITE
sys.path.append(str(ENV_SITE))
import warp as wp
from experiments.campaign_p2_warp_reference_force import WarpReferenceForce
from experiments.campaign_p2_warp_reference_order32 import WarpReferenceForce32
from experiments.run_p2_contact_validation import build

def test_explicit32_buffers_same_bulk_guards_kernel_physics_and_E_g():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))=={30,31}
 wp.config.kernel_cache_dir=str(Path(__file__).resolve().parents[1]/'runs/operator-dev/p2-corrected-contact-order32-v01/warp-cpu-cache');m=build([1,1,1],300000.,9.81);a=WarpReferenceForce32(m,device='cpu');b=WarpReferenceForce(m,device='cpu')
 assert a.surface.E.shape[0]==32 and b.surface.E.shape[0]==64
 for k in ['rest','elements','grad','check_grad','weights','guard_bary']:np.testing.assert_array_equal(a.arrays[k],b.arrays[k])
 assert a.surface.kappa_A==b.surface.kappa_A==1e9
 assert a.minimum_J==b.minimum_J==.1
 for u in [np.zeros_like(m.rest),np.column_stack([np.zeros((len(m.rest),2)),-.051+.002*(m.rest[:,0]/.24)**2])]:
  aa=a.evaluate(u);bb=b.evaluate(u);assert aa['surface_rule']=='root32forceE' and bb['surface_rule']=='root64forceE'
  assert aa['guard_points']==bb['guard_points']==141
  assert abs(aa['energy']-bb['energy'])<=1e-12
  assert np.linalg.norm(aa['gradient']-bb['gradient'])<=1e-10
