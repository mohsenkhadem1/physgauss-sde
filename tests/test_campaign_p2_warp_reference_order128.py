import sys
import numpy as np
from experiments.benchmark_p2_warp_rhs import ENV_SITE
sys.path.append(str(ENV_SITE))
import warp as wp
from experiments.campaign_p2_warp_reference_order128 import WarpReferenceForce128
from experiments.campaign_p2_warp_reference_force import corrected_arrays
from experiments.run_p2_contact_validation import build
from experiments.campaign_p2_contact_root_fixed import integrate_face_fixed
from experiments.diagnose_p2_contact_bulk_quadrature import response
from experiments.run_p2_corrected_contact_order128 import CONFIG,pair_metrics

def test_explicit128_buffers_independent_partial_contact_and_unchanged_bulk(tmp_path):
 wp.config.kernel_cache_dir=str(tmp_path);m=build([4,2,2],300000.,9.81);w=WarpReferenceForce128(m,device='cpu');a=corrected_arrays(m)
 for k in ['rest','elements','grad','check_grad','weights']:np.testing.assert_array_equal(w.arrays[k],a[k])
 assert w.surface.order==128 and w.surface.E.shape==(128,80) and a['check_grad'].shape[0]==141
 u=np.zeros_like(m.rest);u[:,2]=-.055+.02*(m.rest[:,0]/.24)**2
 actual=w.evaluate(u);base=response(m,u,a['bary'],a['weights'][:,0]/m.volumes[0],a['guard_bary']);E=base['energy_J'];g=base['gradient_N'].copy();Uc=0.
 for f,A in zip(w.surface.faces.numpy(),w.surface.areas.numpy()):
  r=integrate_face_fixed((m.rest+u)[f,2],float(A),kappa_A=1e9,outer_order=128);E+=r['energy_J'];Uc+=r['energy_J'];np.add.at(g[:,2],f,r['gradient_N'])
 assert Uc>0 and abs(actual['energy']-E)<=1e-12 and np.linalg.norm(actual['gradient']-g)<=1e-10
 assert actual['surface_rule']=='root128forceE'

def test_unrelaxed_pair_gate_and_fixed_tight_config():
 assert CONFIG==dict(name='outer128_accuracy_1e10',rtol=1e-10,displacement_atol_m=1e-13,velocity_atol_m_s=1e-11,impulse_atol_Ns=1e-13)
 q=np.zeros((9,2,3));v=q.copy();E=np.zeros(9);other=q.copy();other[3,0,0]=20.001e-6
 assert not pair_metrics(q,v,E,other,v,E,4.7088e-7,q,q)['passed']
 other=q.copy();vv=v.copy();vv[3,0,0]=1.00001e-4
 assert not pair_metrics(q,v,E,other,vv,E,4.7088e-7,q,q)['passed']
