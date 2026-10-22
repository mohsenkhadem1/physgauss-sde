"""CPU manufactured corrected combined E/g/guards/fullM control."""
import os,sys
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from experiments.benchmark_p2_warp_rhs import ENV_SITE
sys.path.append(str(ENV_SITE))
import warp as wp
from experiments.campaign_p2_warp_reference_force import WarpReferenceForce
from experiments.run_p2_contact_validation import build
from experiments.diagnose_p2_contact_bulk_quadrature import response
from experiments.campaign_p2_contact_root_integration import integrate_face
from experiments.campaign_p2_consistent_transient import scalar_mass

def independent(m,obj,u):
 a=obj.arrays;elastic=response(m,u,a['bary'],a['weights'][:,0]/m.volumes[0],a['guard_bary']);g=elastic['gradient_N'].copy();energy=elastic['energy_J'];q=m.rest+u;faces=obj.surface.faces.numpy();areas=obj.surface.areas.numpy()
 for face,A in zip(faces,areas):
  r=integrate_face(np.ascontiguousarray(q[face,2]),float(A),kappa_A=obj.surface.kappa_A);energy+=r['energy_J'];np.add.at(g[:,2],face,r['gradient_N'])
 return energy,g,elastic['common_guard_minimum_J']

def test_combined_manufactured_CPU_parity_fullM_R_and_all141_guards():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))==2
 wp.config.kernel_cache_dir=str(Path(__file__).resolve().parents[1]/'runs/operator-dev/p2-warp-reference-force-v01/warp-cpu-cache');m=build([1,1,1],300000.,9.81);obj=WarpReferenceForce(m,device='cpu');X=m.rest;u=np.column_stack([.04*X[:,0]+.3*X[:,1]*X[:,2],-.02*X[:,1]+.1*X[:,0]*X[:,2],-.051+.002*(X[:,0]/.24)**2]);a=obj.evaluate(u);E,g,J=independent(m,obj,u)
 assert abs(a['energy']-E)<1e-12 and np.linalg.norm(a['gradient']-g)<1e-10 and abs(a['minimum_J']-J)<2e-14
 M=scalar_mass(m).toarray();F=M@np.ones_like(m.rest)*np.array([0.,0.,-9.81]);np.testing.assert_allclose(F,m.load,atol=1e-13,rtol=1e-13);L=cho_factor(M);acc=cho_solve(L,F-a['gradient']);assert np.linalg.norm(M@acc-(F-a['gradient']))<1e-8
 assert a['guard_points']==141 and not a['production_Hessian']
 bad=np.zeros_like(u);bad[:,0]=-2*m.rest[:,0]
 try:obj.evaluate(bad)
 except ValueError as e:assert 'Jacobian' in str(e)
 else:raise AssertionError('OriginalJguardmustreject')
