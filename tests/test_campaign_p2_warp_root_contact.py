"""Manufactured CPU-only Warp force/E candidate, no cohort/trajectory reads."""
import os,sys
from pathlib import Path
import numpy as np
import pytest
from experiments.benchmark_p2_warp_rhs import ENV_SITE
sys.path.append(str(ENV_SITE))
import warp as wp
from experiments.campaign_p2_warp_root_contact import WarpRootContact
from experiments.campaign_p2_contact_root_fixed import integrate_face_fixed
from experiments.campaign_p2_contact_root_integration import integrate_face
N=np.array([[0.,0.],[1,0],[0,1],[.5,0],[0,.5],[.5,.5]])

def fixture():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))==2
 wp.config.kernel_cache_dir=str(Path(__file__).resolve().parents[1]/'runs/operator-dev/p2-warp-root-contact-v01/warp-cpu-cache');X=np.c_[.2*N[:,0],.3*N[:,1],np.full(6,.1)];obj=WarpRootContact(rest=X,surface_elements=np.arange(6)[None],reference_areas=np.array([.03]),kappa_A=1500,device='cpu');return X,obj

def test_CPU_force_E_vs_fixed64_and_original_adaptive_pathologies_torque():
 X,obj=fixture()
 functions=[lambda r,s:0*r,lambda r,s:0*r+.1,lambda r,s:0*r-.001,lambda r,s:r-.31+1e-15*s*s,lambda r,s:(r-.3)**2,lambda r,s:(r-.3)**2+(s-.3)**2-.04,lambda r,s:s-.5*r,lambda r,s:(r-.05)**2+(s-.15)**2-.0001]
 for f in functions:
  u=np.zeros_like(X);u[:,2]=f(N[:,0],N[:,1])-X[:,2];h=(X+u)[:,2];a=obj.evaluate(u);b=integrate_face_fixed(h,.03,kappa_A=1500,outer_order=64);c=integrate_face(h,.03,kappa_A=1500)
  for ref in [b,c]:
   assert abs(a['contact_energy']-ref['energy_J'])<=1e-12
   np.testing.assert_allclose(a['gradient'][:,2],ref['gradient_N'],atol=1e-10,rtol=0)
   np.testing.assert_allclose(np.cross(X,-a['gradient']).sum(0),np.cross(X,np.c_[np.zeros((6,2)),-ref['gradient_N']]).sum(0),atol=1e-11,rtol=1e-12)
  np.testing.assert_array_equal(a['gradient'][:,:2],0.)
  assert a['production_Hessian'] is False


def test_CPU_energy_gradient_FD_and_contract():
 X,obj=fixture();u=np.zeros_like(X);u[:,2]=(N[:,0]-.3)**2+(N[:,1]-.3)**2-.04-X[:,2];d=np.random.default_rng(47).normal(size=X.shape);d/=np.linalg.norm(d);a=obj.evaluate(u);steps=[1e-5,3e-6,1e-6,3e-7,1e-7,3e-8,1e-8];errors=[]
 for eps in steps:
  p=obj.evaluate(u+eps*d);m=obj.evaluate(u-eps*d);errors.append(abs((p['contact_energy']-m['contact_energy'])/(2*eps)-np.sum(a['gradient']*d)))
 assert min(errors)<1e-9 and max(errors[-3:])<1e-8
 assert min(errors[2:])<errors[0]*.05
 print({'energy_force_FD_steps_m':steps,'directional_errors_N':errors,'claim':'FiniteouterE/g derivative agreement controlled numerically, not automatically algebraicconservative.'})
 with pytest.raises(ValueError):obj.evaluate(u.astype(np.float32))
 with pytest.raises(ValueError):WarpRootContact(rest=X,surface_elements=np.arange(6)[None],reference_areas=np.array([.03]),outer_order=32)
