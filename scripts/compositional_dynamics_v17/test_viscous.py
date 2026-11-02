import numpy as np
from scipy.spatial.transform import Rotation
from scripts.compositional_dynamics_v17.viscous import ViscousMidpoint
from scripts.compositional_dynamics_v17.dynamics import DynamicContactModel
from scripts.compositional_dynamics_v17.materials import StableNeoHookean
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec

def engine():return ViscousMidpoint(DynamicContactModel(GeometrySpec(dimensions=(.12,.06,.06)),n=1,order=3,support_mode='free',bulk_energy=StableNeoHookean(12000,.35),density=625))
def test_objective_dissipation_and_internal_balance():
 e=engine();X=e.model.rest;u=X@np.diag([.03,-.02,-.08]);x=X+u;v=x@np.array([[1,.2,.3],[.1,-.5,.4],[0,.2,.7]])
 f,p=e.viscous(u,v);assert p>0
 np.testing.assert_allclose(-np.sum(f*v),p,rtol=1e-12,atol=1e-13)
 np.testing.assert_allclose(f.sum(0),0,atol=1e-12);np.testing.assert_allclose(np.cross(x,f).sum(0),0,atol=1e-12)
 R=Rotation.from_rotvec([.3,-.2,.1]).as_matrix();fr,pr=e.viscous(x@R.T+np.array([.1,.2,.3])-X,v@R.T)
 np.testing.assert_allclose(fr,f@R.T,atol=1e-11);np.testing.assert_allclose(pr,p,rtol=1e-12)
 rigid=np.cross([1.,2.,3.],x)+[.2,.4,-.1];ff,pp=e.viscous(u,rigid)
 assert np.max(abs(ff))<1e-12 and pp<1e-20

def test_midpoint_order_and_energy_plus_dissipation():
 outputs=[]
 for dt in [80e-6,40e-6,20e-6]:
  e=engine();u=np.zeros_like(e.model.rest);u[:,2]=-.01*e.model.rest[:,2];s=e.initial_state(u=u);E=e.energy(s);errors=[]
  for _ in range(round(.004/dt)):
   s=e.step(s,dt);errors.append(abs(e.energy(s)+e.dissipated_J-E)/E)
  outputs.append((s.u,max(errors)));assert e.dissipated_J>0
 a=np.linalg.norm(outputs[0][0]-outputs[1][0]);b=np.linalg.norm(outputs[1][0]-outputs[2][0]);assert a/b>3.
 assert outputs[0][1]/outputs[1][1]>2.5 and outputs[1][1]/outputs[2][1]>2.5
 assert outputs[-1][1]<.001

def test_viscous_runner_energy_ledger(tmp_path,monkeypatch):
 import json,time
 from scripts.compositional_dynamics_v17 import viscous_development as run
 from scripts.compositional_dynamics_v17.test_stronger_development import parameters,striker
 p=parameters();p.update(eta_Pa_s=20.,zeta_Pa_s=40.,integrator='explicit midpoint');c=striker();c['dt_s']=.0001
 plan=tmp_path/'plan.json';plan.write_text(json.dumps(p));monkeypatch.setattr(run,'PLAN',plan);monkeypatch.setattr(run,'RAW',tmp_path/'out');monkeypatch.setattr(run,'ROOT',tmp_path)
 r=run.worker((c,time.monotonic()+10));assert r['completed']and not r['reference_qualified']
 with np.load(tmp_path/r['path']/'trajectory.npz')as d:
  assert d['ledger'].shape[1]==24
  np.testing.assert_allclose(d['ledger'][:,23],d['ledger'][:,1]+d['ledger'][:,22])
  assert (np.diff(d['ledger'][:,22])>=0).all()
 assert r['maximum_relative_energy_balance_error']<1e-12
