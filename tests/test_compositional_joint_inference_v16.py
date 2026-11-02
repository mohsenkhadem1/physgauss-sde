import numpy as np
import torch
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.joint_inference import JointMAP,neutral_parameters

def problem():
 torch.set_num_threads(1)
 pose=np.zeros((2,6));pose[1,0]=.0007;pose[1,5]=.012
 return JointMAP(GeometrySpec(),bulk_family='yeoh',interface_family=1,heterogeneity=0,observations=[dict(pose=pose,points=np.array([[.2,.3,.4],[.7,.6,.5]]),displacement=np.zeros((2,3)),forces=np.zeros((2,3)),torques=np.zeros((2,3)))],position_sigma=.002,force_sigma=.2,torque_sigma=.02,n=1)

def test_joint_direct_wrench_and_geometry_adjoint_matches_refits():
 p=problem();z=neutral_parameters();z[0]=.1;z[9]=.44
 loss,grad,diag=p.value_gradient(z)
 for j in [0,5,9]:
  plus=z.copy();minus=z.copy();plus[j]+=1e-4;minus[j]-=1e-4
  fd=(p.value_gradient(plus)[0]-p.value_gradient(minus)[0])/2e-4
  np.testing.assert_allclose(grad[j],fd,rtol=1e-3,atol=2e-5)
 assert all(d['adjoint_relative_residual']<1e-8 for d in diag)

def test_subset_optimization_decreases_declared_map_objective():
 p=problem();z=neutral_parameters();z[0]=.4
 initial=p.value_gradient(z)[0]
 out=p.fit(active=[0,5],initial=z,max_iterations=3,max_evaluations=20,seconds=45)
 assert out['loss']<initial
 mask=np.ones(15,bool);mask[[0,5]]=False
 np.testing.assert_array_equal(out['parameters'][mask],z[mask])
 assert len(out['evaluations'])<=20

def test_known_controls_continuation_cache_and_budget(monkeypatch):
 import scripts.compositional_neural_mechanics.joint_inference as module
 p=problem();p.scene['gravity']=[0.,0.,-.2];p.scene['density_kg_m3']=150.
 original=module.make_model;seen=[]
 def build(scene,**kw):
  m=original(scene,**kw);solve=m.solve
  def wrapped(*args,**kwargs):
   seen.append((m.poses.copy(),kwargs['initial'] is None))
   return solve(*args,**kwargs)
  m.solve=wrapped;return m
 monkeypatch.setattr(module,'make_model',build)
 p.value_gradient(neutral_parameters())
 assert len(seen)==4 and seen[0][1] and not any(v[1] for v in seen[1:])
 for (pose,_),fraction in zip(seen,p.continuation):np.testing.assert_allclose(pose,p.observations[0]['pose']*fraction)
 assert len(p._observation_cache)==1
 monkeypatch.setattr(module,'observation_matrix',lambda *a: (_ for _ in ()).throw(AssertionError('recomputed map')))
 p.value_gradient(neutral_parameters())
 out=p.fit(active=[0],seconds=1e-12)
 assert out['incomplete'] and not out['optimizer_success'] and out['loss'] is None and out['evaluations']==[]
 out=p.fit(active=[0],max_evaluations=1,seconds=30)
 assert out['incomplete'] and not out['optimizer_success'] and len(out['evaluations'])==1
 np.testing.assert_array_equal(out['parameters'],out['evaluations'][0]['parameters'])

def test_device_contract_and_frozen_law_copy():
 import pytest
 from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw
 p=problem();law=ScaleSeparatedMaterialLaw().double()
 args=dict(bulk_family='yeoh',interface_family=1,heterogeneity=0,observations=p.observations,position_sigma=.002,force_sigma=.2,torque_sigma=.02,bulk_law=law)
 clone=JointMAP(GeometrySpec(),**args)
 assert clone.kw['bulk_law'] is not law
 assert all(not v.requires_grad for v in clone.kw['bulk_law'].parameters())
 assert all(v.requires_grad for v in law.parameters())
 with pytest.raises(ValueError,match='CPU/GPU'):JointMAP(GeometrySpec(),device='cuda:3',**args)
 with pytest.raises(ValueError,match='outside bounds'):
  z=neutral_parameters();z[14]=100;clone.fit(active=[0],initial=z)
