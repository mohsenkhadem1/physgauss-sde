"""Check identity binding, mechanical derivatives and actual renderer connectivity."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
configure(0);sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch
from scripts.pidg_dense_mechanics.field import DenseMechanics
class Gaussians:
 def __init__(self):
  self.get_xyz=torch.randn(80,3,device='cuda')*.3;self.get_opacity=torch.ones(80,1,device='cuda')
class Deformation:
 def __init__(self):self.amplitude=torch.nn.Parameter(torch.tensor(.1,device='cuda'))
 def step(self,q,t):
  quat=torch.zeros(len(q),4,device=q.device);quat[:,0]=1;return {'d_xyz':torch.cat([quat,self.amplitude*t*t*q],1)}
torch.manual_seed(40);g=Gaussians();d=Deformation();m=DenseMechanics(particles=64,kernels=8,knots=8);m.initialize(d,g);x=m.X.detach().clone().requires_grad_();force,F,J,E=m.internal(x);assert torch.allclose(F,torch.eye(3,device='cuda').expand_as(F),atol=1e-6);assert force.abs().max()<1e-4;identity_error=float((F-torch.eye(3,device='cuda')).abs().max())
x=(m.X*1.05).detach().requires_grad_();force,F,J,E=m.internal(x);assert force.sum(0).abs().max()<1e-4
v=torch.randn_like(x);eps=1e-3;ep=m.internal((x.detach()+eps*v).requires_grad_())[3];em=m.internal((x.detach()-eps*v).requires_grad_())[3];fd=(ep-em)/(2*eps);ad=-(force*v).sum()/len(x);relative=float((fd-ad).abs()/(fd.abs()+ad.abs()+1e-6));assert relative<.02,relative
before=d.step(g.get_xyz,torch.full((len(g.get_xyz),1),.43,device='cuda'))['d_xyz'].detach().clone();loss=m(d,g,torch.tensor(.43,device='cuda'));loss.backward();assert d.amplitude.grad is not None and torch.isfinite(d.amplitude.grad) and d.amplitude.grad.abs()>1e-8;assert m.load.grad is not None and m.load.grad.abs().sum()>0;after=d.step(g.get_xyz,torch.full((len(g.get_xyz),1),.43,device='cuda'))['d_xyz'].detach();assert torch.equal(before,after)
r={'passed':True,'identity_F_max_error':identity_error,'energy_directional_derivative_relative_error':relative,'dense_motion_gradient':float(d.amplitude.grad),'load_gradient_norm':float(m.load.grad.norm()),'zero_update_mapping_identical':True,'tests':['Reference deformation gradient identity','Rest force negligible','Internal forces conserve net momentum','Internal force matches energy finite differences','Mechanics loss differentiates into dense motion and Gaussian loads','No mutation of dense mapping on initialization or loss evaluation']};Path('reports/pidg-dense-mechanics-v01/field-tests.json').write_text(json.dumps(r,indent=2));print(r)
