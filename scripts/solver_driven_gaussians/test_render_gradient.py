"""Finite-difference verification of RGB -> dynamics -> material scale."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
configure(2);sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch,numpy as np
from torch.utils.checkpoint import checkpoint
from scripts.solver_driven_gaussians.rendering import MechanicalRenderer
torch.set_num_threads(2);r=MechanicalRenderer();m=r.model
p=m.t(json.loads((ROOT/'runs/solver-driven-gaussians-v02/selected-learned-fit.json').read_text())['parameters'])
# A deliberately different target avoids cancellation at a fitted minimum.
with torch.no_grad():target=r.image(m.rest,0).detach()
def loss(scale,ad=False):
 pp=torch.cat([scale[None],p[1:]]);x=m.rest.clone();v=torch.zeros_like(x);dt=.0002
 for begin in range(0,200,20):
  def segment(xx,vv,params,begin=begin):
   for j in range(begin,begin+20):
    f,_,_=m.force(xx,vv,j*dt,params,'calibration','learned',create_graph=ad);vv=vv+dt*f/m.mass[:,None];xx=xx+dt*vv
   return xx,vv
  if ad:x,v=checkpoint(segment,x,v,pp,use_reentrant=False)
  else:x,v=segment(x,v,pp)
 return (r.image(x,0)-target).square().mean()
s=p[0].clone().requires_grad_();value=loss(s,True);g=float(torch.autograd.grad(value,s)[0]);checks=[]
for eps in [.01,.005,.002]:
 with torch.no_grad():fd=float((loss(s.detach()+eps)-loss(s.detach()-eps))/(2*eps))
 checks.append(dict(epsilon=eps,finite_difference=fd,relative_error=abs(fd-g)/max(abs(fd),abs(g),1e-10)))
passed=min(c['relative_error'] for c in checks)<.08 and g!=0
report=dict(passed=passed,autograd=g,checks=checks,scope='Actual rasterizer, transported centers/covariance/radiance, and 200 integration steps; material-scale derivative. Piecewise rasterization causes finite-difference sensitivity.')
(ROOT/'reports/solver-driven-gaussians-v02/render-gradient.json').write_text(json.dumps(report,indent=2));print(report);assert passed
