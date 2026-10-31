"""Small operator checks without benchmark targets or fitting."""
import unittest
import numpy as np
import torch
from scripts.benchmark_improvement_v14.phystwin_attachment import AttachmentFEM
from scripts.external_benchmark.fem import ExternalFEM
class Checks(unittest.TestCase):
 def setUp(self):
  nodes=np.array([[x,y,z] for x in [0.,.1] for y in [0.,.1] for z in [0.,.1]])
  self.data=dict(nodes=nodes,attachment_weights=np.exp(-np.arange(8)[None]/2),controller_targets=np.array([nodes[None]]),attachment_active=np.ones((1,1)),floor=-.05)
  self.m=AttachmentFEM(self.data,device='cpu',field=True,contact=True,materials=6)
  self.th=self.m.t([-4,2,5]+[0]*6+[0,1,0])
 def test_original_operator(self):
  old=ExternalFEM(self.data,device='cpu',field=True,contact=True,materials=6)
  x=self.m.rest+.01*self.m.t(np.arange(24).reshape(8,3)%3)
  a=self.m.evaluate(x,self.th,0,True);b=old.evaluate(x,self.th[:-1],0,True)
  for aa,bb in zip(a,b):np.testing.assert_allclose(aa.numpy(),bb.numpy(),atol=1e-12,rtol=1e-12)
 def test_width_force_derivative(self):
  x=self.m.rest+.03;th=self.th.clone().requires_grad_();direction=torch.arange(24,dtype=torch.float64).reshape(8,3)/24
  _,r,_=self.m.evaluate(x,th,0);g=torch.autograd.grad((r*direction).sum(),th)[0][-1].item();eps=1e-5
  tp=self.th.clone();tm=self.th.clone();tp[-1]+=eps;tm[-1]-=eps
  rp=self.m.evaluate(x,tp,0)[1];rm=self.m.evaluate(x,tm,0)[1];fd=((rp-rm)*direction).sum().item()/(2*eps)
  self.assertAlmostEqual(g,fd,places=8)
 def test_implicit_width_gradient(self):
  th=self.th.clone().requires_grad_();x=self.m.predict(th,0);g=torch.autograd.grad((x*x).sum(),th)[0][-1].item();eps=1e-4
  tp=self.th.clone();tm=self.th.clone();tp[-1]+=eps;tm[-1]-=eps
  xp=self.m.solve(tp,0)[0];xm=self.m.solve(tm,0)[0];fd=((xp*xp).sum()-(xm*xm).sum()).item()/(2*eps)
  self.assertLess(abs(g-fd),2e-5)
 def test_strict_evaluator(self):
  with self.assertRaises(ValueError):AttachmentFEM(dict(self.data,evaluation={}),device='cpu')
if __name__=='__main__':unittest.main()
