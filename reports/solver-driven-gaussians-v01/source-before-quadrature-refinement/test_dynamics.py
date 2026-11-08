import json
from pathlib import Path
import numpy as np
import torch
from scripts.solver_driven_gaussians.dynamics import DynamicFEM
from scripts.solver_driven_gaussians.binding import DenseBinding

torch.set_num_threads(2)
m=DynamicFEM(cells=(2,1,1));rng=np.random.default_rng(7)
points=rng.uniform([-.002,-.032,-.032],[.122,.032,.032],(40,3));cov=np.tile(np.diag([1e-6,2e-6,3e-6]),(40,1,1))
b=DenseBinding(m,points,cov);rest=b.transport(m.rest)
np.testing.assert_allclose(rest['centers'].numpy(),points,atol=1e-13)
A=m.t([[1.07,.04,0],[0,.96,.02],[0,0,1.02]]);shift=m.t([.01,-.02,.03]);x=(m.rest@A.T+shift).requires_grad_();state=b.transport(x)
np.testing.assert_allclose(state['centers'].detach().numpy(),points@A.numpy().T+shift.numpy(),atol=1e-12)
np.testing.assert_allclose(state['covariance'].detach().numpy(),A.numpy()@cov@A.numpy().T,atol=1e-12)
f,E,J=m.bulk(x,create_graph=True);assert float(f.sum(0).norm())<1e-10
direction=m.t(rng.normal(size=x.shape));eps=1e-7
fd=(m.bulk(x+eps*direction)[1]-m.bulk(x-eps*direction)[1])/(2*eps);ad=-(f*direction).sum()
relative=float(abs(fd-ad)/abs(fd));assert relative<1e-6
p=m.t([1.,200.,0,0,0,0,0,0,0,100.,2.]).requires_grad_()
r=m.rollout(p,duration=.002,dt=.0001,sample_dt=.001,create_graph=True)
grad=torch.autograd.grad(r['x'][-1].square().sum(),p)[0];assert torch.isfinite(grad).all() and grad.norm()>0
report=dict(passed=True,identity_binding_max_error=float((rest['centers']-m.t(points)).abs().max()),energy_gradient_relative_error=relative,internal_force_sum=float(f.sum(0).norm()),rollout_parameter_gradient=grad.tolist(),scope='Identity/affine appearance geometry, conservative bulk force and gradients through dynamic integration; not stability or identification proof.')
Path('reports/solver-driven-gaussians-v01/unit-tests.json').write_text(json.dumps(report,indent=2));print(report)
