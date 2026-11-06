"""Renderer identity and general-mesh mechanics checks on synthetic inputs."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
configure(0);sys.path.insert(0,str(EXTERNAL/'upstream'))
import numpy as np,torch
from types import SimpleNamespace
from scripts.pidg_solver_forecast.geometry import volume
from scripts.pidg_solver_forecast.model import SceneFEM
from scripts.pidg_solver_forecast.common import appearance_at
from scripts.solver_driven_gaussians.rendering import cameras
from gaussian_renderer import render
torch.set_num_threads(2);torch.manual_seed(7);rng=np.random.default_rng(7);xyz=rng.uniform([0,-.03,-.03],[.12,.03,.03],(500,3));g=SimpleNamespace(get_xyz=torch.tensor(xyz,device='cuda',dtype=torch.float32),get_scaling=torch.full((500,3),.003,device='cuda'),get_rotation=torch.tensor([1.,0,0,0],device='cuda').expand(500,4),get_features=torch.zeros(500,16,3,device='cuda'),get_opacity=torch.full((500,1),.7,device='cuda'),active_sh_degree=3,max_sh_degree=3)
class Deform:
 def step(self,x,t):return dict(d_xyz=torch.tensor([1.,0,0,0,0,0,0],device='cuda').expand(len(x),7),d_rotation=torch.tensor([.02,.03,-.02,.01],device='cuda').expand(len(x),4),d_scaling=torch.zeros_like(x))
d=Deform();ap=appearance_at(g,d,0.);geom=volume(xyz,np.ones(500),resolution=5);origin=np.zeros(3);length=.12;rest=geom['nodes']/length;adapter=dict(rest=rest,elements=geom['elements'],ids=geom['elements'][geom['bind_tet']],N=geom['bind_N'],origin=origin,length=length,times=np.linspace(0,.6,61),x=np.repeat(rest[None],61,0),v=np.zeros((61,*rest.shape)));m=SceneFEM(adapter,ap);m.set_binding_tets();cam=cameras()[0]
with torch.no_grad():
 pk=d.step(g.get_xyz,None);native=render(cam,g,SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False),torch.zeros(3,device='cuda'),pk['d_xyz'],pk['d_rotation'],pk['d_scaling'],False)['render'];ours=m.image(m.rest,cam);delta=float((native-ours).abs().max());assert delta<3e-5,delta
 x=m.rest*torch.tensor([1.02,.99,1.01],device='cuda');f,_,_=m.bulk(x,scale=1/12000);net=float(f.sum(0).norm());assert net<1e-5
x,v=m.integrate(m.rest,torch.ones_like(m.rest)*.01,0,.005,ad=True);loss=m.image(x,cam).square().mean();loss.backward();assert m.param['body'].grad is not None and torch.isfinite(m.param['body'].grad).all()
rigid_velocity=np.cross(np.array([.2,-.1,.3]),rest)+np.array([.04,.02,-.01]);adapter['v']=np.repeat(rigid_velocity[None],61,0);check=SceneFEM(adapter,ap);check.set_velocity_factor(0.);rigid_error=float((check.observed_v-check.raw_v).abs().max());assert rigid_error<1e-5
report=dict(passed=True,rigid_velocity_max_error=rigid_error,identity_max_rgb_error=delta,internal_force_net=net,render_to_body_force_gradient=m.param['body'].grad.tolist(),scope='Synthetic geometry: released CUDA covariance convention, rest-render preservation, bulk force conservation, RGB gradients through actual general-mesh integration.')
(ROOT/'reports/pidg-solver-forecast-v01/model-tests.json').write_text(json.dumps(report,indent=2));print(report)
