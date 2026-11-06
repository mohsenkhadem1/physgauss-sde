"""Numerical diagnostic without validation/test image access."""
import sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch,numpy as np
from scripts.pidg_solver_forecast.common import BASE
from scripts.pidg_solver_forecast.model import SceneFEM
torch.set_num_threads(2);out=BASE/a.scene/'solver';data=np.load(out/'adapter.npz');m=SceneFEM(data,torch.load(out/'appearance.pt',weights_only=False));m.set_binding_tets();rows=[]
with torch.no_grad():
 m.param['log_scale'].fill_(np.log(.03));m.param['loads'].zero_();m.param['body'].zero_();x0=m.observed[-1];v0=m.observed_v[-1];q=x0-x0.mean(0);wx=torch.stack([torch.stack([q[:,0]*0,q[:,2],-q[:,1]],-1),torch.stack([-q[:,2],q[:,0]*0,q[:,0]],-1),torch.stack([q[:,1],-q[:,0],q[:,0]*0],-1)],1);omega=torch.linalg.lstsq(wx.reshape(-1,3),(v0-v0.mean(0)).reshape(-1,1)).solution[:,0];rigid=v0.mean(0)+torch.einsum('nij,j->ni',wx,omega)
 for mode,vel in [('full',v0),('rigid',rigid),('zero',torch.zeros_like(v0))]:
  for dt in [.001,.0002,.00005]:
   x=x0.clone();v=vel.clone();error=None;step=0
   try:
    for step in range(round(.1/dt)):x,v=m.step(x,v,m.end+step*dt,dt)
   except (RuntimeError,ValueError) as e:error=str(e)
   row=dict(mode=mode,dt=dt,completed=error is None,elapsed_simulation=(step+1)*dt,error=error);rows.append(row);print(row,flush=True)
report=dict(rows=rows,velocity_rms=float(v0.square().mean().sqrt()),max_speed=float(v0.norm(dim=-1).max()),rigid_residual_rms=float((v0-rigid).square().mean().sqrt()),scope='No target images. Same terminal state, varying step and initial-velocity decomposition. Low material scale and zero learned loads.')
(out/'stability-diagnosis.json').write_text(json.dumps(report,indent=2))
