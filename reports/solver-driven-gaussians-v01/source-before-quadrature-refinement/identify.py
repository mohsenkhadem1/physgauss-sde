"""Direct inverse dynamics from observed positions, without shooting iterations."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter
from scripts.solver_driven_gaussians.dynamics import DynamicFEM

def identify(observation,output,device='cuda:0',kind='learned'):
    doc=np.load(observation);x=doc['x'];times=doc['times'];dt=float(np.diff(times).mean())
    if times.max()>.14000001:raise ValueError('Fitting observations extend beyond calibration prefix')
    m=DynamicFEM(device=device);torch.set_num_threads(2)
    if x.shape[1:]!=tuple(m.rest.shape):raise ValueError('Observed mesh identity mismatch')
    # Local polynomial derivative, excluding endpoints; no true velocities used.
    smooth=savgol_filter(x,11,3,axis=0);v=savgol_filter(x,11,3,deriv=1,delta=dt,axis=0);acc=savgol_filter(x,11,3,deriv=2,delta=dt,axis=0)
    designs=[];targets=[];frame_ids=[]
    for i in range(5,len(times)-5):
        xx=m.t(smooth[i]);vv=m.t(v[i]);fb,_,_=m.bulk(xx,kind=kind)
        support,tool,damp,known=m.components(xx,vv,float(times[i]),'calibration')
        A=torch.cat([fb[:,:,None],support.permute(0,2,1),tool[:,:,None],damp[:,:,None]],-1)
        y=m.mass[:,None]*m.t(acc[i])-known
        designs.append(A.detach().cpu().numpy().reshape(-1,11));targets.append(y.cpu().numpy().ravel());frame_ids.append(i)
    A=np.concatenate(designs);y=np.concatenate(targets);norm=np.linalg.norm(A,axis=0);norm=np.maximum(norm,1e-10)
    bounds=(np.r_[.15,np.zeros(10)],np.r_[6.,np.full(8,2000.),1000.,20.])
    def solve(ids):
        aa=np.concatenate([designs[i] for i in ids]);yy=np.concatenate([targets[i] for i in ids]);n=np.linalg.norm(aa,axis=0).clip(1e-10)
        r=lsq_linear(aa/n,yy,bounds=(bounds[0]*n,bounds[1]*n),tol=1e-10,max_iter=500)
        if not r.success:raise RuntimeError(r.message)
        return r.x/n
    theta=solve(range(len(designs)));rng=np.random.default_rng(73);ensemble=[]
    blocks=np.array_split(np.arange(len(designs)),6)
    for _ in range(8):ensemble.append(solve(np.concatenate([blocks[i] for i in rng.integers(0,6,6)])))
    sv=np.linalg.svd(A/norm,compute_uv=False)
    result=dict(completed=True,kind=kind,parameters=theta.tolist(),bootstrap_parameters=np.array(ensemble).tolist(),observations_sha256=hashlib.sha256(Path(observation).read_bytes()).hexdigest(),source=str(observation),calibration_end_s=float(times[-1]),future_positions_read=False,force_labels_read=False,material_labels_read=False,normalized_design_condition=float(sv[0]/sv[-1]),force_residual_rms_N=float(np.sqrt(np.mean((A@theta-y)**2))),scope='Direct inverse dynamics using supplied position tracks, known density, gravity and floor. Gaussian attachment coefficients inferred, candidate locations fixed. Bootstrap is sensitivity, not calibrated posterior.')
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='bootstrap_parameters'})
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--observations',default='runs/solver-driven-gaussians-v01/observations/calibration.npz');p.add_argument('--output',required=True);p.add_argument('--gpu',default='0');p.add_argument('--kind',choices=['learned','analytic'],default='learned');a=p.parse_args();identify(a.observations,a.output,'cuda:'+a.gpu,a.kind)
