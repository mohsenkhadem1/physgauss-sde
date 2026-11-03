"""End-to-end native latent gradient audit with nonrigid initial velocity."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
import taichi as ti
from scripts.complex_scenes_v18.uniphy_runtime import models,wrapper
from scripts.complex_scenes_v18.native_smoke import REPORT


def run(origin=.4,corrected=False):
    ti.init(arch=ti.cuda,device_memory_GB=2,debug=False)
    networks,latent,bank=models();x=np.stack(np.meshgrid(*[np.linspace(origin,origin+.02,3)]*3,indexing='ij'),-1).reshape(-1,3).astype(np.float32)
    steps=24;chunk=32;dt=50e-6
    w=wrapper(x,np.repeat(x[:,None],chunk+1,axis=1),networks,dt=dt,steps=steps,chunk=chunk)
    if corrected:
        from scripts.complex_scenes_v18.uniphy_adjoint import CompleteAdjointBridge
        bridge=CompleteAdjointBridge(w.simulator).attach()
    w.init_velocities=((x-x.mean(0))*np.array([-20.,10.,0.])).astype(np.float32)
    z0=latent.weight.detach().clone()
    def evaluate(z,backward=False):
        with torch.no_grad():latent.weight.copy_(z)
        latent.weight.grad=None;w.simulator_variables_initialize();w.update_traj_latent_learnable(latent.weight[0].detach());sim=w.simulator
        for i in range(steps):
            sim.advance_F_stress(i,0)
            if not sim.cfl_satisfy[None]:raise RuntimeError('CFL failure')
        w.compute_loss();loss=float(sim.loss[None])
        if not np.isfinite(loss):raise FloatingPointError('Nonfinite native loss')
        if backward:
            sim.loss.grad[None]=1.;w.compute_loss.grad();sim.advance_grad_F_stress(0,0)
            return loss,latent.weight.grad.detach().clone()
        return loss
    loss,g=evaluate(z0,True);norm=float(g.norm())
    if not np.isfinite(norm)or norm<=1e-15:raise RuntimeError('No finite informative gradient')
    direction=g/g.norm();repeat=evaluate(z0);rows=[]
    for eps in [.1,.03,.01]:
        plus=evaluate(z0+eps*direction);minus=evaluate(z0-eps*direction);fd=(plus-minus)/(2*eps)
        rows.append(dict(epsilon=eps,plus_loss=plus,minus_loss=minus,finite_difference=fd,autodiff=norm,relative_error=abs(fd-norm)/max(abs(fd),norm,1e-30),signal_to_repeat_noise=abs(plus-minus)/max(abs(loss-repeat),1e-30)))
    passed=sum(r['relative_error']<.1 and r['signal_to_repeat_noise']>10 for r in rows)>=2
    return dict(passed=passed,scope='Native MPM nonrigid rollout directional derivative, not benchmark performance',corrected_adjoint=corrected,coordinate_origin=origin,steps=steps,dt_s=dt,loss=loss,repeat_loss=repeat,gradient_norm=norm,checks=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corrected-adjoint',action='store_true');p.add_argument('--origin',type=float,default=.4);p.add_argument('--output',default='uniphy-gradient-audit.json');args=p.parse_args()
    if Path(args.output).name!=args.output:raise ValueError('Report basename required')
    torch.set_num_threads(1);out=REPORT/args.output;t=time.monotonic()
    if out.exists():raise RuntimeError('Preserve original audit')
    try:r=run(args.origin,args.corrected_adjoint)
    except Exception as e:r=dict(passed=False,error=str(e),traceback=traceback.format_exc())
    r.update(elapsed_s=time.monotonic()-t,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));raise SystemExit(0 if r['passed']else 1)
