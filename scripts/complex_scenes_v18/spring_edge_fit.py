"""Checkpointed per-edge Spring-Gaus calibration with common scene boundaries.

Uses only exported visible calibration positions. Best parameters are selected by
calibration loss and sealed before full rollout; forecast targets are never read.
This is an adapted mechanics baseline, not author end-to-end RGB reconstruction.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from scripts.complex_scenes_v18.differentiable_spring import DifferentiableSpringScene
from scripts.complex_scenes_v18.run_development import ROOT,write


def snapshot(sim):return [p.detach().cpu().clone()for p in sim.parameters]


def restore(sim,values):
    with torch.no_grad():
        for p,v in zip(sim.parameters,values):p.copy_(v.to(p))


def fit(args):
    name=f'{args.case}-v{args.variant}';out=ROOT/'runs/complex-scenes-v18'/args.label/name;out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'runs/complex-scenes-v18/development-prefixes'/name;meta=json.loads((source/'metadata.json').read_text());target=source/'observations.npz'
    if hashlib.sha256(target.read_bytes()).hexdigest()!=meta['observations_sha256']:raise ValueError('Prefix integrity')
    method=getattr(args,'method','spring-gaus');warm=None
    if method=='spring-gaus':
        warm=ROOT/'runs/complex-scenes-v18/spring-scale-fit-01'/name/'fit-seal.json';warmdata=json.loads(warm.read_text())
        if not warmdata['completed']or warmdata['observations_sha256']!=meta['observations_sha256']:raise ValueError('Matching completed initialization required')
        initial_k=warmdata['best']['log10_k'];Class=DifferentiableSpringScene
    else:
        from scripts.complex_scenes_v18.differentiable_phystwin import DifferentiablePhysTwinScene
        Class=DifferentiablePhysTwinScene;initial_k=-.3
    a=np.load(target);observed=torch.tensor(a['positions_m'],dtype=torch.float32,device='cuda');indices=torch.tensor(a['surface_indices'],device='cuda');dt=25e-6;stride=round(.002/dt)
    sim=Class(args.case,args.variant,log10_k=initial_k,per_edge=True)
    optimizer=torch.optim.Adam(sim.parameters,lr=args.lr);rows=[];best=None;best_loss=float('inf');start=time.monotonic();failures=0
    sources=[Path(__file__),ROOT/'scripts/complex_scenes_v18/explicit_boundaries.py',ROOT/'scripts/complex_scenes_v18/torch_boundaries.py']
    sources+=[ROOT/'scripts/complex_scenes_v18'/p for p in (['differentiable_spring.py','spring_scene.py']if method=='spring-gaus'else['differentiable_phystwin.py','phystwin_autograd.py','phystwin_scene.py'])]
    plan=dict(method=method,adaptation='Author internal force; differentiable common scene boundaries; PhysTwin training uses equivalent Torch step and forecast uses author Warp integration',development_only=True,known_boundary=True,epochs=args.epochs,learning_rate=args.lr,dt_s=dt,checkpoint_steps=stride,parameter_count=sum(p.numel()for p in sim.parameters),observations_sha256=meta['observations_sha256'],initialization_sha256=hashlib.sha256(warm.read_bytes()).hexdigest()if warm else None,initial_log10_k=initial_k,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sources},forecast_targets_read=False)
    if method=='phystwin':
        np.savez_compressed(out/'graph.npz',**{f'edges_{i}':e.cpu().numpy()for i,e in enumerate(sim.edges)})
        plan.update(graph_sha256=hashlib.sha256((out/'graph.npz').read_bytes()).hexdigest(),damping='Fixed zero dashpot and drag damping')
    else:
        topology={}
        for i,model in enumerate(sim.native):
            topology[f'knn_index_{i}']=model.knn_index.detach().cpu().numpy()
            topology[f'origin_len_{i}']=model.origin_len.detach().cpu().numpy()
        np.savez_compressed(out/'graph.npz',**topology)
        plan['graph_sha256']=hashlib.sha256((out/'graph.npz').read_bytes()).hexdigest()
    write(out/'plan.json',plan)
    for epoch in range(args.epochs):
        if time.monotonic()-start>args.seconds:rows.append(dict(epoch=epoch,completed=False,error='Declared fitting time budget'));break
        before=snapshot(sim);optimizer.zero_grad(set_to_none=True);tick=time.monotonic();row=dict(epoch=epoch,completed=False)
        try:
            state=sim.initial();loss=observed.new_zeros(())
            for frame in range(1,len(observed)):
                begin=(frame-1)*stride
                def segment(*s,begin=begin):
                    for i in range(begin,begin+stride):s=sim.step(s,i,dt,training=True)
                    return s
                state=checkpoint(segment,*state,use_reentrant=False)
                prediction=sim.positions(state)[:,indices]
                if not torch.isfinite(prediction).all()or prediction.abs().max()>10:raise FloatingPointError('Invalid native prediction')
                loss=loss+(prediction-observed[frame]).square().sum(-1).mean()*1e6/(len(observed)-1)
                write(out/'status.json',dict(pid=os.getpid(),epoch=epoch,frame=frame,frames=len(observed),stage='forward',history=rows,elapsed_s=time.monotonic()-start))
            value=float(loss.detach());loss.backward()
            grads=[p.grad for p in sim.parameters]
            if any(g is None or not torch.isfinite(g).all()for g in grads):raise FloatingPointError('Missing/nonfinite parameter gradient')
            norm=float(torch.nn.utils.clip_grad_norm_(sim.parameters,100.))
            if value<best_loss:
                best_loss=value;best=before;torch.save(dict(parameters=best,epoch=epoch,loss_mm2=value,plan=plan),out/'best.pt')
            optimizer.step()
            with torch.no_grad():
                for model in sim.native:model.global_k.clamp_(-4,4);model.damp.clamp_(-10,1)
            row.update(completed=True,loss_mm2=value,gradient_norm=norm,gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved());failures=0
        except Exception as e:
            row.update(error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());restore(sim,best if best is not None else before);failures+=1
            optimizer=torch.optim.Adam(sim.parameters,lr=args.lr*(.5**failures))
        row['seconds']=time.monotonic()-tick;rows.append(row);write(out/'training.json',dict(history=rows,best_loss_mm2=best_loss if best else None,elapsed_s=time.monotonic()-start))
        if failures>=3:break
    seal=dict(**plan,completed=best is not None,requested_epochs_completed=sum(r['completed']for r in rows)==args.epochs,history=rows,best_loss_mm2=best_loss if best else None,elapsed_s=time.monotonic()-start)
    if best is not None:seal['checkpoint_sha256']=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest()
    write(out/'fit-seal.json',seal)
    if best is None:return seal
    restore(sim,best);state=sim.initial();frames=[sim.positions(state).detach().cpu().numpy()];times=[0.];result=dict(completed=False)
    native=sim.native_replay()if method=='phystwin'else None
    try:
        for i in range(round(sim.base.metadata['duration_s']/dt)):
            if native is not None:native.step(dt)
            else:state=sim.step(state,i,dt,training=False)
            if (i+1)%stride==0:
                p=native.sample_surface()if native is not None else sim.positions(state).detach().cpu().numpy()
                if not np.isfinite(p).all():raise FloatingPointError('Nonfinite sealed forecast')
                frames.append(p);times.append((i+1)*dt)
        result['completed']=True
    except Exception as e:result['error']=str(e)
    np.savez_compressed(out/'sealed-rollout.npz',surface_positions_m=frames,times_s=times)
    result.update(checkpoint_sha256=seal['checkpoint_sha256'],trajectory_sha256=hashlib.sha256((out/'sealed-rollout.npz').read_bytes()).hexdigest(),forecast_targets_read=False,saved_frames=len(frames));write(out/'rollout-complete.json',result);return seal


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--variant',type=int,default=0);p.add_argument('--label',default='spring-edge-fit-01');p.add_argument('--epochs',type=int,default=40);p.add_argument('--lr',type=float,default=.03);p.add_argument('--seconds',type=int,default=7200);p.add_argument('--method',choices=['spring-gaus','phystwin'],default='spring-gaus');a=p.parse_args();torch.set_num_threads(1);r=fit(a);raise SystemExit(0 if r['completed']else 1)
