"""Native UniPhy latent calibration from the shared visible-track prefix.

Unmodified released constitutive networks and MPM forward. Explicit
adaptations: fixed surface binding, masked track loss, bowl/sphere geometry hooks,
and standard per-iteration gradient clearing. No forecast target is read.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
import taichi as ti
from sklearn.cluster import KMeans
from scripts.complex_scenes_v18.uniphy_runtime import models,wrapper
from scripts.complex_scenes_v18.uniphy_drop import add_bowl,add_sphere
from scripts.complex_scenes_v18.uniphy_track_loss import TrackedLoss
from scripts.complex_scenes_v18.particle_tracks import binding
from scripts.complex_scenes_v18.pair_contact import material_surface
from scripts.complex_scenes_v18.run_development import ROOT,write


def fit(args):
    base=ROOT/'runs/complex-scenes-v18';name=f'container_drop-v{args.variant}';out=base/args.label/name;out.mkdir(parents=True,exist_ok=False)
    source=Path(args.prefix)if args.prefix else base/'development-prefixes'/name;meta=json.loads((source/'metadata.json').read_text());path=source/'observations.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['observations_sha256']:raise ValueError('Prefix integrity')
    a=np.load(path);observed=a['positions_m'][:,0];queries=a['material_queries'];clock=a['times_s'];offset=np.array([.3,.3,.3],np.float32);dt=50e-6
    frame_steps=np.rint(clock/dt).astype(np.int32);steps=int(frame_steps[-1]);chunk=steps+1
    if not np.allclose(frame_steps*dt,clock,atol=1e-9):raise ValueError('Observation clock')
    ti.init(arch=ti.cuda,device_memory_GB=3,debug=False);nets,latent,bank=models()
    centers=KMeans(n_clusters=8,random_state=42,n_init=10).fit(bank.numpy()).cluster_centers_
    np.save(out/'initial-centroids.npy',centers)
    from scripts.complex_scenes_v18.world_queries import Query,drop_particle_cloud
    query=Query(**meta.get('public_query',dict(family='container_drop',variant=args.variant)))
    if query.family!='container_drop'or query.variant!=args.variant or meta['case']!='container_drop':raise ValueError('Native MPM query identity mismatch')
    if not np.isfinite(observed).all()or np.any(np.diff(clock)<=0)or not np.isclose(clock[-1],meta['calibration_duration_s'],atol=1e-9):raise ValueError('Invalid calibration prefix')
    q,initial=drop_particle_cloud(query);x=(initial+offset).astype(np.float32)
    w=wrapper(x,np.repeat(x[:,None],1,axis=1),nets,dt=dt,steps=steps,chunk=chunk,density=625.,gravity=(0,0,-9.81))
    sim=w.simulator;add_bowl(sim,(np.array([.06,0,-.08 if args.variant else -.07])+offset).tolist(),.10 if args.variant else .12)
    if args.corrected_adjoint:
        from scripts.complex_scenes_v18.uniphy_adjoint import CompleteAdjointBridge
        bridge=CompleteAdjointBridge(sim).attach()
    if args.variant:add_sphere(sim,(np.array([.09,.035,-.06])+offset).tolist(),.015)
    indices,weights=binding(queries);criterion=TrackedLoss(sim,indices,weights,observed+offset,frame_steps)
    plan=dict(method='Native UniPhy released demo MPM and constitutive models with visible-track loss and analytic bowl/sphere hooks',development_only=True,known_boundary=True,observations_sha256=meta['observations_sha256'],dt_s=dt,grid_spacing_m=.02,particles=432,epochs=args.epochs,learning_rate=args.lr,optimizer='AdamW; gradients cleared each iteration',initialization=dict(kmeans_clusters=8,seed=42,n_init=10,author_demo_centroid=2,screen_centroids=args.screen_centroids),binding='Fixed trilinear cell-center interpolation with half-spacing surface extrapolation',forecast_targets_read=False,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),ROOT/'scripts/complex_scenes_v18/uniphy_runtime.py',ROOT/'scripts/complex_scenes_v18/uniphy_track_loss.py',ROOT/'scripts/complex_scenes_v18/particle_tracks.py',ROOT/'scripts/complex_scenes_v18/uniphy_drop.py']})
    plan['corrected_adjoint']=args.corrected_adjoint
    plan.update(calibration_prefix=str(source),prefix_metadata_sha256=hashlib.sha256((source/'metadata.json').read_bytes()).hexdigest(),public_query=meta.get('public_query'),evaluation_split=meta.get('split','development'),development_only=meta.get('split','development')=='development')
    if args.corrected_adjoint:
        bridge_path=ROOT/'scripts/complex_scenes_v18/uniphy_adjoint.py'
        plan['source_sha256'][str(bridge_path.relative_to(ROOT))]=hashlib.sha256(bridge_path.read_bytes()).hexdigest()
    write(out/'plan.json',plan);start=time.monotonic();rows=[];best=None;best_loss=float('inf')
    def evaluate(stage,backward=False):
        latent.weight.grad=None;w.simulator_variables_initialize();w.update_traj_latent_learnable(latent.weight[0].detach())
        for i in range(steps):
            if time.monotonic()-start>args.seconds:raise TimeoutError('Declared fitting budget')
            with torch.no_grad():sim.advance_F_stress(i,0)
            if not sim.cfl_satisfy[None]:raise FloatingPointError('Native CFL failure')
            if i%200==0:write(out/'status.json',dict(pid=os.getpid(),stage=stage,step=i,steps=steps,history=rows,elapsed_s=time.monotonic()-start))
        criterion.compute();value=float(sim.loss[None])
        if not np.isfinite(value):raise FloatingPointError('Nonfinite calibration loss')
        if backward:
            sim.loss.grad[None]=1.;criterion.compute.grad();write(out/'status.json',dict(pid=os.getpid(),stage=stage+'-backward',history=rows,elapsed_s=time.monotonic()-start));sim.advance_grad_F_stress(0,0)
            if latent.weight.grad is None or not torch.isfinite(latent.weight.grad).all():raise FloatingPointError('Missing/nonfinite native latent gradient')
        return value
    for index in (range(8)if args.screen_centroids else[2]):
        row=dict(stage='initialization',index=index,completed=False);tick=time.monotonic()
        with torch.no_grad():latent.weight.copy_(torch.as_tensor(centers[index:index+1],device='cuda'))
        try:
            loss=evaluate(f'centroid-{index}');row.update(completed=True,loss_mm2=loss)
            if loss<best_loss:best_loss=loss;best=latent.weight.detach().cpu().clone()
        except Exception as e:row.update(error=str(e),traceback=traceback.format_exc())
        row['seconds']=time.monotonic()-tick;rows.append(row);write(out/'training.json',dict(history=rows))
    if best is None:write(out/'fit-seal.json',dict(**plan,completed=False,history=rows));return False
    with torch.no_grad():latent.weight.copy_(best.to('cuda'))
    optimizer=torch.optim.AdamW([latent.weight],lr=args.lr);failures=0
    for epoch in range(args.epochs):
        row=dict(stage='fit',epoch=epoch,completed=False);tick=time.monotonic()
        try:
            optimizer.zero_grad(set_to_none=True);loss=evaluate(f'epoch-{epoch}',True);row.update(completed=True,loss_mm2=loss,gradient_norm=float(latent.weight.grad.norm()))
            if loss<best_loss:best_loss=loss;best=latent.weight.detach().cpu().clone()
            optimizer.step();failures=0
        except Exception as e:
            row.update(error=str(e),traceback=traceback.format_exc());failures+=1
            with torch.no_grad():latent.weight.copy_(best.to('cuda'))
            optimizer=torch.optim.AdamW([latent.weight],lr=args.lr*(.5**failures))
        row['seconds']=time.monotonic()-tick;rows.append(row);write(out/'training.json',dict(history=rows,best_loss_mm2=best_loss))
        if failures>=3 or time.monotonic()-start>args.seconds:break
    torch.save(dict(latent=best,plan=plan,loss_mm2=best_loss),out/'best.pt');checkpoint_sha=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest()
    write(out/'fit-seal.json',dict(**plan,completed=True,requested_epochs_completed=sum(r.get('stage')=='fit'and r['completed']for r in rows)==args.epochs,history=rows,best_loss_mm2=best_loss,checkpoint_sha256=checkpoint_sha,elapsed_s=time.monotonic()-start))
    with torch.no_grad():latent.weight.copy_(best.to('cuda'))
    w.simulator_variables_initialize();w.update_traj_latent_learnable(latent.weight[0].detach());fullq,_=material_surface(9);bi,bw=binding(fullq);frames=[];times=[];result=dict(completed=False)
    def save(i):
        particles=sim.x.to_numpy()[:len(x),i%chunk].astype(float)-offset
        frames.append((particles[bi]*bw[...,None]).sum(1)[None]);times.append(i*dt)
    try:
        save(0)
        for i in range(round(.24/dt)):
            with torch.no_grad():sim.advance_F_stress(i,0)
            if not sim.cfl_satisfy[None]:raise FloatingPointError('Native forecast CFL failure')
            if (i+1)%40==0:save(i+1)
        result['completed']=True
    except Exception as e:result['error']=str(e)
    np.savez_compressed(out/'sealed-rollout.npz',surface_positions_m=frames,times_s=times)
    result.update(checkpoint_sha256=checkpoint_sha,trajectory_sha256=hashlib.sha256((out/'sealed-rollout.npz').read_bytes()).hexdigest(),saved_frames=len(frames),forecast_targets_read=False);write(out/'rollout-complete.json',result);return True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',type=int,choices=[0,1],default=0);p.add_argument('--label',default='uniphy-latent-fit-01');p.add_argument('--epochs',type=int,default=2);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--screen-centroids',action='store_true');p.add_argument('--corrected-adjoint',action='store_true');p.add_argument('--seconds',type=int,default=7200);p.add_argument('--prefix');a=p.parse_args();torch.set_num_threads(1);raise SystemExit(0 if fit(a)else 1)
