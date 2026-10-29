"""Black-box position-only inverse fitting around official PhysGaussian MPM.

This inverse adapter is ours; PhysGaussian's original paper is not presented as
having this environmental-identification objective or observation protocol.
"""
from pathlib import Path
import argparse,fcntl,json,sys,time
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.physgaussian import PositionMPM
from scripts.position_ood.observations import make
from scripts.position_ood.study import BASE,within_deadline
from scripts.phystwin.sequential_fixture import LAYOUTS
from scripts.phystwin.sequential_reference import write


def parameters(z,layout):
    g=np.r_[z[4:6],-1.];g*=9.81/np.linalg.norm(g)
    return dict(E=float(np.exp(z[0])),Kleft=float(np.exp(z[1])),Kright=float(np.exp(z[2])),layout=LAYOUTS[layout],material_gradient=float(z[3]),gravity=g)


def fit(model,obs,layout,z0,iterations,seconds=2.):
    poses=obs['poses'][:3];target=obs['measurements'][:3];visible=obs['visible_ids'];cache={};trace=[];start=time.time()
    def residual(z):
        within_deadline();key=tuple(z)
        if key in cache:return cache[key]
        model.parameters(**parameters(z,layout));rows=[]
        try:
            for pose in poses:rows.append(model.solve(pose,seconds=seconds)['field'][visible])
        except RuntimeError as error:
            r=np.full(target.size,1e3);cache[key]=r;trace.append(dict(z=z.tolist(),loss=float(np.mean(r*r)),valid=False,error=str(error)));return r
        r=((np.array(rows)-target)/.0003).ravel();cache[key]=r;trace.append(dict(z=z.tolist(),loss=float(np.mean(r*r)),valid=True));print('MPM_FIT',layout,len(trace),trace[-1]['loss'],flush=True);return r
    def jac(z):
        r=residual(z);cols=[]
        for k in range(6):
            step=.02 if z[k]+.02<=hi[k] else -.02;a=z.copy();a[k]+=step;cols.append((residual(a)-r)/step)
        return np.stack(cols,axis=1)
    lo=np.r_[np.log([3000,300,300]),-1.2,-.8,-.8];hi=np.r_[np.log([100000,60000,60000]),1.2,.8,.8]
    result=least_squares(residual,np.asarray(z0),jac=jac,bounds=(lo,hi),max_nfev=iterations,ftol=1e-5,xtol=1e-4,gtol=1e-4,x_scale='jac')
    # Perturbation evaluations are real feasible parameter candidates; select
    # the best observed residual among them, without consulting held targets.
    valid=[r for r in trace if r.get('valid',True) and np.all(np.array(r['z'])>=lo) and np.all(np.array(r['z'])<=hi)]
    if not valid:raise RuntimeError('No valid MPM parameter candidate; see numerical configuration')
    best=min(valid,key=lambda r:r['loss'])
    return dict(**best,layout=layout,evaluations=len(trace),optimizer_success=bool(result.success),message=result.message,elapsed_seconds=time.time()-start,trace=trace)


def run(a):
    if ('test' in a.split.split('_')):
        from scripts.position_ood.seal import verify
        verify()
    obs=np.load(make(a.split,a.object));tag='mpm' if a.attachment_width==0 else f'mpm-layer{a.attachment_width:g}'
    if (a.seconds,a.ramp_seconds,a.fine_dt,a.fine_grid,a.fine_resolution)!=(2.,0.,.0001,64,2):tag+=f'-s{a.seconds:g}-ramp{a.ramp_seconds:g}-dt{a.fine_dt:g}-g{a.fine_grid}-r{a.fine_resolution}'
    dest=BASE/f'{tag}/{a.split}-{a.object}';dest.mkdir(parents=True,exist_ok=True);lock=(dest/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if (dest/'summary.json').exists():return
    start=time.time();coarse=PositionMPM(1,32,.0002,attachment_width=a.attachment_width,ramp_seconds=a.ramp_seconds,rpic=-1. if a.attachment_width else 0.);rows=[]
    for layout in range(3):
        path=dest/f'coarse-layout{layout}.json'
        if path.exists():row=json.loads(path.read_text())
        else:row=fit(coarse,obs,layout,np.r_[np.log([20000,5000,5000]),0.,0.,0.],a.coarse_iterations,seconds=a.seconds);write(path,row)
        rows.append(row)
    best=min(rows,key=lambda r:r['loss']);del coarse
    path=dest/'fine-fit.json';fine=PositionMPM(a.fine_resolution,a.fine_grid,a.fine_dt,attachment_width=a.attachment_width,ramp_seconds=a.ramp_seconds,rpic=-1. if a.attachment_width else 0.)
    if path.exists():fitted=json.loads(path.read_text())
    else:fitted=fit(fine,obs,best['layout'],best['z'],a.fine_iterations,seconds=a.seconds);write(path,fitted)
    fine.parameters(**parameters(fitted['z'],fitted['layout']));pred=[];drift=[]
    for pose in obs['poses']:
        within_deadline();r=fine.solve(pose,seconds=a.seconds);pred.append(r['field']);drift.append(r['drift_m'])
    np.savez_compressed(dest/'predictions.npz',predicted=pred,drift_m=drift,poses=obs['poses']);write(dest/'summary.json',dict(state='complete',split=a.split,object=a.object,coarse_rows=rows,best=fitted,resolution=a.fine_resolution,grid=a.fine_grid,dt=a.fine_dt,seconds=a.seconds,ramp_seconds=a.ramp_seconds,attachment_width=a.attachment_width,elapsed_seconds=time.time()-start,scope='Official PhysGaussian fixed-corotated MPM core with our compliant positional-boundary and black-box inverse adapter. Fits spatial material coefficient, two compliances, discrete support layout and gravity direction from the same three positional observations. No true physical labels, no spatial external-potential oracle. Full Gaussian appearance pipeline is not evaluated.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--object',type=int,default=1);p.add_argument('--attachment-width',type=float,default=0.);p.add_argument('--coarse-iterations',type=int,default=12);p.add_argument('--fine-iterations',type=int,default=15);p.add_argument('--seconds',type=float,default=2.);p.add_argument('--ramp-seconds',type=float,default=0.);p.add_argument('--fine-dt',type=float,default=.0001);p.add_argument('--fine-grid',type=int,default=64);p.add_argument('--fine-resolution',type=int,default=2);a=p.parse_args();run(a)
