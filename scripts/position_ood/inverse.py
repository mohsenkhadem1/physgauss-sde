"""Fit latent material/environment fields from position observations only."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.differentiable import DensePositionFEM
from scripts.phystwin.sequential_fixture import points
from scripts.phystwin.sequential_reference import write

def physical(z,environment=True):
    if environment:
        g=torch.stack([z[3],z[4],z.new_tensor(-1.)]);g=9.81*g/torch.linalg.vector_norm(g);amplitude=.1*z[5]
    else:g=z.new_tensor([0,0,-9.81]);amplitude=z.new_zeros(())
    theta=torch.cat([z[:3],g,amplitude[None]])
    if (environment and z.numel()>=7) or (not environment and z.numel()>=4):theta=torch.cat([theta,z[6:7] if environment else z[3:4]])
    if environment and z.numel()==11:
        center=z.new_tensor([.12,0,-.09])+z[7:10]*z.new_tensor([.06,.04,.04]);logwidth=z.new_tensor(np.log(.08))+.5*z[10];theta=torch.cat([theta,center,logwidth[None]])
    return theta

def fit(model,poses,observed,visible,layout,environment=True,iterations=80,initial=None,material_field=False,flexible_field=False):
    rest=points();ids=np.argmin(np.linalg.norm(model.geo.rest[:,None]-rest[None],axis=-1),axis=0)
    ids=model.tensor(ids[visible],torch.long);target=model.tensor(observed[:,visible]);ps=model.tensor(poses)
    z0=np.r_[np.log([20000,5000,5000]),0.,0.,0.] if environment else np.log([20000,5000,5000])
    if material_field:z0=np.r_[z0,0.]
    if flexible_field:
        assert environment and material_field
        z0=np.r_[z0,0.,0.,0.,0.];z0[5]=.5
    if initial is not None:z0=np.asarray(initial)
    bounds=[(np.log(3000),np.log(100000)),(np.log(300),np.log(60000)),(np.log(300),np.log(60000))]+([(-.8,.8),(-.8,.8),(-5,5)] if environment else [])
    if material_field:bounds+=[(-1.2,1.2)]
    if flexible_field:bounds+=[(-1.,1.),(-1.,1.),(-1.,1.),(-1.2,.8)]
    trace=[];best=[np.inf,None];started=time.time()
    def objective(z):
        var=model.tensor(z).requires_grad_()
        try:predictions=torch.stack([model.predict(physical(var,environment),pose,layout)[ids] for pose in ps])
        except (ValueError,RuntimeError):
            anchor=z0 if best[1] is None else best[1];delta=np.asarray(z)-anchor;return 1e12+1e6*float(delta@delta),2e6*delta
        loss=((predictions-target)/.0003).square().mean();loss.backward();value=float(loss.detach())
        if value<best[0]:best[:]=[value,np.array(z).copy()]
        trace.append(value);return value,var.grad.detach().cpu().numpy()
    result=minimize(objective,z0,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=iterations,ftol=1e-9,gtol=1e-5,maxls=30))
    z=best[1];return dict(z=z.tolist(),theta=physical(model.tensor(z),environment).detach().cpu().numpy().tolist(),layout=layout,environment=environment,loss=best[0],optimizer_success=bool(result.success),optimizer_message=str(result.message),evaluations=len(trace),iterations=int(result.nit),trace=trace,elapsed_seconds=time.time()-started)

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,default=2);p.add_argument('--mesh',type=int,default=2);p.add_argument('--iterations',type=int,default=80);p.add_argument('--environment',choices=['joint','material_only'],default='joint');a=p.parse_args()
    base=Path('runs/position-ood-v09');qualification=json.loads(Path(f'reports/position-ood-v09/pilot-{a.case}-qualification.json').read_text());data=np.load(base/f'pilot-{a.case}/n{qualification["mesh"]}-q3.npz');observed=data['field'][:3];poses=data['poses'][:3];visible=np.isclose(points()[:,1],-.04)
    # Development diagnostic uses exact observed coordinates to isolate inverse
    # model error. Production visual evaluation adds rendered tracking noise.
    model=DensePositionFEM(a.mesh);rows=[];destination=base/f'pilot-{a.case}/inverse-{a.environment}-n{a.mesh}.json'
    for layout in range(3):
        row=fit(model,poses,observed,visible,layout,a.environment=='joint',a.iterations);rows.append(row);write(destination,dict(stage='running',rows=rows));print('FIT',a.case,a.environment,layout,row['loss'],row['elapsed_seconds'],flush=True)
    best=min(rows,key=lambda r:r['loss']);ids=np.argmin(np.linalg.norm(model.geo.rest[:,None]-points()[None],axis=-1),axis=0)
    with torch.no_grad():pred=np.array([model.predict(model.tensor(best['theta']),model.tensor(pose),best['layout'])[ids].cpu().numpy() for pose in data['poses']])
    errors=1000*np.sqrt(np.mean((pred-data['field'])**2,axis=(1,2)));np.savez_compressed(destination.with_suffix('.npz'),predicted=pred,reference=data['field'],poses=data['poses'])
    write(destination,dict(stage='complete',scope='Development exact-coordinate inverse diagnostic; no test claims.',rows=rows,best=best,per_action_rmse_mm=errors.tolist()));print('RESULT',errors,flush=True)
if __name__=='__main__':main()
