"""Local predictive sensitivity, empirical calibration, and positional probing.

The Gaussian approximation is local and is not claimed to be an exact physical
posterior. Calibration is measured on separate objects, including OOD groups.
"""
from pathlib import Path
import json,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.differentiable import DensePositionFEM
from scripts.position_ood.inverse import physical
from scripts.position_ood.study import BASE,within_deadline
from scripts.phystwin.sequential_fixture import points
from scripts.phystwin.sequential_reference import write


def sensitivity(model,z,pose,layout,environment=True,epsilon=1e-5):
    """Implicit sensitivity using finite differences of the residual, not solves."""
    if hasattr(model,'coarse'):
        uf,jf=sensitivity(model.fine,z,pose,layout,environment,epsilon);uc,jc=sensitivity(model.coarse,z,pose,layout,environment,epsilon)
        return model.fine_weight*uf-model.coarse_weight*model.interpolation@uc, model.fine_weight*jf-model.coarse_weight*(model.interpolation@jc.reshape(len(jc),-1)).reshape(jf.shape)
    with torch.no_grad():
        z=model.tensor(z);pose=model.tensor(pose);theta=physical(z,environment);u,H=model.solve(theta,pose,layout)
        columns=[]
        for k in range(len(z)):
            a=z.clone();b=z.clone();a[k]+=epsilon;b[k]-=epsilon
            ra=model.evaluate(u,physical(a,environment),pose,layout)[1];rb=model.evaluate(u,physical(b,environment),pose,layout)[1];columns.append(((ra-rb)/(2*epsilon)).reshape(-1))
        J=torch.linalg.solve(H,-torch.stack(columns,dim=1)).reshape(*u.shape,len(z))
        return u.cpu().numpy(),J.cpu().numpy()


def covariance(jacobian,noise_sigma=.0003):
    p=jacobian.shape[-1];prior=np.r_[1.,1.,1.,.3,.3,2.,1.,.6,.6,.6,.8][:p]
    A=jacobian.reshape(-1,p)/noise_sigma;precision=A.T@A+np.diag(1/prior**2)
    return np.linalg.inv(precision)


def probe_utilities(J,C,visible_ids,candidates=(4,5,6,7,8),target=9,noise_sigma=.0003):
    """Expected local target-variance reduction without seeing probe outcomes."""
    B=J[target].reshape(-1,J.shape[-1]);before=float(np.einsum('ip,pq,iq->',B,C,B));utilities=[]
    for action in candidates:
        A=J[action,visible_ids].reshape(-1,J.shape[-1])/noise_sigma
        after=np.linalg.inv(np.linalg.inv(C)+A.T@A);remaining=float(np.einsum('ip,pq,iq->',B,after,B));utilities.append(before-remaining)
    return dict(candidates=list(candidates),target=target,expected_variance_reduction_m2=utilities,selected=int(candidates[int(np.argmax(utilities))]))


def run(split,i,method='joint',mesh=2,prefix=(0,1,2)):
    within_deadline();path=BASE/f'inverse/{split}-{i}/{method}-n{mesh}-p{len(prefix)}.json';record=json.loads(path.read_text());best=record['best'];obs=np.load(BASE/f'observations/{split}-{i}.npz');ids=obs['visible_ids'];model=DensePositionFEM(mesh,field_frame='material' if method=='material_frame' else 'world')
    if method.startswith('extrapolated'):
        from scripts.position_ood.extrapolated import ExtrapolatedPositionFEM
        model=ExtrapolatedPositionFEM(field_frame='material' if method=='extrapolated_material_frame' else 'world')
    sample_ids=np.argmin(np.linalg.norm(model.geo.rest[:,None]-points()[None],axis=-1),axis=0)
    means=[];gradients=[]
    for pose in obs['poses']:
        within_deadline();u,J=sensitivity(model,best['z'],pose,best['layout'],best['environment']);means.append(u[sample_ids]);gradients.append(J[sample_ids])
    J=np.array(gradients);C=covariance(J[np.array(prefix)][:,ids]);var=np.einsum('...p,pq,...q->...',J,C,J);std=np.sqrt(np.maximum(var,0))
    dest=path.with_name(path.stem+'-uncertainty');np.savez_compressed(str(dest)+'.npz',mean=means,std=std,jacobian=J,covariance=C)
    plans=probe_utilities(J,C,ids,candidates=(3,4,5,6,8),target=7) if split.startswith('transfer_') else probe_utilities(J,C,ids);write(Path(str(dest)+'.json'),dict(scope='Local Gaussian sensitivity approximation with weak parameter-scale regularization. No claim of calibrated physical posterior; predictive scale calibrated on separate objects.',prefix=list(prefix),**plans));return plans

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--object',type=int,default=1);p.add_argument('--method',default='joint');a=p.parse_args();print(run(a.split,a.object,a.method))
