"""Predeclared signed-traction excitation/observability diagnostic, not acquisition test.

Uses only nominal/corner parameter configurations. Existing frozen campaigns and
mechanics are read-only. All probes have sum absolute actuator coefficients .08N.
Balanced actions contain pulling traction and are not unilateral pokes.
"""
from pathlib import Path
import argparse,json,hashlib,itertools
import numpy as np
import torch
from .mechanics import LinearGaussianBody,PARAMETERS
from . import bank

ROOT=bank.ROOT

def kabsch(rest,displacement,mass):
    w=mass/mass.sum();xc=rest-(w[:,None]*rest).sum(0);out=[]
    for d in displacement:
        y=rest+d;yc=y-(w[:,None]*y).sum(0);U,_,Vh=np.linalg.svd(xc.T@(w[:,None]*yc));V=Vh.T;D=np.eye(3);D[2,2]=np.linalg.det(V@U.T);R=V@D@U.T
        residual=yc-xc@R.T;nr=float(np.sqrt(np.sum(w[:,None]*residual**2)));total=float(np.sqrt(np.sum(w[:,None]*d**2)))
        out.append(dict(nonrigid_rms_mm=nr*1000,total_rms_mm=total*1000,nonrigid_rms_fraction=nr/max(total,1e-30),rigid_rotation_rad=float(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');args=ap.parse_args();torch.set_num_threads(1)
    coefficients=np.concatenate([np.eye(6)*.08,np.array([[-.04,0,0,0,.04,0],[0,-.04,0,0,0,.04],[.02,0,-.04,0,.02,0],[0,.02,0,-.04,0,.02]])])
    names=[f'one_sided_{i}' for i in range(6)]+['top_dipole','side_dipole','top_center_vs_ends','side_center_vs_ends'];axes=np.array([2,1,2,1,2,1,2,1,2,1])
    bounds=bank.BOUNDS.copy();theta=np.concatenate([bounds.mean(1)[None],np.array(list(itertools.product(*[[lo,hi] for lo,hi in bounds])))])
    # Known calibrated probe responses do not depend on the latent unknown-load
    # location theta[4]. Retain that exact null direction explicitly.
    parameter_scale=(bounds[:,1]-bounds[:,0])/2
    increments=np.array([1e-4,1e-4,1e-6,1e-4,1e-6])
    points=bank.points();indices=[13,1,18];noise=bank.NOISE
    record=dict(scope='Read-only pre-freeze candidate diagnostic: nominal + all32 parameter corners. Not learned policy or final-test improvement.',parameters=list(PARAMETERS),parameter_scale_half_range=parameter_scale.tolist(),noise_m=noise,forces=dict(names=names,coefficients_N=coefficients.tolist(),basis=bank.probe_specs()[:6],all_L1_coefficient_budget_N=.08,qualification='Signed traction: negative coefficients pull; balanced probes are not unilateral pokes.'),selected_points={str(i):points[i].tolist() for i in indices},results=[])
    for shape in ['block','step']:
        body=LinearGaussianBody(shape,(8,4,4),device=args.device);C=torch.as_tensor(coefficients,dtype=torch.float64,device=args.device)
        with torch.no_grad():
            base=body.solve(theta)['probe_response'];displacements=torch.einsum('bncp,ap->banc',base,C)
            values=body.point_response(base,points).cpu().numpy()
            scalar=np.einsum('bpcl,al->bapc',values,coefficients)
            strain=body.strain(displacements).cpu().numpy();strainmax=np.abs(np.linalg.eigvalsh(strain)).max((-1,-2))
            rest=body.points.cpu().numpy();mass=body.mass.cpu().numpy();actual=displacements.cpu().numpy()
            jac=np.empty((len(theta),len(names),len(points),5))
            for j,step in enumerate(increments):
                plus=theta.copy();minus=theta.copy();plus[:,j]+=step;minus[:,j]-=step
                rp=body.point_response(body.solve(plus)['probe_response'],points).cpu().numpy();rm=body.point_response(body.solve(minus)['probe_response'],points).cpu().numpy()
                derivative=np.einsum('bpcl,al->bapc',(rp-rm)/(2*step),coefficients)
                jac[:,:,:,j]=np.take_along_axis(derivative,axes[None,:,None,None],axis=3)[...,0]
        for ci,th in enumerate(theta):
            decomp=kabsch(rest,actual[ci],mass)
            row=dict(shape=shape,case=ci,kind='nominal' if ci==0 else 'corner',theta=th.tolist(),excitation=[dict(name=name,max_principal_strain=float(strainmax[ci,a]),**decomp[a]) for a,name in enumerate(names)],observability=[])
            for pi in indices:
                for label,selected in [('one_sided',list(range(6))),('balanced',list(range(6,10))),('top_center_vs_ends',[8]),('side_center_vs_ends',[9]),('all',list(range(10)))]:
                    J=jac[ci,selected,pi]*parameter_scale[None]/noise
                    singular=np.linalg.svd(J,compute_uv=False)
                    # Material sensitivity after projecting away support center
                    # and support stiffness directions within this action set.
                    support=J[:,2:4];material=J[:,:2];residual=material-support@np.linalg.lstsq(support,material,rcond=1e-10)[0]
                    material_singular=np.linalg.svd(residual,compute_uv=False)
                    row['observability'].append(dict(point=pi,actions=label,count=len(selected),whitened_half_range_jacobian_singular_values=singular.tolist(),material_after_support_projection_singular_values=material_singular.tolist(),known_probe_load_location_column_norm=float(np.linalg.norm(J[:,4])),mean_per_scalar_information_trace=float(np.sum(J*J)/len(selected))))
            row['equal_four_action_comparison']=[]
            for pi in indices:
                subsets=[]
                for selected in itertools.combinations(range(6),4):
                    J=jac[ci,list(selected),pi]*parameter_scale[None]/noise
                    material=J[:,:2];support=J[:,2:4]
                    residual=material-support@np.linalg.lstsq(support,material,rcond=1e-10)[0]
                    singular=np.linalg.svd(residual,compute_uv=False)
                    subsets.append(dict(actions=list(selected),minimum_material_singular_value=float(singular[-1]),mean_per_scalar_information_trace=float(np.sum(J*J)/4)))
                row['equal_four_action_comparison'].append(dict(point=pi,one_sided_subsets=subsets,median_one_sided_minimum_material_singular_value=float(np.median([s['minimum_material_singular_value'] for s in subsets])),best_one_sided_minimum_material_singular_value=max(s['minimum_material_singular_value'] for s in subsets)))
            record['results'].append(row)
    def aggregate(shape,group):
        rows=[r for r in record['results'] if r['shape']==shape];inds=list(range(6)) if group=='one_sided' else list(range(6,10))
        return dict(shape=shape,group=group,max_strain=max(r['excitation'][i]['max_principal_strain'] for r in rows for i in inds),median_nonrigid_fraction=float(np.median([r['excitation'][i]['nonrigid_rms_fraction'] for r in rows for i in inds])),median_nonrigid_rms_mm=float(np.median([r['excitation'][i]['nonrigid_rms_mm'] for r in rows for i in inds])))
    record['aggregate']=[aggregate(shape,group) for shape in ['block','step'] for group in ['one_sided','balanced']]
    record['source_sha256']={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scripts/joint_field_identification/mechanics.py':hashlib.sha256(Path(__file__).with_name('mechanics.py').read_bytes()).hexdigest()}
    path=ROOT/'reports/joint-field-identification-v01/balanced-probe-development.json';path.write_text(json.dumps(record,indent=2));print(json.dumps(record['aggregate'],indent=2))
    for r in record['results']:
        if r['case']==0:
            print(r['shape'],'nominal')
            for o in r['observability']:
                if o['actions'] in ['one_sided','balanced']:print(o)

if __name__=='__main__':main()
