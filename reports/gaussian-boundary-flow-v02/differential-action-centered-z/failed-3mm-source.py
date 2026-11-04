"""Audit-driven non-affine position action, qualified before frozen-cohort scoring.

Only the future command is new. The v01 mechanics, observed prefix, fitted flow,
posterior samples and deterministic regressor are unchanged. This is a controlled
regression study, not a new unseen-object benchmark or a learned contact model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from physgauss.physics import deformation
from scripts.gaussian_boundary_flow.refined import RefinedBlock
from scripts.gaussian_boundary_flow.model import observe


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def target_profile(rest):
    """Centered quadratic z warp, using only the known fixed z-footprint width.

    Subtract its Gaussian-weighted 1D interpolation mean to suppress rigid force.
    No hidden support x-center, stiffness or material parameter enters the command.
    """
    from numpy.polynomial.legendre import leggauss
    z=rest[:,2];levels=torch.unique(z,sorted=True)
    if len(levels)<3:raise ValueError('Non-affine target needs at least three z levels')
    center=(levels.max()+levels.min())/2;half=(levels.max()-levels.min())/2
    nodal=2*((levels-center)/half).square()-1
    xi,wi=leggauss(12);xi=rest.new_tensor(xi);wi=rest.new_tensor(wi)
    left=levels[:-1,None];right=levels[1:,None]
    points=(left+right)/2+(right-left)*xi/2
    weights=(right-left)*wi/2*torch.exp(-.5*((points-center)/.045).square())
    interpolated=nodal[:-1,None]*(1-xi)/2+nodal[1:,None]*(1+xi)/2
    mean=(weights*interpolated).sum()/weights.sum()
    field=2*((z-center)/half).square()-1-mean
    field=field/field.abs().max()
    profile=torch.zeros_like(rest);profile[:,1]=field
    return profile


def rollout(block,z,amplitude=.003,*,full=False):
    dt=.0025;system,load,gravity=block.operators(z,dt);theta=block.parameters(z)
    Ks=system.stiffness-block.K[None]*theta[:,0].exp()[:,None,None]
    Ds=system.damping-torch.diag_embed(system.mass*.25)
    profile=target_profile(block.mesh.rest).flatten()[None]
    u=torch.zeros_like(load);v=u.clone();states=[u]
    for step in range(1,401):
        t=step*dt
        if t<=.4:
            command=.5*(1-torch.cos(z.new_tensor(2*torch.pi*2*t)))
            force=gravity+load*command
        else:
            phase=z.new_tensor((t-.4)/.6*torch.pi)
            target=amplitude*.5*(1-phase.cos())*profile
            velocity=amplitude*.5*torch.pi/.6*phase.sin()*profile
            force=gravity+torch.einsum('bij,bj->bi',Ks,target.expand(len(z),-1))+torch.einsum('bij,bj->bi',Ds,velocity.expand(len(z),-1))
        u,v,_=system.step(u,v,force)
        if step%10==0:states.append(u)
    result=torch.stack(states,1).reshape(len(z),41,-1,3)
    return result if full else result[:,:,block.corners]


def rigid_residual(reference,target):
    """Proper Kabsch alignment, metric only; row-vector coordinates [...,N,3]."""
    x=reference-reference.mean(-2,keepdim=True)
    y=target-target.mean(-2,keepdim=True)
    U,_,Vh=torch.linalg.svd(x.transpose(-1,-2)@y)
    signs=torch.ones(*U.shape[:-2],3,dtype=U.dtype,device=U.device)
    signs[...,-1]=torch.linalg.det(U@Vh)
    R=(U*signs[...,None,:])@Vh
    return target-(x@R+target.mean(-2,keepdim=True))


def qualification(block,z,amplitude):
    u=rollout(block,z,amplitude,full=True)
    released=block.rollout(z,kind='release',corners_only=False)
    assert torch.equal(u[:,:17],released[:,:17]),'Future action altered observed prefix'
    F=deformation(u+block.mesh.rest,block.mesh)
    strain=.5*(F+F.transpose(-1,-2))-torch.eye(3,dtype=u.dtype,device=u.device)
    delta=u[:,17:]-released[:,17:]
    residual=rigid_residual(released[:,17:]+block.mesh.rest,u[:,17:]+block.mesh.rest)
    rms=delta.square().mean().sqrt();nonrigid=residual.square().mean().sqrt()
    maxstrain=float(strain.abs().max());minimum_J=float(torch.linalg.det(F).min())
    return {'amplitude_m':amplitude,'max_abs_linear_strain':maxstrain,'minimum_J':minimum_J,
            'added_action_RMS_mm':float(rms*1000),'rigid_removed_action_RMS_mm':float(nonrigid*1000),
            'nonrigid_RMS_fraction':float(nonrigid/rms),
            'qualified':bool(maxstrain<=.1 and minimum_J>=.85 and nonrigid>=.0001 and nonrigid/rms>=.2)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');p.add_argument('--out',default='reports/gaussian-boundary-flow-v02/differential-action-centered-z');args=p.parse_args()
    if args.device not in ('cpu','cuda:2'):raise ValueError('This audit owns only CPU/GPU 2')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    device=args.device;out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    b=RefinedBlock(device=device)
    # No saved regression truths/posteriors are opened until this qualification ends.
    torch.manual_seed(53173);dev=torch.logit(torch.rand(24,6).clamp(.001,.999)).to(device)
    plan={'development_seed':53173,'development_cases':24,'candidate_amplitudes_m':[.003,.002,.001],
          'thresholds':{'max_abs_linear_strain':.1,'minimum_J':.85,'minimum_rigid_removed_RMS_m':.0001,'minimum_nonrigid_RMS_fraction':.2},
          'profile':'vertical quadratic in z, Gaussian-weighted interpolation mean removed; normalized peak amplitude; known z-width .045 m',
          'selection':'First amplitude in predefined descending list satisfying all development thresholds',
          'test_scope':'Frozen v01 128-world regression cohort; no claims of newly unseen objects',
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(out/'plan.json',plan)
    with torch.no_grad():
        qualified=[];chosen=None
        for amplitude in plan['candidate_amplitudes_m']:
            row=qualification(b,dev,amplitude);qualified.append(row);print('development',row,flush=True)
            if row['qualified']:chosen=amplitude;break
        write(out/'development.json',{'checks':qualified,'selected_amplitude_m':chosen})
        if chosen is None:
            write(out/'results.json',{'status':'not_qualified','reason':'No predefined amplitude passes deformation/informativeness thresholds','development':qualified});return
        root=Path('runs/gaussian-boundary-flow-refined-v03')
        d=torch.load(root/'evaluation-reverse.pt',weights_only=True,map_location=device)
        bank=torch.load(root/'training.pt',weights_only=True,map_location=device)
        truth=d['truth'];draws=d['posterior']
        net=nn.Sequential(nn.Linear(144,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,6)).to(device)
        net.load_state_dict(torch.load(root/'point-baseline.pt',weights_only=True,map_location=device));net.eval()
        prefix=b.rollout(truth)
        context=(observe(prefix,generator=torch.Generator(device=device).manual_seed(1981))-bank['mean'])/bank['std']
        point=net(context);target=rollout(b,truth,chosen)
        assert torch.equal(target[:,:17],prefix[:,:17])
        rows={};means={}
        for name,samples in [('flow',draws),('neural_regressor',point[:,None])]:
            flat=samples.flatten(0,1)
            pred=torch.cat([rollout(b,flat[i:i+128],chosen)for i in range(0,len(flat),128)]).reshape(len(truth),samples.shape[1],41,8,3)
            mean=pred.mean(1);error=(mean[:,17:]-target[:,17:]).square().mean((1,2,3)).sqrt()*1000
            shape=rigid_residual(mean[:,17:]+b.mesh.rest[b.corners],target[:,17:]+b.mesh.rest[b.corners]).square().mean((1,2,3)).sqrt()*1000
            rows[name]={'future_coordinate_RMSE_mm':float(error.mean()),'future_rigid_aligned_RMSE_mm':float(shape.mean()),
                        'per_case_coordinate_mm':error.tolist(),'per_case_rigid_aligned_mm':shape.tolist()}
            means[name]=mean.cpu()
        result={'status':'evaluated','selected_amplitude_m':chosen,'development':qualified,'regression_cases':len(truth),
                'prefix_identical':True,'no_inverse_refitting':True,'methods':rows,
                'regression_motion_diagnostics':qualification(b,truth,chosen),
                'checkpoint_sha256':hashlib.sha256((root/'checkpoint.pt').read_bytes()).hexdigest(),
                'scope':plan['test_scope']}
        write(out/'results.json',result)
        torch.save({'target':target.cpu(),'prediction':means,'amplitude_m':chosen},out/'predictions-local.pt')
        print('results',rows,flush=True)

if __name__=='__main__':main()
