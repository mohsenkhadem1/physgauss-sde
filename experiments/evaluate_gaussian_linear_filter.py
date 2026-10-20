"""Validation of the explicit RGB-D analytical-load neural hybrid baseline."""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from evaluate_gaussian_inverse import load_model,load_public_observations
from gaussian_linear_filter import range_observations,linear_design,gaussian_control_prior,gaussian_linear_update


def design_prior(reference,samples,seed,device):
    """Sample the DECLARED training design, with new seeds, never query labels.

    This is an informative simulator-design baseline, separately identified from
    the learned prior. Geometry construction samples support/contact hypotheses;
    it performs no dynamics and opens no episode files.
    """
    from isaac_dataset import geometry
    from gaussian_operator import physical_vector,BOUNDARY_TYPES,FORCE_TYPES
    rng=np.random.default_rng(seed);physical=[];bc=[];force=[];controls=[]
    for _ in range(samples):
        b=int(rng.integers(4));f=int(rng.integers(4))
        material={'youngs_modulus_Pa':float(np.exp(rng.uniform(np.log(1500),np.log(20000)))),
                  'density_kg_m3':float(rng.uniform(400,850)),'poissons_ratio':float(rng.uniform(.28,.42)),
                  'linear_damping_s_inverse':float(rng.uniform(.35,.85)),'elasticity_damping':.02}
        for attempt in range(100):
            case={'cells':[12,6,4],'seed':int(rng.integers(10000000,20000000)),
                  'boundary_type':BOUNDARY_TYPES[b],'force_type':FORCE_TYPES[f],
                  'load_patch_nodes':9,'seconds':3.,'material':material}
            try:points,_,_,spec=geometry(case);break
            except ValueError:continue
        else:raise ValueError('Unable to sample a valid support/contact prior')
        if not np.allclose(points,reference['rest_points_m']):raise ValueError('Design prior is specific to the declared cuboid mesh')
        p=physical_vector(case,spec,points);physical.append(p);bc.append(b);force.append(f)
        direction=p[16:19]/np.linalg.norm(p[16:19]);scale=.25 if b==0 else 1.
        mean,covariance=gaussian_control_prior(b)
        mean[:3]=direction*.18*scale
        covariance[:3,:3]=scale**2*(.035**2*np.outer(direction,direction)+.003**2*(np.eye(3)-np.outer(direction,direction)))
        if b==3:mean[3:]=p[22:25];covariance[3:,3:]=np.eye(3)*.001**2
        controls.append((mean,covariance))
    return {'effective_physical':torch.tensor(np.array(physical)[None],device=device),
            'boundary_type':torch.tensor([bc],device=device),'force_type':torch.tensor([force],device=device)},controls


def infer(bundle,inputs,samples=64,seed=20261104,noise_m=.002,common_noise_m=.002,rays=12,prior_kind='learned'):
    if prior_kind=='learned' and bundle['checkpoint']['model_kwargs'].get('context_mode')!='prior':
        raise ValueError('An unconditional prior is required to avoid counting video twice')
    device=bundle['device'];generator=torch.Generator(device=device).manual_seed(seed)
    if prior_kind=='simulator-design':draw,control_priors=design_prior(inputs['reference'],samples,seed,device)
    elif prior_kind=='learned':
        context=torch.zeros((1,128),device=device)
        with torch.no_grad():draw=bundle['inverse'].posterior.sample(context,samples=samples,generator=generator)
        control_priors=None
    else:raise ValueError('Unknown prior kind')
    rows=range_observations(inputs,rays,seed)
    lengths=np.array([len(r['value_m']) for r in rows]);frame_ids=np.repeat(np.arange(len(rows)),lengths)
    starts=np.r_[0,np.cumsum(lengths)]
    points=np.concatenate([r['points_m'] for r in rows]);jacobian=np.concatenate([r['range_jacobian'] for r in rows])
    y=np.concatenate([r['value_m'] for r in rows]);times=np.array([r['time_s'] for r in rows])
    r=np.eye(len(y))*noise_m**2+(frame_ids[:,None]==frame_ids[None])*common_noise_m**2
    x=torch.tensor(points,device=device,dtype=torch.float32);t=torch.tensor(times,device=device,dtype=torch.float32)
    posterior=[];log_evidence=[]
    with torch.no_grad():
        for i in range(samples):
            physical=draw['effective_physical'][0,i:i+1]
            bc=draw['boundary_type'][0,i:i+1];force=draw['force_type'][0,i:i+1]
            offset,design=linear_design(bundle['forward'],inputs['reference'],physical,bc,force,x,t)
            offset=offset[0].cpu().numpy();design=design[0].cpu().numpy()
            # Each point was queried in its own frame's camera; only those
            # diagonal time/point blocks are observations.
            b=np.concatenate([np.einsum('nc,nc->n',offset[j,starts[j]:starts[j+1]]-points[starts[j]:starts[j+1]],jacobian[starts[j]:starts[j+1]]) for j in range(len(rows))])
            h=np.concatenate([np.einsum('ncd,nc->nd',design[j,starts[j]:starts[j+1]],jacobian[starts[j]:starts[j+1]]) for j in range(len(rows))])
            mean,covariance=gaussian_control_prior(int(bc[0])) if control_priors is None else control_priors[i]
            mean,covariance,evidence=gaussian_linear_update(y,b,h,mean,covariance,r)
            posterior.append({'physical':physical,'boundary_type':bc,'force_type':force,
                              'control_mean':mean,'control_covariance':covariance})
            log_evidence.append(evidence)
    log_evidence=np.array(log_evidence);weights=np.exp(log_evidence-log_evidence.max());weights/=weights.sum()
    return {'particles':posterior,'weights':weights,'observations':len(y),
            'missing_queries':sum(row['missing_queries'] for row in rows),
            'effective_sample_size':float(1/np.square(weights).sum()),'log_evidence':log_evidence}


def predict(bundle,posterior,reference,times,draws_per_particle=4,seed=20261105):
    rng=np.random.default_rng(seed);device=bundle['device']
    points=torch.as_tensor(reference['rest_points_m'],device=device,dtype=torch.float32)
    times=torch.as_tensor(times,device=device,dtype=torch.float32);predictions=[]
    with torch.no_grad():
        for row in posterior['particles']:
            offset,design=linear_design(bundle['forward'],reference,row['physical'],row['boundary_type'],row['force_type'],points,times)
            control=rng.multivariate_normal(row['control_mean'],row['control_covariance'],size=draws_per_particle)
            positions=offset[0,None]+torch.einsum('tncd,sd->stnc',design[0],torch.tensor(control,device=device,dtype=torch.float32))
            predictions.append(positions.cpu().numpy())
    predictions=np.concatenate(predictions)
    weights=np.repeat(posterior['weights']/draws_per_particle,draws_per_particle)
    return predictions,weights


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--prior-checkpoint',type=Path,required=True);p.add_argument('--forward-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--episodes',nargs='*')
    p.add_argument('--samples',type=int,default=64);p.add_argument('--rays',type=int,default=12)
    p.add_argument('--max-time-s',type=float);p.add_argument('--noise-mm',type=float,default=2.)
    p.add_argument('--prior-kind',choices=['learned','simulator-design'],default='learned')
    p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(2)
    a.output.mkdir(parents=True,exist_ok=False);bundle=load_model(a.prior_checkpoint,a.forward_checkpoint,a.device)
    plan=json.loads((a.dataset/'plan.json').read_text());rows=[]
    cases=[c for c in plan['episodes'] if c['split']=='validation' and (not a.episodes or c['episode_id'] in a.episodes)]
    for index,case in enumerate(cases):
        started=time.monotonic();episode=a.dataset/'episodes'/case['episode_id']
        inputs=load_public_observations(episode,'rgbd',max_time_s=a.max_time_s)
        posterior=infer(bundle,inputs,a.samples,seed=20261104+index,noise_m=a.noise_mm/1000,rays=a.rays,prior_kind=a.prior_kind)
        times=np.arange(61)/20
        predictions,weights=predict(bundle,posterior,inputs['reference'],times)
        mean=np.tensordot(weights,predictions,axes=1)
        np.savez_compressed(a.output/(case['episode_id']+'-prediction.npz'),times_s=times,positions_m=predictions,weights=weights)
        # Evaluation target is first opened after all inference/prediction.
        with np.load(episode/'truth/trajectory.npz') as raw:truth=raw['positions_m']
        gap=(times>1)&(times<2);rest=inputs['reference']['rest_points_m']
        row={'episode_id':case['episode_id'],'coordinate_rmse_m':float(np.sqrt(np.mean((mean-truth)**2))),
             'gap_coordinate_rmse_m':float(np.sqrt(np.mean((mean[gap]-truth[gap])**2))),
             'static_coordinate_rmse_m':float(np.sqrt(np.mean((rest-truth)**2))),
             'effective_sample_size':posterior['effective_sample_size'],'missing_queries':posterior['missing_queries'],
             'observations':posterior['observations'],'seconds_elapsed':time.monotonic()-started}
        print(json.dumps(row),flush=True);rows.append(row)
    result={'episodes':rows,'aggregate':{key:float(np.mean([row[key] for row in rows])) for key in ['coordinate_rmse_m','static_coordinate_rmse_m','gap_coordinate_rmse_m','effective_sample_size']},
            'qualification':'Validation only; unconditional flow marginal for nonlinear parameters, independent Gaussian control prior; approximate reference-normal RGBD likelihood; analytic load inference, noFEM/optimization',
            'arguments':{key:str(value) if isinstance(value,Path) else value for key,value in vars(a).items()}}
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['aggregate'],indent=2))


if __name__=='__main__':main()
