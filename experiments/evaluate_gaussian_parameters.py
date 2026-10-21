"""Validation-only physical-parameter credibility, distinct from motion accuracy.

The validation set also selects checkpoints: reported empirical coverage is an
optimistic development diagnostic, not a held-out calibration guarantee.
Inactive nuisance padding is excluded from physical control error summaries.
"""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from train_gaussian_inverse import collect_cases,apply_catalog
from evaluate_gaussian_inverse import encode_episode,posterior_samples,load_observation_posterior
from gaussian_operator import physical_vector,category_labels


def scalar_summary(samples,target):
    values=np.asarray(samples);lower,median,upper=np.quantile(values,[.05,.5,.95])
    return {'target':float(target),'median':float(median),'lower90':float(lower),'upper90':float(upper),
            'covered90':bool(lower<=target<=upper),'absolute_error':float(abs(median-target)),
            'width90':float(upper-lower)}


def categorical_summary(probabilities,target):
    p=np.asarray(probabilities);truth=np.eye(len(p))[target]
    return {'target':int(target),'probabilities':p.tolist(),'prediction':int(p.argmax()),
            'correct':bool(p.argmax()==target),'brier':float(np.sum((p-truth)**2)),
            'negative_log_probability':float(-np.log(max(p[target],1e-15)))}


def evaluate(roots,catalog,inverse_checkpoint,output,samples=512):
    torch.set_num_threads(2);model,checkpoint=load_observation_posterior(inverse_checkpoint,'cpu')
    bundle={'inverse':model,'checkpoint':checkpoint,'device':torch.device('cpu')}
    records=[r for r in apply_catalog(collect_cases(roots),catalog) if r['recipe']['split']=='validation']
    output=Path(output);output.mkdir(parents=True,exist_ok=False);rows=[]
    (output/'source.py').write_bytes(Path(__file__).read_bytes())
    for index,record in enumerate(records):
        encoded=encode_episode(bundle,record['path']);draw=posterior_samples(bundle,encoded,samples,20261210+index)
        with torch.no_grad():
            bc_logits,force_logits=model.posterior.categorical_logits(encoded['context'])
            if hasattr(model.posterior,'joint_head'):
                joint_prob=model.posterior.joint_head(encoded['context']).softmax(-1)[0].numpy()
            else:joint_prob=(bc_logits.softmax(-1)[0,:,None]*force_logits.softmax(-1)[0,None,:]).flatten().numpy()
        physical=draw['physical'].cpu().numpy();effective=draw['effective_physical'].cpu().numpy()
        # Private labels are first opened after the public-only posterior is fixed.
        spec=json.loads((record['path']/'truth/boundaries.json').read_text())
        target=physical_vector(record['recipe'],spec,encoded['inputs']['reference']['rest_points_m'])
        bc,force=category_labels(record['recipe'])
        row={'key':record['key'],'material':{},'boundary_type':categorical_summary(bc_logits.softmax(-1)[0].numpy(),bc),
             'force_type':categorical_summary(force_logits.softmax(-1)[0].numpy(),force),
             'joint_type':categorical_summary(joint_prob,bc*4+force),'active_control_mean_errors':{}}
        if 'support_face' in draw:
            axis,sign=spec['support_faces'][0];support_truth=int(axis+(3 if sign>0 else 0))
            row['support_face']=categorical_summary(np.bincount(draw['support_face'].numpy(),minlength=6)/samples,support_truth)
            for load in range(2):
                normal=np.asarray(spec['reference_load_normals'][load]);axis=int(abs(normal).argmax())
                truth=axis+(3 if normal[axis]>0 else 0)
                row[f'load{load}_face']=categorical_summary(np.bincount(draw['load_faces'][:,load].numpy(),minlength=6)/samples,truth)
        for name,dimension,logged in [('youngs_modulus_Pa',0,True),('density_kg_m3',1,True),
                                      ('poissons_ratio',2,False),('damping_s_inverse',3,True)]:
            values=np.exp(physical[:,dimension]) if logged else physical[:,dimension]
            truth=np.exp(target[dimension]) if logged else target[dimension]
            row['material'][name]=scalar_summary(values,truth)
        # Both attached load handles alter inertia/rigidity even when force1=0.
        controls=[('support0_m',4,7),('load0_m',10,13),('load1_m',13,16),('force0_N',16,19)]
        if bc==2:controls.append(('support1_m',7,10))
        if force==3:controls.append(('force1_N',19,22))
        if bc==3:controls.append(('moving_support_m',22,25))
        for name,start,stop in controls:
            row['active_control_mean_errors'][name]=float(np.linalg.norm(effective[:,start:stop].mean(0)-target[start:stop]))
        extra={k:draw[k].numpy() for k in ('chart','support_face','load_faces') if k in draw}
        np.savez_compressed(output/(record['recipe']['episode_id']+'.npz'),physical=physical,effective_physical=effective,
                            boundary_type=draw['boundary_type'].numpy(),force_type=draw['force_type'].numpy(),target=target,**extra)
        rows.append(row)
    summary={'material':{},'categorical':{},'active_control_mean_errors':{}}
    for name in rows[0]['material']:
        summary['material'][name]={stat:float(np.mean([r['material'][name][stat] for r in rows])) for stat in ('covered90','absolute_error','width90')}
    for name in ('boundary_type','force_type','joint_type','support_face','load0_face','load1_face'):
        if name not in rows[0]:continue
        summary['categorical'][name]={stat:float(np.mean([r[name][stat] for r in rows])) for stat in ('correct','brier','negative_log_probability')}
    for name in sorted({k for r in rows for k in r['active_control_mean_errors']}):
        values=[r['active_control_mean_errors'][name] for r in rows if name in r['active_control_mean_errors']]
        summary['active_control_mean_errors'][name]={'mean_error':float(np.mean(values)),'cases':len(values)}
    result={'summary':summary,'episodes':rows,'samples':samples,'inverse_checkpoint':str(inverse_checkpoint),
            'checkpoint_sha256':hashlib.sha256(Path(inverse_checkpoint).read_bytes()).hexdigest(),
            'qualification':__doc__,'density_measure':checkpoint.get('density_measure','Raw25D flow including nuisance padding; not physical posterior calibration')}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',nargs='+',required=True)
    p.add_argument('--catalog',required=True);p.add_argument('--inverse-checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=512);a=p.parse_args();print(json.dumps(evaluate(**vars(a))['summary'],indent=2))
