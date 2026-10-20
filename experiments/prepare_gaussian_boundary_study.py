"""Freeze fresh train/validation/test recipes before fitting learned models."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from isaac_dataset import make_plan,geometry,digest_json,BOUNDARIES,LOADS


def make_training_plan():
    template=make_plan()['episodes'][0];cases=[];rejected=[]
    for split,repeats,start in [('train',8,531000),('validation',2,541000),('test',4,551000)]:
        for rep in range(repeats):
            for li,load in enumerate(LOADS):
                for bi,bc in enumerate(BOUNDARIES):
                    index=16*rep+4*li+bi;seed=start+index
                    rng=np.random.default_rng(seed+100000)
                    case=copy.deepcopy(template)
                    case.update(episode_id={'train':'t','validation':'v','test':'z'}[split]+f'{index:04d}',
                        group_id=f'{split}-{index:04d}',split=split,seed=seed,boundary_type=bc,force_type=load,
                        material={'youngs_modulus_Pa':float(np.exp(rng.uniform(np.log(1500),np.log(20000)))),
                            'poissons_ratio':float(rng.uniform(.28,.42)),
                            'density_kg_m3':float(rng.uniform(400,850)),
                            'linear_damping_s_inverse':float(rng.uniform(.35,.85)),'elasticity_damping':.02},
                        qualification='Independent layout; fixed-discretization simulated supervision; no continuum accuracy claim')
                    # Reject only impossible patch construction, before any simulator
                    # trajectory/image is generated. Never replace a failed physics case.
                    for attempt in range(50):
                        case['seed']=seed+attempt*1000000
                        try:geometry(case)
                        except ValueError as error:
                            rejected.append({'episode_id':case['episode_id'],'seed':case['seed'],'reason':str(error)})
                        else:break
                    else:raise RuntimeError('Could not construct declared patch geometry')
                    case.pop('recipe_sha256',None);case['recipe_sha256']=digest_json(case);cases.append(case)
    plan={'dataset':'gaussian-boundary-neural-v01','episodes':cases,'episode_count':len(cases),
          'design':'128 fresh train,32 validation,64 final test; continuous stiffness independent of geometry; original12test retained separately',
          'split_rule':'Disjoint layout seeds and groups; no test pixels or physical labels enter fitting/checkpoint selection',
          'geometry_construction_rejections':rejected,
          'source_policy':'Freeze source and plan; preserve all physics/capture failures without outcome-based replacement',
          'generation_qc':'Finite states; tet volume ratio>=0.1; support drift<=0.2mm; saved sensors align with cameras and mesh'}
    plan['plan_sha256']=digest_json(plan);return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plan=make_training_plan();args.output.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'episodes':len(plan['episodes']),'geometry_rejections':len(plan['geometry_construction_rejections']),'sha256':plan['plan_sha256']}))
