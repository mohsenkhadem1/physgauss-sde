"""Frozen three-plane prefix-only feasibility screen, no evaluator imports."""
import json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.push_backend import PrefixPlanePushBackend,estimate_plane,NORMAL_POLICY
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.external_benchmark.data import fps


def main():
    torch.set_num_threads(2)
    out=Path('reports/gaussian-boundary-flow-v02/push-prefix-feasibility');out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():raise RuntimeError('Frozen experiment already exists')
    package=Path('runs/gaussian-boundary-flow-v02-expansion-affine-inputs/single_push_sloth')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    plan=dict(case='single_push_sloth',scope='Prefix-only geometric/mechanical feasibility; no suffix evaluation, fitting or posterior training',
              package=str(package),manifest_sha256=sha(package/'manifest.json'),normal_policy=NORMAL_POLICY,
              worlds='Three fixed plane offsets, shared median prior logits z=0; not random search',
              standoffs_m=[-.01,0.,.01],prefix_frames=12,point_ids=48,device='cpu',
              failure_policy='Preserve failures without fallback normals or modified solver gates',
              source_sha256={str(p):sha(p) for p in [Path(__file__),Path('scripts/gaussian_boundary_flow_v02/push_backend.py'),Path('scripts/gaussian_boundary_flow_v02/unilateral_attachment.py')]})
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    data=load_package(package);frames=np.unique(np.linspace(0,data['train_end']-1,12,dtype=int));ids=fps(data['observed_initial'],48)
    rows=[];start=time.time()
    for offset in plan['standoffs_m']:
        row=dict(standoff_m=offset,completed=False)
        try:
            _,_,info=estimate_plane(data,frames,standoff_m=offset);row['normal_hypothesis']=info
            b=PrefixPlanePushBackend(data,'cpu',prefix_frames=frames,standoff_m=offset)
            positions,_=b.rollout(np.zeros(b.parameter_dim),frames)
            obs=b.observation_features(positions,frames,ids);target=data['observed'][frames][:,ids]
            mask=data['observed_mask'][frames][:,ids]
            row.update(completed=True,prefix_absolute_coordinate_rmse_mm=float(np.sqrt(np.mean(((obs-target)[mask])**2))*1000),
                       prefix_relative_coordinate_rmse_mm=float(np.sqrt(np.mean((((obs-obs[:1])-(target-target[:1]))[mask])**2))*1000),
                       path=b.last_path_diagnostics,parameter_logits=[0.]*b.parameter_dim)
        except (RuntimeError,ValueError) as e:row['failure']=str(e)
        rows.append(row);(out/f'offset-{offset:+.3f}.json').write_text(json.dumps(row,indent=2)+'\n')
        print(row,flush=True)
    result=dict(plan=plan,rows=rows,seconds=time.time()-start,held_target_access=False,
                interpretation='Feasibility only, no plane selection or inferred-normal accuracy claim')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
