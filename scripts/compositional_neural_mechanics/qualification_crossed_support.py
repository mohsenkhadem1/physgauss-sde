"""Prospective crossed-family DEVELOPMENT control: case0 soft saturated support.

Only interface family descriptor flips0→1; allbulk/geometry/pose/K/footprint
parameters retained. Original four-case qualification and failures stay intact.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import copy,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.compositional_neural_mechanics import qualification_nonlinear_range as stage
from scripts.compositional_neural_mechanics.qualification import ROOT,sha,write
OUT=ROOT/'runs/compositional-neural-mechanics-v16/crossed-soft-saturated-development'


def main():
    prior_root=ROOT/'runs/compositional-neural-mechanics-v16/nonlinear-range-development'
    if not (prior_root/'report.json').exists():raise ValueError('Original four-case stage must finish first')
    if OUT.exists():raise ValueError('Preserve prior crossed diagnostic')
    original=json.loads((prior_root/'manifest.json').read_text());stage.guard(original)
    plan=copy.deepcopy(original['development_plan']);case=copy.deepcopy(plan['cases'][0]);case['name']='development-crossed-soft-saturated'
    if any(z[2]!=0 for z in case['interface_latents']):raise ValueError('Expected original polynomial interface')
    for z in case['interface_latents']:z[2]=1.
    plan.update(role=__doc__,cases=[case])
    sources=dict(original['source_sha256'])
    for path in (Path(__file__),prior_root/'manifest.json',prior_root/'report.json'):sources[str(path.relative_to(ROOT))]=sha(path)
    manifest=dict(scope=__doc__,development_plan=plan,source_sha256=sources,bulk_checkpoint=original['bulk_checkpoint'],interface_checkpoint=original['interface_checkpoint'],
                  deadline_unix=time.time()+900,devices=['cuda:1'],workers=1,reset=original['reset'])
    write(OUT/'manifest.json',manifest);write(ROOT/'reports/compositional-neural-mechanics-v16/crossed-support-development-plan.json',dict(development_plan=plan,source_sha256=sources,manifest_sha256=sha(OUT/'manifest.json')))
    stage.OUT=OUT;stage.worker(0,'cuda:1');rows=json.loads((OUT/'case-0.json').read_text())['rows'];comparisons=[]
    for action in plan['actions']:
        for label,pair in [('mesh4to6',[('teacher',4),('teacher',6)]),('learned_same_mesh',[('shared',2),('teacher',2)]),('coarse_teacher_bias',[('teacher',2),('teacher',6)])]:
            records=[next(r for r in rows if (r['model'],r['n'],r['action'])==(*key,action)) for key in pair];record=dict(action=action,comparison=label)
            if any(r['state']!='complete' for r in records):record.update(complete=False,passed=False,reason='Failed solve')
            else:
                data=[]
                for r in records:
                    path=ROOT/r['artifact']
                    if sha(path)!=r['sha256']:raise ValueError('Completed crossed state changed')
                    with np.load(path) as a:data.append({k:a[k].copy() for k in a.files})
                a,b=data;field=float(np.sqrt(np.mean((a['field']-b['field'])**2)));force=np.linalg.norm(a['forces_N']-b['forces_N'],axis=1);torque=np.linalg.norm(a['torques_Nm']-b['torques_Nm'],axis=1)
                record.update(complete=True,field_rms_mm=1000*field,force_difference_N=force.tolist(),torque_difference_Nm=torque.tolist())
                if label=='mesh4to6':
                    g=plan['gates'];ft=g['force_absolute_N']+g['force_relative']*np.linalg.norm(b['forces_N'],axis=1);tt=g['torque_absolute_Nm']+g['torque_relative']*np.linalg.norm(b['torques_Nm'],axis=1)
                    record.update(passed=bool(field<=g['field_rms_m'] and (force<=ft).all() and (torque<=tt).all()),force_limits_N=ft.tolist(),torque_limits_Nm=tt.tolist())
            comparisons.append(record)
    stage.guard(manifest);report=dict(scope=__doc__,source_sha256=sources,manifest_sha256=sha(OUT/'manifest.json'),rows=rows,comparisons=comparisons,failed_solves=sum(r['state']!='complete' for r in rows),all_mesh_gates_pass=all(r['passed'] for r in comparisons if r['comparison']=='mesh4to6'))
    write(OUT/'report.json',report);write(ROOT/'reports/compositional-neural-mechanics-v16/crossed-support-qualification.json',report)
if __name__=='__main__':main()
