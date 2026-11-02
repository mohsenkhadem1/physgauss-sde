"""Isolated replay of one recorded validation runtime-initialization exception.
No model, parameter, solver or action selection changes; original report retained.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,time
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics.oracle_campaign import ROOT,REPORT,inventory,sha
from scripts.compositional_neural_mechanics.dataset_runner import write,npz
from scripts.compositional_neural_mechanics.forward_evaluation import solve_forward,score_prediction
from scripts.compositional_neural_mechanics.feature_pack import load_frozen_laws
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec

def main():
    torch.set_num_threads(1);out=ROOT/'runs/compositional-neural-mechanics-v16/validation-runtime-recovery'
    if out.exists():raise ValueError('Preserve recovery attempt')
    original_path=REPORT/'development-forward-comparison-results.json';original=json.loads(original_path.read_text())
    failures=[r for r in original['rows'] if r['state']!='complete']
    if len(failures)!=1 or 'lazy wrapper should be called at most once' not in failures[0]['error']:raise ValueError('Only the recorded runtime exception is eligible')
    failed=failures[0];dataset=json.loads((REPORT/'training-data-plan.json').read_text());oldplan=json.loads((REPORT/'development-forward-comparison-plan.json').read_text())
    record=next(r for r in dataset['scenes'] if r['scene']['name']==failed['scene_id']);action=next(a for a in record['actions'] if a['name']==failed['action'])
    sources=inventory();sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    plan=dict(original_report_sha256=sha(original_path),original_failure=failed,checkpoints=oldplan['checkpoints'],source_sha256=sources,device='cuda:1',n=4,
              rationale='Replay same single call in an isolated process; original threaded failure retained. Exact root cause not established from the exception string.')
    out.mkdir(parents=True);write(REPORT/'validation-runtime-recovery-plan.json',plan)
    cps=plan['checkpoints'];bulk,interface=load_frozen_laws(*(ROOT/cps[k]['path'] for k in ('bulk','interface')),{k:cps[k]['sha256'] for k in cps},'cuda:1')
    path=ROOT/'runs/compositional-neural-mechanics-v16/training-data-n4/scenes'/record['scene']['name']/action['name']/'labels.npz'
    with np.load(path) as data:q=data['material_query'].copy();reference={k:data[k].copy() for k in ('displacement','wrenches')}
    start=time.monotonic();value=solve_forward(record['scene'],action['pose'],q,kind=failed['method'],n=4,device='cuda:1',bulk_law=bulk,interface_law=interface)
    result=dict(scene_id=failed['scene_id'],action=failed['action'],method=failed['method'],state='complete',scores=score_prediction(value,reference,GeometrySpec(**record['scene']['geometry']).map(q)),seconds=time.monotonic()-start)
    target=out/'prediction.npz';npz(target,material_query=q,displacement=value['displacement'],wrenches=value['wrenches'])
    combined=[r for r in original['rows'] if r['state']=='complete']+[result];summary={}
    for method in ('shared_laws','neo_hookean'):
        rows=[r for r in combined if r['method']==method]
        if len(rows)!=24:raise ValueError('All declared validation commands required')
        summary[method]={key:float(np.mean([r['scores'][key] for r in rows])) for key in rows[0]['scores']}
    if sha(original_path)!=plan['original_report_sha256'] or any(sha(ROOT/path)!=digest for path,digest in sources.items()):raise ValueError('Recovery source/evidence changed')
    write(REPORT/'validation-runtime-recovery-results.json',dict(plan=plan,recovered_row=result,summary=summary,original_completed=47,original_runtime_failed=1,isolated_recovered=1,
          prediction_sha256=sha(target),reference_sha256=sha(path),scope='Same-call development runtime recovery; original failure and all other outcomes retained'))

if __name__=='__main__':main()
