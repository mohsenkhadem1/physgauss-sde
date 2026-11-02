"""Validation-only forward comparison against existing n4 teacher labels."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import concurrent.futures,json,time
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics.forward_evaluation import solve_forward,score_prediction
from scripts.compositional_neural_mechanics.feature_pack import load_frozen_laws
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.dataset_runner import verify_complete,write,npz
from scripts.compositional_neural_mechanics.oracle_campaign import inventory,sha
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';OUT=ROOT/'runs/compositional-neural-mechanics-v16/development-forward-comparison'

def main():
    torch.set_num_threads(1)
    if OUT.exists():raise ValueError('Preserve previous validation evaluation')
    OUT.mkdir(parents=True);dataset_path=R/'training-data-plan.json';dataset=json.loads(dataset_path.read_text());direct=json.loads((R/'direct-training-plan.json').read_text())
    records=[r for r in dataset['scenes'] if r['role']=='validation']
    sources=inventory();sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    plan=dict(role=__doc__,scene_ids=[r['scene']['name'] for r in records],methods=['shared_laws','neo_hookean'],n=4,
              source_sha256=sources,dataset_sha256=sha(dataset_path),checkpoints=direct['checkpoints'],devices=['cuda:1','cuda:2'],scope='24 known-command validation responses. Same n4 teacher; not a fine-reference reserved test.')
    write(R/'development-forward-comparison-plan.json',plan)
    def worker(index):
        device=f'cuda:{index+1}';cps=plan['checkpoints'];bulk,interface=load_frozen_laws(*(ROOT/cps[k]['path'] for k in ('bulk','interface')),{k:cps[k]['sha256'] for k in cps},device)
        rows=[]
        for record in records[index::2]:
            scene=record['scene'];geometry=GeometrySpec(**scene['geometry'])
            for action in record['actions']:
                folder=ROOT/'runs/compositional-neural-mechanics-v16/training-data-n4/scenes'/scene['name']/action['name']
                identity=dict(scene_id=scene['name'],split='validation',action=action['name'],prescribed_pose=action['pose'],n=4,order=3)
                complete=verify_complete(folder,sha(dataset_path),dataset['source_sha256'],identity)
                if complete is None:raise ValueError('Verified validation reference required')
                with np.load(folder/'labels.npz') as data:q=data['material_query'].copy();reference={k:data[k].copy() for k in ('displacement','wrenches')}
                for method in plan['methods']:
                    row=dict(scene_id=scene['name'],action=action['name'],method=method);start=time.monotonic()
                    try:
                        value=solve_forward(scene,action['pose'],q,kind=method,n=4,device=device,bulk_law=bulk,interface_law=interface)
                        row.update(state='complete',scores=score_prediction(value,reference,geometry.map(q)),diagnostics=value['diagnostics'])
                        target=OUT/(scene['name']+'-'+action['name']+'-'+method+'.npz');npz(target,material_query=q,displacement=value['displacement'],wrenches=value['wrenches']);row['sha256']=sha(target)
                    except Exception as error:row.update(state='failed',error=repr(error))
                    row['seconds']=time.monotonic()-start;rows.append(row);write(OUT/f'progress-{index}.json',dict(rows=rows))
        return rows
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=sum(list(pool.map(worker,range(2))),[])
    if any(sha(ROOT/name)!=digest for name,digest in sources.items()):raise ValueError('Validation evaluation sources changed')
    summary={}
    for method in plan['methods']:
        selected=[r for r in rows if r['method']==method]
        if len(selected)==24 and all(r['state']=='complete' for r in selected):summary[method]={key:float(np.mean([r['scores'][key] for r in selected])) for key in selected[0]['scores']}
    write(R/'development-forward-comparison-results.json',dict(plan_sha256=sha(R/'development-forward-comparison-plan.json'),rows=rows,summary=summary,failures=sum(r['state']!='complete' for r in rows)))

if __name__=='__main__':main()
