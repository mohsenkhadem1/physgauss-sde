"""Sealed Stage-A ORACLE forward benchmark; no inverse-policy selection here."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,ast,concurrent.futures,hashlib,json,queue,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.compositional_neural_mechanics.evaluation_protocol import construct_reserved_records,paired_object_bootstrap
from scripts.compositional_neural_mechanics.forward_evaluation import solve_forward,direct_forward,score_prediction
from scripts.compositional_neural_mechanics.feature_pack import load_frozen_laws,feature_contract
from scripts.compositional_neural_mechanics.direct_baseline import load_checkpoint
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.dataset_runner import write,npz

REPORT=ROOT/'reports/compositional-neural-mechanics-v16'
OUT=ROOT/'runs/compositional-neural-mechanics-v16/oracle-reserved-forward'
FREEZE=REPORT/'oracle-method-freeze.json'
KINDS=('matched_analytic','neo_hookean','shared_laws','custom_direct','custom_direct_long')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def inventory():
    pending=[Path(__file__).resolve()];seen=set()
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module,*[node.module+'.'+x.name for x in node.names if x.name!='*']] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                if name.startswith(('scripts.','experiments.')):
                    candidate=ROOT/(name.replace('.','/')+'.py')
                    if candidate.exists() and candidate not in seen:pending.append(candidate)
    return {str(path.relative_to(ROOT)):sha(path) for path in sorted(seen)}

def freeze():
    if FREEZE.exists():raise ValueError('Preserve existing method freeze')
    training=json.loads((REPORT/'direct-training-plan.json').read_text())
    path=ROOT/'runs/compositional-neural-mechanics-v16/direct-baseline-campaign/direct-baseline.pt'
    _,saved=load_checkpoint(path)
    if saved['metadata']['steps']!=6000 or saved['metadata']['seed']!=926701:raise ValueError('Declared final baseline required')
    if saved['metadata']['split_manifest']!=training['split_manifest']:raise ValueError('Baseline training membership changed')
    checkpoints=dict(training['checkpoints']);checkpoints['direct']=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
    for key in ('bulk','interface'):
        if sha(ROOT/checkpoints[key]['path'])!=checkpoints[key]['sha256']:raise ValueError('Pretrained law changed')
        if saved['feature_contract'][key+'_law_fingerprint']['checkpoint_sha256']!=checkpoints[key]['sha256']:raise ValueError('Baseline law fingerprint differs')
    if saved['feature_contract']!=feature_contract(training['probes'],{key:checkpoints[key]['sha256'] for key in ('bulk','interface')}):raise ValueError('Baseline feature/probe contract differs')
    long_path=ROOT/'runs/compositional-neural-mechanics-v16/direct-capacity-campaign/direct-baseline.pt'
    _,long_saved=load_checkpoint(long_path)
    if long_saved['metadata']['steps']!=30000 or long_saved['metadata']['seed']!=926701 or long_saved['metadata']['split_manifest']!=training['split_manifest']:raise ValueError('Declared capacity-check baseline required')
    if long_saved['feature_contract']!=saved['feature_contract']:raise ValueError('Capacity baseline feature contract differs')
    checkpoints['direct_long']=dict(path=str(long_path.relative_to(ROOT)),sha256=sha(long_path))
    evidence={name:sha(REPORT/name) for name in ('nonlinear-range-qualification.json','nonlinear-range-fine-qualification.json','crossed-support-qualification.json','wide-interface-equilibrium-qualification.json','direct-training-plan.json','direct-capacity-plan.json')}
    if not json.loads((REPORT/'nonlinear-range-fine-qualification.json').read_text())['all_gates_pass']:raise ValueError('Fine development qualification required')
    if not json.loads((REPORT/'crossed-support-qualification.json').read_text())['all_mesh_gates_pass']:raise ValueError('Crossed attachment qualification required')
    if json.loads((REPORT/'wide-interface-equilibrium-qualification.json').read_text())['failed_solves']:raise ValueError('Learned law development solves incomplete')
    plan=dict(status='frozen',stage='oracle-forward-composition-v16',source_sha256=inventory(),checkpoints=checkpoints,
              qualification_evidence=evidence,seeds=dict(composition=926521,geometry=926523),prediction_n=4,reference_n=8,order=3,
              node_material=training['node_material'],material_query=training['material_query'],probes=training['probes'],
              methods=list(KINDS),devices=['cuda:0','cuda:1','cuda:2'],per_scene_seconds=1800,stage_seconds=10800,
              scope='Known geometry and oracle material/support descriptors. Six withheld composition objects; six draws on two novel geometries with familiar categorical compositions. Four unseen position/orientation commands each. No unknown-physics or active-policy claim.',
              statistics='Aggregate commands within each object. Report 6 composition objects separately from 6 draws conditional on two novel shapes; no geometry-population confidence claim.',
              reference='Fixed n8 analytic reference. Development mesh qualification is evidence, not a per-reserved-case convergence proof.',
              inverse_cohort='Future inverse/active evaluation must use a separate untouched cohort; these oracle outcomes cannot select inverse methods on the same objects.')
    write(FREEZE,plan);return plan

def guard():
    plan=json.loads(FREEZE.read_text())
    if plan['source_sha256']!=inventory():raise ValueError('Oracle source changed after freeze')
    for item in plan['checkpoints'].values():
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Oracle checkpoint changed after freeze')
    for path,digest in plan['qualification_evidence'].items():
        if sha(REPORT/path)!=digest:raise ValueError('Development evidence changed')
    return plan

def runtime_records():
    plan=guard();manifest=json.loads((OUT/'run-manifest.json').read_text());path=OUT/'reserved-records.json'
    if manifest['freeze_sha256']!=sha(FREEZE) or manifest['records_sha256']!=sha(path):raise ValueError('Reserved runtime commitment changed')
    data=json.loads(path.read_text());records=data['records']
    if data['freeze_sha256']!=sha(FREEZE) or len(records)!=12 or len({r['scene']['name'] for r in records})!=12:raise ValueError('Reserved scene census changed')
    if any(sum(r['role']==role for r in records)!=6 for role in ('reserved_composition','reserved_geometry')):raise ValueError('Reserved cohort census changed')
    if any(len(r['actions'])!=4 or len({a['name'] for a in r['actions']})!=4 for r in records):raise ValueError('Reserved command census changed')
    return plan,records

def validate_value(value,queries):
    if np.shape(value['displacement'])!=(len(queries),3) or np.shape(value['wrenches'])!=(2,6):raise ValueError('Physical prediction shape mismatch')
    if not all(np.isfinite(value[key]).all() for key in ('displacement','wrenches')):raise ValueError('Nonfinite reference or prediction')

def worker(index,device):
    torch.set_num_threads(1);plan,records=runtime_records();record=records[index]
    if device not in plan['devices']:raise ValueError('Undeclared device')
    bulk,interface=load_frozen_laws(*(ROOT/plan['checkpoints'][k]['path'] for k in ('bulk','interface')),
                                   {k:plan['checkpoints'][k]['sha256'] for k in ('bulk','interface')},device)
    direct_models={}
    for name in ('direct','direct_long'):
        direct,_=load_checkpoint(ROOT/plan['checkpoints'][name]['path'],map_location=device);direct_models[name]=direct.to(device)
    q=np.asarray(plan['material_query']);scene=record['scene'];folder=OUT/scene['name'];folder.mkdir(exist_ok=True)
    rows=[];started=time.monotonic()
    for action in record['actions']:
        reference=None;actiondir=folder/action['name'];actiondir.mkdir(exist_ok=True)
        for kind in ('reference',*KINDS):
            row=dict(scene_id=scene['name'],cohort=record['role'],action=action['name'],method=kind);begin=time.monotonic()
            try:
                if kind.startswith('custom_direct'):
                    value=direct_forward(scene,action['pose'],q,predictor=direct_models['direct_long' if kind=='custom_direct_long' else 'direct'],nodes=plan['node_material'],bulk_law=bulk,interface_law=interface,
                                         probes=plan['probes'],checkpoint_sha256={k:plan['checkpoints'][k]['sha256'] for k in ('bulk','interface')})
                else:
                    value=solve_forward(scene,action['pose'],q,kind='matched_analytic' if kind=='reference' else kind,
                                        n=plan['reference_n'] if kind=='reference' else plan['prediction_n'],order=plan['order'],device=device,
                                        bulk_law=bulk,interface_law=interface)
                validate_value(value,q)
                target=actiondir/(kind+'.npz');npz(target,material_query=q,displacement=value['displacement'],wrenches=value['wrenches'])
                row.update(state='complete',artifact=str(target.relative_to(ROOT)),sha256=sha(target),diagnostics=value['diagnostics'])
                if kind=='reference':reference=value
                elif reference is None:row.update(scoring_state='missing_reference')
                else:row['scores']=score_prediction(value,reference,GeometrySpec(**scene['geometry']).map(q))
            except Exception as error:row.update(state='failed',error=repr(error))
            row['seconds']=time.monotonic()-begin;rows.append(row);write(folder/'progress.json',dict(rows=rows))
    runtime_records()
    write(folder/'complete.json',dict(rows=rows,seconds=time.monotonic()-started))

def summarize(records):
    plan,committed=runtime_records()
    if records!=committed:raise ValueError('Aggregation records differ from commitment')
    allrows=[]
    for record in records:
        folder=OUT/record['scene']['name'];path=folder/'complete.json'
        if not path.exists():path=folder/'progress.json'
        existing=json.loads(path.read_text())['rows'] if path.exists() else []
        for action in record['actions']:
            for kind in ('reference',*KINDS):
                found=next((r for r in existing if r['action']==action['name'] and r['method']==kind),None)
                if found and found['state']=='complete':
                    if sha(ROOT/found['artifact'])!=found['sha256']:raise ValueError('Reserved prediction artifact changed')
                    with np.load(ROOT/found['artifact'],allow_pickle=False) as data:
                        validate_value(data,plan['material_query'])
                        if set(data.files)!={'material_query','displacement','wrenches'} or not np.array_equal(data['material_query'],np.asarray(plan['material_query'])):raise ValueError('Reserved query/settings mismatch')
                allrows.append(found or dict(scene_id=record['scene']['name'],cohort=record['role'],action=action['name'],method=kind,state='failed',error='No completed prediction before bounded stage ended'))
    # Recompute numerical scores from the committed arrays, not a mutable metric field.
    for row in allrows:
        if row['method']=='reference' or row['state']!='complete':continue
        ref=next(r for r in allrows if r['scene_id']==row['scene_id'] and r['action']==row['action'] and r['method']=='reference')
        if ref['state']!='complete':row.pop('scores',None);continue
        scene=next(r['scene'] for r in records if r['scene']['name']==row['scene_id'])
        with np.load(ROOT/row['artifact'],allow_pickle=False) as prediction,np.load(ROOT/ref['artifact'],allow_pickle=False) as reference:
            row['scores']=score_prediction(prediction,reference,GeometrySpec(**scene['geometry']).map(np.asarray(plan['material_query'])))
    summaries=[];paired=[]
    for cohort in ('reserved_composition','reserved_geometry'):
        names=[r['scene']['name'] for r in records if r['role']==cohort];means={}
        for kind in KINDS:
            selected=[r for r in allrows if r['cohort']==cohort and r['method']==kind]
            complete=all(r['state']=='complete' and 'scores' in r for r in selected)
            entry=dict(cohort=cohort,method=kind,objects=len(names),expected_commands=24,scored_commands=sum('scores' in r for r in selected),complete=complete)
            if complete:
                means[kind]={key:[float(np.mean([r['scores'][key] for r in selected if r['scene_id']==name])) for name in names] for key in selected[0]['scores']}
                entry['object_mean_scores']={key:float(np.mean(value)) for key,value in means[kind].items()}
            summaries.append(entry)
        if 'shared_laws' in means:
            for baseline in ('matched_analytic','neo_hookean','custom_direct','custom_direct_long'):
                if baseline not in means:continue
                for metric in ('displacement_coordinate_rmse_m','force_coordinate_rmse_N','torque_coordinate_rmse_Nm'):
                    paired.append(dict(cohort=cohort,baseline=baseline,metric=metric,**paired_object_bootstrap(means['shared_laws'][metric],means[baseline][metric])))
    report=dict(scope=guard()['scope'],freeze_sha256=sha(FREEZE),rows=allrows,summaries=summaries,paired_object_comparisons=paired,
                failed_predictions=sum(r['state']!='complete' for r in allrows),limitations=guard()['statistics'])
    write(REPORT/'oracle-reserved-forward-results.json',report)

def run():
    plan=guard()
    if OUT.exists():raise ValueError('Preserve previous reserved evaluation; no silent rerun or filtering')
    OUT.mkdir(parents=True);records=construct_reserved_records(FREEZE,ROOT)
    write(OUT/'reserved-records.json',dict(freeze_sha256=sha(FREEZE),records=records));deadline=time.time()+plan['stage_seconds'];pending=queue.Queue()
    write(OUT/'run-manifest.json',dict(freeze_sha256=sha(FREEZE),records_sha256=sha(OUT/'reserved-records.json'),deadline_unix=deadline))
    for i in range(len(records)):pending.put(i)
    def runner(device):
        while True:
            try:index=pending.get_nowait()
            except queue.Empty:return
            remaining=deadline-time.time()
            if remaining<=0:return
            with (OUT/f'worker-{index:02d}.log').open('w') as stream:
                child=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',str(index),'--device',device],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=min(remaining,plan['per_scene_seconds']))
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
            write(OUT/f'worker-{index:02d}-exit.json',dict(code=code))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(runner,plan['devices']))
    guard();summarize(records)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',type=int);parser.add_argument('--device');args=parser.parse_args()
    if args.freeze:freeze()
    elif args.worker is not None:worker(args.worker,args.device)
    elif args.execute:run()
    else:parser.error('Explicit freeze/execute/worker required')
