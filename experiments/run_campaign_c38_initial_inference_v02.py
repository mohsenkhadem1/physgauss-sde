"""V02 explicit continuation of initial PUBLIC-prefix inference after exit race.

Source/preparation only until its exact runtime protocol is reviewed. Candidate
E/support labels describe hypothetical models, never observed-object labels.
No private outcome/latent archive, evaluator, noise generator or truth API exists
here. All64 object predictions (including numerical failures) seal before any
separate evaluation. This trusted file/API boundary is not an OS sandbox.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, platform, re, resource
import shutil, signal, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
from experiments import run_campaign_c38_controlled_observations as codec
from experiments import campaign_c38_controlled_observations as obs
from experiments import prepare_campaign_c38_point_table as table

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'experiments/run_campaign_c38_initial_inference_v02.py'
WORKER = 'experiments/campaign_c38_public_prefix_inference_v02.py'
MODELS = ('nonlinear_table', 'linear_rest_tangent')
RESOURCES = dict(controller_cores=[32,33], worker_slots=[[i,i+1] for i in range(6,22,2)],
                 threads=2, worker_wall_s=300, global_wall_s=3600, cleanup_s=3,
                 worker_RSS_bytes=2*1024**3, controller_RSS_bytes=2*1024**3,
                 disk_free_reserve_bytes=30*1024**3, projected_output_bytes=1024**3,
                 max_workers=8, no_CUDA=True)
MANIFEST_KEYS = {'schema','completed','information_role','records','count','physical_configurations',
                 'noise_repeats_per_configuration','independent_population_objects','design_sha256',
                 'is_RGBD_perception','is_temporal_forecast'}
LINEAR_KEYS = {'case_ids','cells','support','young_Pa','Fz_N','material_probes_m','probe_positions_m'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''): h.update(b)
    return h.hexdigest()


def local(root,name):
    root=Path(root).resolve(); p=Path(name)
    if not isinstance(name,str) or not name or p.is_absolute() or '..' in p.parts or p.as_posix()!=name:
        raise ValueError('Canonical repository-relative path required')
    q=(root/p).resolve()
    if not q.is_relative_to(root): raise ValueError('Path escapes repository')
    return q


def record(root,path):
    root=Path(root).resolve(); p=Path(path).resolve()
    if not p.is_relative_to(root): raise ValueError('Repository-local record required')
    return dict(path=p.relative_to(root).as_posix(),sha256=sha(p))


def checked(root,r):
    if not isinstance(r,dict) or set(r)!= {'path','sha256'} or not re.fullmatch('[0-9a-f]{64}',str(r['sha256'])):
        raise ValueError('Exact hash-bound file record required')
    p=local(root,r['path'])
    if sha(p)!=r['sha256']: raise ValueError('Changed input: '+r['path'])
    return p


def read_json(root,r): return json.loads(checked(root,r).read_text())
def write_new(path,v):
    with Path(path).open('x') as f: json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def atomic(path,v):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');temp.replace(path)


def source_names():
    return sorted(set(table.source_names()) | {SOURCE,'experiments/run_campaign_c38_initial_inference.py',WORKER,codec.SOURCE,
        'experiments/campaign_c38_controlled_observations.py',
        'experiments/campaign_c38_candidate_point_forward.py',
        'experiments/campaign_c38_linear_candidate_forward.py',
        'experiments/campaign_c38_polynomial_likelihood.py',
        'experiments/campaign_c38_polynomial_likelihood_v02.py',
        'experiments/campaign_c38_polynomial_cache.py'})


def runtime():
    return dict(python=platform.python_version(), **{n:importlib.metadata.version(n) for n in ('numpy','scipy','sympy','mpmath')})


def public_plan(root,r):
    """Strict basename resolution; no prefix member extraction during prepare."""
    p=checked(root,r); m=json.loads(p.read_text())
    if set(m)!=MANIFEST_KEYS or m.get('schema')!='C38-public-controlled-ideal-prefixes-v01' or m.get('completed') is not True or m.get('information_role')!='PUBLIC_ACQUIRED_IDEAL_STATIC_POINTS_B' or m.get('count')!=64 or m.get('physical_configurations')!=4 or m.get('noise_repeats_per_configuration')!=16 or m.get('independent_population_objects') is not False or m.get('design_sha256')!=obs.DESIGN_SHA256 or m.get('is_RGBD_perception') is not False or m.get('is_temporal_forecast') is not False:
        raise ValueError('Exact complete PUBLIC64 manifest required')
    rows=[]
    for row in m['records']:
        if set(row)!= {'object_id','prefix'} or not re.fullmatch('[0-9a-f]{32}',str(row['object_id'])) or set(row['prefix'])!= {'path','sha256'}:
            raise ValueError('Public identity/prefix record only')
        name=row['prefix']['path']
        if not isinstance(name,str) or name!=row['object_id']+'.npz' or Path(name).name!=name:
            raise ValueError('Public prefix must be its identity basename')
        target=(p.parent/name).resolve()
        if target.parent!=p.parent.resolve(): raise ValueError('Public basename escapes manifest parent')
        rr=dict(path=target.relative_to(Path(root).resolve()).as_posix(),sha256=row['prefix']['sha256'])
        checked(root,rr); rows.append(dict(object_id=row['object_id'],prefix=rr))
    ids=[r['object_id'] for r in rows]
    if len(ids)!=64 or len(set(ids))!=64 or ids!=sorted(ids): raise ValueError('All64 unique sorted opaque public IDs required')
    return rows


def linear_plan(root,summary_record,terminal_record):
    """Read only sealed LINEAR metadata; never consume nonlinear result files."""
    s=read_json(root,summary_record); t=read_json(root,terminal_record)
    if s.get('completed') is not True or s.get('predictions_completed')!=28 or s.get('all8_basis_original_residual_checks_passed') is not True or s.get('all8_independent_direction_checks_passed') is not True or s.get('nonlinear_target_reads') is not False:
        raise ValueError('Completed independent linear28 summary required')
    if t.get('completed') is not True or t.get('all4_basis_and28_prediction_capture_hashes_match') is not True or t.get('all39_source_pins_unchanged') is not True or not t.get('owned_PID_absence') or not all(v is True for v in t['owned_PID_absence'].values()):
        raise ValueError('Linear terminal/source/capture proof required')
    for name,digest in t['source_sha256'].items():
        if sha(local(root,name))!=digest: raise ValueError('Frozen linear source changed')
    seal_record=s['predictions_seal']; seal=read_json(root,seal_record)
    if seal.get('completed') is not True or seal.get('all28_predictions_sealed_before_nonlinear_target_access') is not True or seal.get('nonlinear_target_access') is not False:
        raise ValueError('Independent pre-truth linear prediction seal required')
    from experiments.run_campaign_c38_static_prequalification import cases
    expected={c['id']:c for c in cases()}; rows=[]; seen=set()
    allowed={(r['path'],r['sha256']) for r in t['captures']}
    for group in seal['rows']:
        case=group['case']; cid=case['id']
        if cid not in expected or any(case[k]!=expected[cid][k] for k in ('young_Pa','support','Fz_N')): raise ValueError('Linear physical case mismatch')
        for row in group['prediction_records']:
            cells=row['cells']; key=(cid,tuple(cells))
            if key in seen or tuple(cells) not in ((32,16,16),(40,20,20)) or row['case']!=case or row.get('finite') is not True: raise ValueError('Exact unique finite linear meshes required')
            cap=row['capture']
            if (cap['path'],cap['sha256']) not in allowed: raise ValueError('Linear capture absent from terminal proof')
            checked(root,cap); seen.add(key);rows.append(dict(case=case,cells=cells,capture=cap))
    if seen!={(c,n) for c in expected for n in ((32,16,16),(40,20,20))}: raise ValueError('All28 sealed linear hypotheses required')
    return dict(summary=summary_record,terminal=terminal_record,predictions_seal=seal_record,rows=sorted(rows,key=lambda r:(r['case']['id'],r['cells'][0])))


def collect_reuse(root,predecessor_record,failed_record):
    """Every finished arm, including numerical negatives, reused by exact hash.

    No extraction/inference/selection by posterior values. Interrupted arms are
    explicitly allowed new computations, never reported as completed old work.
    """
    from experiments import run_campaign_c38_initial_inference as old
    root=Path(root).resolve()
    before=old.validate(predecessor_record['path'],predecessor_record['sha256'],root=root)
    failed=read_json(root,failed_record)
    if failed.get('terminal') is not True or failed.get('completed') is not False or failed.get('protocol')!=predecessor_record or failed.get('primary_failure',{}).get('message')!='Owned RSS measurement unavailable' or failed.get('controller_and_all12_worker_PIDs_absent') is not True:
        raise ValueError('Preserved exact v01 controller-exit-race failure required')
    execution=read_json(root,failed['execution'])
    if execution.get('protocol')!=predecessor_record or execution.get('completed') is not False or execution.get('all_owned_workers_reaped') is not True:
        raise ValueError('Frozen failed/reaped predecessor execution required')
    declared={r['object_id']:r for r in before['objects']}
    lineage={r['object_id']:r for r in failed['result_lineage']}
    if len(lineage)!=len(execution['records']) or set(lineage)!={r['object_id'] for r in execution['records']}:
        raise ValueError('Every launched predecessor attempt retained')
    byobject={};saved_count=0;complete_count=0
    for row in execution['records']:
        token=row['object_id'];proof=lineage[token]
        if token not in declared: raise ValueError('Undeclared prior identity')
        expected=local(root,before['output'])/'objects'/token/'result.json'
        if not expected.exists():
            if proof.get('result') is not None: raise ValueError('Lost previously saved result')
            continue
        result_record=record(root,expected)
        if result_record!=proof.get('result'): raise ValueError('Preserved original result changed')
        result=read_json(root,result_record)
        if result.get('object_id')!=token or result.get('pid')!=row['pid'] or result.get('protocol_sha256')!=predecessor_record['sha256'] or result.get('private_truth_access') is not False:
            raise ValueError('Old result/source/input/identity mismatch')
        saved={}
        for name,model in result.get('models',{}).items():
            if name not in MODELS or model.get('execution_completed') is not True or set(model['files'])!={'posterior-arrays.npz','posterior-structure.json','numerics.json'}:
                raise ValueError('Only complete persisted model executions reusable')
            for filename,r in model['files'].items():
                if local(root,r['path'])!=expected.parent/name/filename: raise ValueError('Exact prior model file binding required')
                checked(root,r)
            saved[name]=dict(source_result=result_record,model_name=name,model=model,predecessor_protocol_sha256=predecessor_record['sha256']);saved_count+=1
        if saved: byobject[token]=saved
        if set(saved)==set(MODELS): complete_count+=1
    return dict(predecessor_protocol=predecessor_record,failed_terminal=failed_record,
                predecessor_execution=failed['execution'],model_inputs=before['model_inputs'],objects=before['objects'],
                models_by_object=byobject,fully_reused_objects=complete_count,
                reused_model_executions=saved_count,new_model_executions_required=128-saved_count,
                previously_launched_objects=len(execution['records']),
                interrupted_objects_requiring_continuation=len(execution['records'])-complete_count,
                original_unlaunched_objects=len(execution['queued_object_ids']),
                no_observations_or_identities_regenerated=True,
                numerical_failures_not_filtered=True)


def reused_model(root,descriptor):
    original=read_json(root,descriptor['source_result'])
    if original.get('protocol_sha256')!=descriptor['predecessor_protocol_sha256'] or original['models'][descriptor['model_name']]!=descriptor['model']:
        raise ValueError('Exact reused model provenance required')
    model=descriptor['model']
    for r in model['files'].values(): checked(root,r)
    return dict(model,reused=True,reused_from_result=descriptor['source_result'])


def prepare(protocol,output,*,candidate_archive,public_manifest,linear_summary,linear_terminal,predecessor_protocol,predecessor_protocol_sha256,failed_terminal,root=ROOT):
    root=Path(root).resolve();dest=local(root,protocol);out=local(root,output)
    if dest.exists() or out.exists(): raise FileExistsError('Fresh exclusive protocol/output; no resume or retry')
    design=obs.load_frozen_design(root)
    candidate=record(root,local(root,candidate_archive));manifest=record(root,local(root,public_manifest))
    rows=public_plan(root,manifest)
    linear=linear_plan(root,record(root,local(root,linear_summary)),record(root,local(root,linear_terminal)))
    taskset=shutil.which('taskset')
    if not taskset: raise RuntimeError('taskset required')
    reuse=collect_reuse(root,dict(path=predecessor_protocol,sha256=predecessor_protocol_sha256),record(root,local(root,failed_terminal)))
    if reuse['model_inputs']!=dict(candidate_archive=candidate,public_manifest=manifest,linear=linear) or reuse['objects']!=rows: raise ValueError('Continuation must use exact SAME observations and hypotheses')
    p=dict(schema='C38-initial-public-inference-v02',status='FROZEN_BEFORE_PUBLIC_PREFIX_INFERENCE',
           output=output,source_sha256={n:sha(root/n) for n in source_names()},runtime=runtime(),
           taskset=dict(path=taskset,sha256=sha(taskset)),design=dict(path=obs.DESIGN_PATH,sha256=obs.DESIGN_SHA256),
           model_inputs=dict(candidate_archive=candidate,public_manifest=manifest,linear=linear),objects=rows,
           resources=RESOURCES,models=list(MODELS),candidate_mesh=[40,20,20],retries=0,
           policy='No automatic retry/drop. Explicit v01 controller-bug continuation reuses every finished arm; interrupted/incomplete arms may be recomputed ONCE under this new protocol. Returned numerical failures remain retained.',
           prediction_seal_policy='All64x2 executions and outputs sealed before separate private evaluation; no evaluator/private truth API in this program.',
           private_target_or_latent_access=False,raw_local=True,continuation=reuse,technical_interruption_recomputation_explicitly_authorized=True)
    dest.parent.mkdir(parents=True,exist_ok=True);write_new(dest,p);return sha(dest)


def validate(protocol,digest,*,root=ROOT,full_inputs=True):
    root=Path(root).resolve();p=read_json(root,dict(path=protocol,sha256=digest))
    if p.get('schema')!='C38-initial-public-inference-v02' or p.get('status')!='FROZEN_BEFORE_PUBLIC_PREFIX_INFERENCE' or p.get('resources')!=RESOURCES or p.get('models')!=list(MODELS) or p.get('candidate_mesh')!=[40,20,20] or p.get('retries')!=0 or p.get('private_target_or_latent_access') is not False:
        raise ValueError('Exact no-truth64x2 inference/resource protocol required')
    local(root,p['output'])
    if p.get('runtime')!=runtime() or p.get('source_sha256')!={n:sha(root/n) for n in source_names()}: raise ValueError('Full source/runtime pin mismatch')
    if p.get('design')!={'path':obs.DESIGN_PATH,'sha256':obs.DESIGN_SHA256}: raise ValueError('Frozen design identity required')
    obs.load_frozen_design(root)
    if set(p['model_inputs'])!= {'candidate_archive','public_manifest','linear'}: raise ValueError('Public-only model input plan required')
    if p['taskset']['path']!=shutil.which('taskset') or sha(p['taskset']['path'])!=p['taskset']['sha256']: raise ValueError('Taskset pin mismatch')
    if full_inputs:
        inputs=p['model_inputs'];checked(root,inputs['candidate_archive'])
        if p['objects']!=public_plan(root,inputs['public_manifest']): raise ValueError('Public bundle changed')
        old=inputs['linear']
        if old!=linear_plan(root,old['summary'],old['terminal']): raise ValueError('Linear closure changed')
        reused=p['continuation']
        if reused!=collect_reuse(root,reused['predecessor_protocol'],reused['failed_terminal']): raise ValueError('Exact completed reuse lineage changed')
    return p


def build_linear_library(root,p,destination):
    """All28 capture hashes before ANY NPZ read; only two q63 members extracted."""
    rows=p['model_inputs']['linear']['rows']
    for row in rows: checked(root,row['capture'])
    from experiments.campaign_c38_table_validation import POINTS
    from experiments.campaign_c38_linear_candidate_forward import LinearCandidatePointForward
    extracted=[]
    for row in rows:
        with np.load(local(root,row['capture']['path']),allow_pickle=False) as z:
            points=z['material_probes_m'].copy();q=z['probe_positions_m'].copy()
        if not np.array_equal(points,POINTS) or q.shape!=(63,3) or not np.isfinite(q).all(): raise ValueError('Exact finite linear q63 scope required')
        extracted.append(dict(case=row['case'],cells=row['cells'],material_probes_m=points,probe_positions_m=q))
    LinearCandidatePointForward(extracted,cells=(40,20,20))
    data=dict(case_ids=np.array([r['case']['id'] for r in rows]),cells=np.array([r['cells'] for r in rows],dtype=np.int64),support=np.array([r['case']['support'] for r in rows]),young_Pa=np.array([r['case']['young_Pa'] for r in rows]),Fz_N=np.array([r['case']['Fz_N'] for r in rows]),material_probes_m=POINTS,probe_positions_m=np.array([r['probe_positions_m'] for r in extracted]))
    codec.exclusive_npz(destination,data);return record(root,destination)


def load_linear_library(root,r):
    from experiments.campaign_c38_linear_candidate_forward import LinearCandidatePointForward
    with np.load(checked(root,r),allow_pickle=False) as z:
        if set(z.files)!=LINEAR_KEYS: raise ValueError('Exact derived linear point library required')
        a={k:z[k].copy() for k in LINEAR_KEYS}
    if a['case_ids'].shape!=(28,) or a['cells'].shape!=(28,3) or a['probe_positions_m'].shape!=(28,63,3): raise ValueError('All28 linear hypothesis rows required')
    rows=[dict(case=dict(id=str(a['case_ids'][i]),young_Pa=float(a['young_Pa'][i]),support=str(a['support'][i]),Fz_N=float(a['Fz_N'][i])),cells=a['cells'][i].tolist(),material_probes_m=a['material_probes_m'],probe_positions_m=a['probe_positions_m'][i]) for i in range(28)]
    return LinearCandidatePointForward(rows,cells=(40,20,20))


def compact_result(r):
    """Numerical diagnostics only; no errors against truth/interval coverage."""
    keys=('completed','numerical_controls_passed','object_id','failures','gates','comparisons','cache_stats','physical_calibration_established','is_RGBD_perception','is_temporal_forecast')
    out={k:r[k] for k in keys if k in r};out['methods']={}
    for name,m in r.get('methods',{}).items():
        out['methods'][name]={k:m[k] for k in ('completed','stage','error','log_normalizer','normalizer_error_estimate','category_probabilities','material_equal_tail_Pa','canonical_mean_coordinate_variance_m2') if k in m}
    checks=r.get('GL64_endpoint_CDF_checks',[])
    out['endpoint_CDF_checks']=dict(count=len(checks),passed=sum(c.get('passed') is True for c in checks))
    return jsonable(out)


def jsonable(v):
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,dict): return {k:jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [jsonable(x) for x in v]
    return v


def save_full_result(directory,r):
    arrays={}
    def pack(value,path='root'):
        if isinstance(value,np.ndarray):
            key='array_'+str(len(arrays));arrays[key]=value
            return dict(array=key,shape=list(value.shape),dtype=str(value.dtype))
        if isinstance(value,dict): return {k:pack(v,path+'/'+k) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [pack(v,path+'/'+str(i)) for i,v in enumerate(value)]
        if isinstance(value,np.generic): return value.item()
        return value
    tree=pack(r);codec.exclusive_npz(directory/'posterior-arrays.npz',arrays)
    write_new(directory/'posterior-structure.json',tree)
    write_new(directory/'numerics.json',compact_result(r))


def worker(protocol,digest,object_id,slot,library_record,barrier_record,*,root=ROOT):
    root=Path(root).resolve();p=validate(protocol,digest,root=root,full_inputs=False)
    if type(slot) is not int or not 0<=slot<8: raise ValueError('Declared CPU slot required')
    guard(root,RESOURCES['worker_slots'][slot]);found=[r for r in p['objects'] if r['object_id']==object_id]
    if len(found)!=1: raise ValueError('Exactly declared public identity required')
    out=local(root,p['output']);barrier=read_json(root,barrier_record)
    if barrier_record['path']!=(out/'model-input-barrier.json').relative_to(root).as_posix() or barrier.get('completed') is not True or barrier.get('protocol_sha256')!=digest or barrier.get('linear_library')!=library_record or barrier.get('candidate')!=p['model_inputs']['candidate_archive'] or barrier.get('public_manifest')!=p['model_inputs']['public_manifest'] or barrier.get('original28_linear_captures_hashed_before_extraction') is not True or barrier.get('all64_public_prefix_hashes_verified_before_inference') is not True or barrier.get('private_truth_access') is not False: raise ValueError('Hash barrier/library binding required')
    checked(root,p['model_inputs']['candidate_archive']);checked(root,p['model_inputs']['public_manifest']);checked(root,found[0]['prefix']);checked(root,library_record)
    from experiments.campaign_c38_candidate_point_forward import CandidatePointForward,ARCHIVE_KEYS
    from experiments.campaign_c38_public_prefix_inference_v02 import infer_public_prefix
    design=obs.load_frozen_design(root);public=codec.load_public_prefix(root,found[0]['prefix'],design=design)
    if public['object_id']!=object_id: raise ValueError('Public codec identity mismatch')
    with np.load(checked(root,p['model_inputs']['candidate_archive']),allow_pickle=False) as z:
        if set(z.files)!=ARCHIVE_KEYS: raise ValueError('Candidate archive only; no direct validation/truth')
        candidate=CandidatePointForward({k:z[k].copy() for k in ARCHIVE_KEYS},cells=(40,20,20))
    linear=load_linear_library(root,library_record)
    directory=out/'objects'/object_id;directory.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();state=dict(completed=False,pid=os.getpid(),object_id=object_id,slot=slot,protocol_sha256=digest,models={},private_truth_access=False,raw_local=True)
    atomic(directory/'result.json',state)
    for name,model in zip(MODELS,(candidate,linear)):
        guard(root,RESOURCES['worker_slots'][slot]);d=directory/name;d.mkdir()
        saved=p['continuation']['models_by_object'].get(object_id,{}).get(name)
        if saved is not None:
            state['models'][name]=reused_model(root,saved)
            state['seconds']=time.monotonic()-start;atomic(directory/'result.json',state);continue
        r=infer_public_prefix(model,public,design=design)
        if not isinstance(r,dict) or r.get('object_id',object_id)!=object_id: raise ValueError('Worker result identity/schema mismatch')
        save_full_result(d,r)
        files={n:record(root,d/n) for n in ('posterior-arrays.npz','posterior-structure.json','numerics.json')}
        state['models'][name]=dict(execution_completed=True,completed=r.get('completed') is True,numerical_controls_passed=r.get('numerical_controls_passed') is True,files=files)
        state['seconds']=time.monotonic()-start;atomic(directory/'result.json',state)
    state.update(completed=True,all_models_numerical_passed=all(v['completed'] and v['numerical_controls_passed'] for v in state['models'].values()),seconds=time.monotonic()-start)
    atomic(directory/'result.json',state);return state


def current_rss(pid):
    """None means only an observed exit/Z/X state, never a live RSS guess."""
    try: lines=Path('/proc',str(pid),'status').read_text().splitlines()
    except FileNotFoundError: return None
    state=next((line.split()[1] for line in lines if line.startswith('State:')),None)
    if state in ('Z','X','x'): return None
    for line in lines:
        if line.startswith('VmRSS:'): return int(line.split()[1])*1024
    raise RuntimeError('Truly live owned RSS measurement unavailable')


def guard(root,cores):
    if set(os.sched_getaffinity(0))!=set(cores) or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')): raise ValueError('Exact CPU affinity/thread2/noCUDA guard')
    rss=current_rss(os.getpid())
    if rss is None or rss>RESOURCES['controller_RSS_bytes']: raise MemoryError('Live self RSS required under2GiB cap')
    if shutil.disk_usage(root).free-RESOURCES['projected_output_bytes']<RESOURCES['disk_free_reserve_bytes']: raise OSError('30GiB+projected1GiB disk reserve')


def cleanup_owned(proc,grace=3):
    """Only our new-session PGID; always wait/reap before returning."""
    # The session leader can exit before descendants; its saved PGID remains
    # ours. Never infer ownership from an unrelated PID or scan/kill by name.
    try: os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError: pass
    try: code=proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        code=proc.wait()
    # Also terminate an ignoring descendant after the leader exited on TERM.
    try: os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError: pass
    return code


def verify_worker_output(root,p,row):
    r=read_json(root,row['result'])
    if r.get('completed') is not True or r.get('pid')!=row['pid'] or r.get('object_id')!=row['object_id'] or r.get('protocol_sha256')!=row['protocol_sha256'] or set(r.get('models',{}))!=set(MODELS) or r.get('private_truth_access') is not False: raise ValueError('Complete exact64x2 output identity required')
    for model in r['models'].values():
        if model.get('execution_completed') is not True or set(model['files'])!={'posterior-arrays.npz','posterior-structure.json','numerics.json'}: raise ValueError('All model outputs retained')
        for f in model['files'].values(): checked(root,f)
    return r


def run(protocol,digest,*,root=ROOT):
    root=Path(root).resolve();p=validate(protocol,digest,root=root);guard(root,RESOURCES['controller_cores']);out=local(root,p['output'])
    if out.exists(): raise FileExistsError('Fresh output only; no retry/resume')
    out.mkdir(parents=True,exist_ok=False);(out/'objects').mkdir();(out/'logs').mkdir()
    start=time.monotonic();state=dict(completed=False,protocol=dict(path=protocol,sha256=digest),pid=os.getpid(),records=[],queued_object_ids=[r['object_id'] for r in p['objects']],live=[],technical_failure=False,retries=0,raw_local=True,private_truth_access=False)
    live={};stopped=False
    def save():
        state.update(seconds=time.monotonic()-start,live=[dict(pid=x['process'].pid,object_id=x['row']['object_id'],slot=s) for s,x in live.items()]);atomic(out/'execution.json',state)
    save()
    try:
        # validate() hashed ALL original28 captures, candidate and PUBLIC64 files
        # before extraction. Repeat linear hash barrier immediately at extraction.
        library=build_linear_library(root,p,out/'linear-query-library.npz')
        write_new(out/'model-input-barrier.json',dict(completed=True,protocol_sha256=digest,linear_library=library,candidate=p['model_inputs']['candidate_archive'],public_manifest=p['model_inputs']['public_manifest'],original28_linear_captures_hashed_before_extraction=True,all64_public_prefix_hashes_verified_before_inference=True,private_truth_access=False))
        barrier_record=record(root,out/'model-input-barrier.json');state['model_input_barrier']=barrier_record;save()
        for token,models in p['continuation']['models_by_object'].items():
            if set(models)!=set(MODELS): continue
            directory=out/'objects'/token;directory.mkdir()
            saved={name:reused_model(root,r) for name,r in models.items()}
            result=dict(completed=True,pid=None,object_id=token,protocol_sha256=digest,models=saved,private_truth_access=False,all_models_numerical_passed=all(v['completed'] and v['numerical_controls_passed'] for v in saved.values()),reused_complete_object=True)
            write_new(directory/'result.json',result)
            row=dict(object_id=token,pid=None,kind='REUSED_COMPLETE_OBJECT_NO_PROCESS',protocol_sha256=digest,completed=True,all_models_numerical_passed=result['all_models_numerical_passed'],result=record(root,directory/'result.json'))
            state['records'].append(row);state['queued_object_ids'].remove(token)
        save()
        while state['queued_object_ids'] or live:
            guard(root,RESOURCES['controller_cores'])
            if time.monotonic()-start>RESOURCES['global_wall_s']: raise TimeoutError('Global3600s cap; queued IDs retained')
            if not stopped:
                for slot in range(8):
                    if slot in live or not state['queued_object_ids']: continue
                    token=state['queued_object_ids'].pop(0);log=(out/'logs'/(token+'.log')).open('xb');cores=RESOURCES['worker_slots'][slot]
                    cmd=[p['taskset']['path'],'-c',','.join(map(str,cores)),sys.executable,'-m','experiments.run_campaign_c38_initial_inference_v02','worker','--protocol',protocol,'--protocol-sha256',digest,'--object-id',token,'--slot',str(slot),'--linear-library',library['path'],'--linear-library-sha256',library['sha256'],'--input-barrier-sha256',barrier_record['sha256']]
                    try: proc=subprocess.Popen(cmd,cwd=root,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    except BaseException: log.close();state['queued_object_ids'].insert(0,token);raise
                    row=dict(object_id=token,pid=proc.pid,PGID=proc.pid,slot=slot,command=cmd,protocol_sha256=digest,start_monotonic=time.monotonic())
                    live[slot]=dict(process=proc,row=row,log=log);save()
            for slot,x in list(live.items()):
                proc=x['process'];row=x['row'];code=proc.poll();reason=None
                if code is None:
                    if time.monotonic()-row['start_monotonic']>RESOURCES['worker_wall_s']: reason='worker300s wall cap'
                    else:
                        rss=current_rss(proc.pid)
                        if rss is None:
                            # Exit may occur between poll and /proc read. Reap
                            # the SAME owned child; no zero-RSS live substitute.
                            try: code=proc.wait(timeout=.05)
                            except subprocess.TimeoutExpired: code=None
                        elif rss>RESOURCES['worker_RSS_bytes']: reason='worker2GiB RSS cap'
                    if reason is not None: code=cleanup_owned(proc,RESOURCES['cleanup_s'])
                if code is None: continue
                proc.wait();x['log'].close();row.update(exit_code=code,reaped=True,seconds=time.monotonic()-row['start_monotonic']);del live[slot]
                try:
                    if reason is not None or code!=0: raise RuntimeError(reason or 'Worker technical exit '+str(code))
                    row['result']=record(root,out/'objects'/row['object_id']/'result.json');result=verify_worker_output(root,p,row)
                    row.update(completed=True,all_models_numerical_passed=result['all_models_numerical_passed'])
                except Exception as error: row.update(completed=False,failure=repr(error));stopped=True;state['technical_failure']=True
                state['records'].append(row);save()
            if stopped and not live: break
            time.sleep(.2)
        if not state['technical_failure'] and len(state['records'])==64 and not state['queued_object_ids']:
            for r in state['records']: verify_worker_output(root,p,r)
            validate(protocol,digest,root=root)
            seal=dict(schema='C38-initial-public-inference-seal-v02',completed=True,protocol_sha256=digest,objects=sorted([dict(object_id=r['object_id'],result=r['result'],all_models_numerical_passed=r['all_models_numerical_passed']) for r in state['records']],key=lambda r:r['object_id']),count=64,model_executions=128,all_numerical_controls_passed=all(r['all_models_numerical_passed'] for r in state['records']),all_predictions_sealed_before_private_evaluation=True,private_truth_access=False,raw_local=True)
            write_new(out/'prediction-seal.json',seal);state.update(completed=True,prediction_seal=record(root,out/'prediction-seal.json'))
    except BaseException as error:
        state.update(technical_failure=True,failure=dict(type=type(error).__name__,message=str(error)),traceback=traceback.format_exc());save()
        raise
    finally:
        state['live_before_owned_cleanup']=[x['process'].pid for x in live.values()];save()
        for slot,x in list(live.items()):
            code=cleanup_owned(x['process'],RESOURCES['cleanup_s']);x['log'].close();x['row'].update(completed=False,exit_code=code,reaped=True,failure='Owned controller cleanup; no retry');state['records'].append(x['row']);del live[slot]
        state['all_owned_workers_reaped']=not live;save()
    return state


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','run','worker'));a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a.add_argument('--candidate-archive');a.add_argument('--public-manifest');a.add_argument('--linear-summary');a.add_argument('--linear-terminal');a.add_argument('--object-id');a.add_argument('--slot',type=int);a.add_argument('--linear-library');a.add_argument('--linear-library-sha256');a.add_argument('--input-barrier-sha256');a.add_argument('--predecessor-protocol');a.add_argument('--predecessor-protocol-sha256');a.add_argument('--failed-terminal');v=a.parse_args()
    if v.mode=='prepare': print(prepare(v.protocol,v.output,candidate_archive=v.candidate_archive,public_manifest=v.public_manifest,linear_summary=v.linear_summary,linear_terminal=v.linear_terminal,predecessor_protocol=v.predecessor_protocol,predecessor_protocol_sha256=v.predecessor_protocol_sha256,failed_terminal=v.failed_terminal))
    elif v.mode=='worker':
        p=validate(v.protocol,v.protocol_sha256,full_inputs=False)
        worker(v.protocol,v.protocol_sha256,v.object_id,v.slot,dict(path=v.linear_library,sha256=v.linear_library_sha256),dict(path=p['output']+'/model-input-barrier.json',sha256=v.input_barrier_sha256))
    else:
        s=run(v.protocol,v.protocol_sha256);print(json.dumps(dict(completed=s['completed'],technical_failure=s['technical_failure'])));sys.exit(0 if s['completed'] else 1)
