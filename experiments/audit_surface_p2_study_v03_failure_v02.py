"""Partial negative-run audit: provenance/physics only, no target scoring.

Never accesses matching_linear/NH80_stress response arrays or displacement
field arrays. Hashes raw files, reads only original prior coordinates and bound
physical-diagnostic scalar records. A controlled stop and nonzero child exits
are preserved; all16 owned slots must still be reaped and absent.
"""
import json,os,resource,signal,time,shutil
from pathlib import Path
import numpy as np
from experiments import audit_surface_p2_study_v03 as core

ROOT=core.ROOT;LOCAL=core.LOCAL;PROTOCOL=core.PROTOCOL
AUDIT=ROOT/'runs/operator-dev/surface-p2-study-v03-negative-audit-v02'
DESIGN=dict(schema='surface-p2-study-v03-negative-audit-v02',CPU=12,threads=1,GPU=False,
    wall_cap_s=900,RSS_cap_bytes=4*1024**3,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,objects=16,physical_records=320,
    no_response_or_displacement_arrays_accessed=True,no_posterior_or_quadrature_rerun=True,
    no_policy_or_prediction_scores=True,negative_stopped_run_preserved=True)


def closed_execution(base,digest):
    e=core.read(base/'execution.json')
    core.ensure(e['protocol_SHA256']==digest and e['all_worker_groups_absent'],'Bound absent worker groups')
    core.ensure(len(e['records'])==16 and {r['slot']for r in e['records']}==set(range(16)),'All16 retained worker slots')
    core.require_absent(e['pid'])
    for row in e['records']:
        core.ensure(row['reaped'],'Stopped child not reaped');core.require_absent(row['pid'])
    return e


def run(base=LOCAL,digest=PROTOCOL):
    began=time.monotonic();manifest_gate(base); core.ensure(core.sha(base/'protocol.json')==digest,'Original protocol hash')
    p=core.read(base/'protocol.json');e=closed_execution(base,digest)
    for name,h in p['source_SHA256'].items():core.ensure(core.sha(ROOT/name)==h,'Original source changed '+name)
    physical=[];prefix=[];bindings={};prefix_records=[]
    for i in range(16):
        folder=base/f'object-{i:02d}-LOCAL';meta=core.read(folder/'truth-checkpoint-LOCAL.json');expected=dict(unit=np.random.default_rng(p['design']['cohort_seed']).uniform(size=(16,2))[i])
        path=folder/'truth-LOCAL.npz';core.ensure(core.sha(path)==meta['raw_SHA256'],'Original truth raw hash')
        # Lazy NPZ access: no response, noise, target, or displacement arrays.
        with np.load(path,allow_pickle=False)as z:
            core.ensure(np.array_equal(z['unit'],expected['unit']),'Original prior draw was changed')
            theta=np.exp(np.log([2000.,5.])+expected['unit']*np.log([4.,6.]))
            core.same(theta.tolist(),z['theta'].tolist(),'Original E/K draw')
        bindings[str(path.relative_to(base))]=core.sha(path)
        core.ensure(meta['labels']==list(p['force_SHA256']),'All10 original force labels')
        context=dict(protocol_SHA256=digest,index=i,unit_SHA256=core.array_sha(expected['unit']),force_SHA256=p['force_SHA256'])
        core.commit(folder/'linear-fields-LOCAL.npz',context)
        bindings[str((folder/'linear-fields-LOCAL.npz').relative_to(base))]=core.sha(folder/'linear-fields-LOCAL.npz')
        local_valid=[]
        for label in meta['labels']:
            path=folder/f'truth-load-{label}-LOCAL.npz';core.commit(path,dict(context,label=label))
            bindings[str(path.relative_to(base))]=core.sha(path)
            with np.load(path,allow_pickle=False)as z:
                core.ensure(np.array_equal(z['unit'],expected['unit'])and str(z['force_SHA256'])==p['force_SHA256'][label],'Force/material field context')
                recorded=json.loads(str(z['diagnostics_json']))
            for law in core.LAWS:
                d=recorded[law];core.ensure(d==meta['physics'][law][label],'Physical record bound to raw field')
                valid=core.physical_valid(d,label==p['design']['prefix']);local_valid.append(valid)
                physical.append(dict(index=i,law=law,label=label,valid=valid,J=d['minimum_J'],strain=d['maximum_absolute_symmetric_strain_eigenvalue']))
                if label==p['design']['prefix']:prefix.append(dict(index=i,law=law,valid=valid,RMS=d['nonrigid_RMS_m'],fraction=d['nonrigid_fraction']))
        core.same(all(local_valid),meta['all_physical_valid'],'Object physical-validity aggregate')
        for law in core.LAWS:
            path=folder/f'{law}-prefix-LOCAL.json'
            if path.exists():
                r=core.read(path);core.same(r['qualified'],r['diagnostics']['qualified'],'Prefix qualified binding')
                prefix_records.append(dict(index=i,law=law,qualified=r['qualified'],record_sha256=core.sha(path),unresolved_levels=sum(v['status']=='unresolved'for v in r['diagnostics']['levels'])))
    core.ensure(len(physical)==320 and len(prefix)==32,'All320 physics/32law signals retained')
    attempts=attempt_counts(base,p)
    result=dict(schema=DESIGN['schema'],completed=True,audit_passed=True,scope='Original failed/stopped study: physical/raw provenance only; no held-target reads or policy comparison',protocol_SHA256=digest,execution_SHA256=core.sha(base/'execution.json'),all17_owned_PID_and_groups_absent=True,worker_slots_retained=16,nonzero_worker_exits_retained=sum(r['exit_code']!=0 for r in e['records']),original_execution_completed=e['completed'],original_execution_status=e['status'],objects_retained=16,original_prior_draws_verified=16,physical_records_verified=320,physical_records_valid=sum(d['valid']for d in physical),prefix_objects_verified=16,prefix_law_signals_verified=32,prefix_law_signals_valid=sum(d['valid']for d in prefix),minimum_prefix_nonrigid_RMS_m=min(d['RMS']for d in prefix),minimum_prefix_nonrigid_fraction=min(d['fraction']for d in prefix),minimum_recorded_J=min(d['J']for d in physical),maximum_recorded_principal_strain=max(d['strain']for d in physical),prefix_records_present=len(prefix_records),prefix_records_qualified=sum(r['qualified']for r in prefix_records),prefix_records_unqualified=sum(not r['qualified']for r in prefix_records),prefix_unresolved_levels=sum(r['unresolved_levels']for r in prefix_records),all_failures_retained=True,no_response_or_displacement_arrays_accessed=True,no_prediction_or_policy_claim=True,no_numerical_admission_claim=True,raw_arrays_LOCAL=True,wall_s=time.monotonic()-began,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    result.update(attempts)
    return result,dict(raw_file_SHA256=bindings,prefix_records=prefix_records)


STOP_SHA='6e8882f2b082618a9a8da688bfea7857de8b898b427c188ad6a94a8825eec36a'
MANIFEST_SHA='9afed9ce5e80fbaf79e5e244a12332f6d8bc7cf1fe2317eccfffe009e62b2818'


def manifest_gate(base):
    core.ensure(core.sha(base/'controlled-stop-ownership.json')==STOP_SHA,'Controlled-stop ownership SHA')
    core.ensure(core.sha(base/'stopped-completed-manifest-LOCAL.json')==MANIFEST_SHA,'Stopped immutable manifest SHA')
    ownership=core.read(base/'controlled-stop-ownership.json');manifest=core.read(base/'stopped-completed-manifest-LOCAL.json')
    core.ensure(ownership['protocol_SHA256']==PROTOCOL and manifest['protocol_SHA256']==PROTOCOL,'Stopped protocol binding')
    core.ensure(ownership['controlled_stop']and ownership['all16_workers_reaped']and ownership['all17_PID_groups_absent']and ownership['negative_run_preserved'],'Controlled-stop flags')
    core.ensure(ownership['completed_data_manifest_SHA256']==MANIFEST_SHA and ownership['execution_SHA256']==manifest['execution_SHA256']==core.sha(base/'execution.json'),'Stopped execution/manifest binding')
    core.ensure(manifest['all16_physics_retained']and manifest['no_completed_bytes_modified'],'Original completed bytes retained')
    for name,digest in manifest['files'].items():
        path=base/name;core.ensure(path.resolve().is_relative_to(base.resolve()),'Manifest path escapes namespace')
        core.ensure(core.sha(path)==digest,'Stopped completed artifact changed')
    return len(manifest['files'])


def attempt_counts(base,p):
    manifest=core.read(base/'stopped-completed-manifest-LOCAL.json');cache=core.read(base/'cache-ready.json');counts=dict(posterior_attempts=0,qualified_attempts=0,failed_attempts=0,prefix_attempts=0,prefix_qualified=0,prefix_failed=0,joint_attempts=0,joint_qualified=0,joint_failed=0)
    actions=sum(p['design']['actions'].values(),[])
    recipe=core.hashlib.sha256(json.dumps(dict(numerical_engine=p['numerical_engine'],tensor=p['integration_plan'],amplitude=p['amplitude_plan']),sort_keys=True).encode()).hexdigest()
    for name in manifest['files']:
        if not name.endswith('-LOCAL.json')or not any('/'+law+'-'in name for law in core.LAWS):continue
        prefix=name.endswith('-prefix-LOCAL.json');joint=any('-'+action+'-r'in name for action in actions)
        if not(prefix or joint):continue
        path=base/name;r=core.read(path)
        core.same(r['qualified'],r['diagnostics']['qualified'],'Attempt qualification binding')
        core.ensure(r['model_fingerprint']==cache['model_fingerprint']and r['recipe_SHA256']==recipe and r['numerical_engine']==p['numerical_engine'],'Stopped posterior model/recipe binding')
        if r.get('raw_SHA256'):
            raw=str(path.with_suffix('.npz').relative_to(base));core.ensure(manifest['files'][raw]==r['raw_SHA256'],'Attempt raw archive binding')
        counts['posterior_attempts']+=1;counts['qualified_attempts']+=r['qualified'];counts['failed_attempts']+=not r['qualified']
        key='prefix'if prefix else'joint';counts[key+'_attempts']+=1;counts[key+'_qualified']+=r['qualified'];counts[key+'_failed']+=not r['qualified']
    core.same(dict(posterior_attempts=49,qualified_attempts=38,failed_attempts=11,prefix_attempts=21,prefix_qualified=11,prefix_failed=10,joint_attempts=28,joint_qualified=27,joint_failed=1),counts,'Stopped negative inventory counts')
    counts.update(stopped_manifest_files_verified=len(manifest['files']),controlled_stop_ownership_SHA256=STOP_SHA,stopped_manifest_SHA256=MANIFEST_SHA,observation_fingerprints_not_reconstructed=True,no_held_target_arrays_deserialized=True)
    return counts


def guard():
    core.ensure(os.sched_getaffinity(0)=={12}and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU12/noGPU guard')
    core.ensure(all(os.environ.get(k)=='1'for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']),'One-thread audit guard')
    available=int(next(l for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')).split()[1])*1024
    core.ensure(available>=DESIGN['RAM_reserve_bytes']and shutil.disk_usage(ROOT).free>=DESIGN['disk_reserve_bytes'],'Audit RAM/disk reserve')


def recipe_sources():
    files=['experiments/audit_surface_p2_study_v03_failure_v02.py','tests/test_audit_surface_p2_study_v03_failure_v02.py','experiments/audit_surface_p2_study_v03.py','tests/test_audit_surface_p2_study_v03.py']
    return {name:core.sha(ROOT/name)for name in files}


def main():
    guard()
    recipe=core.read(AUDIT/'protocol.json')
    core.ensure(recipe['source_SHA256']==recipe_sources()and recipe['design']==DESIGN and recipe['study_protocol_SHA256']==PROTOCOL and recipe['controlled_stop_ownership_SHA256']==STOP_SHA and recipe['stopped_manifest_SHA256']==MANIFEST_SHA,'Prospective audit recipe changed')
    resource.setrlimit(resource.RLIMIT_AS,(DESIGN['RSS_cap_bytes'],DESIGN['RSS_cap_bytes']))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Audit900s cap')));signal.alarm(900)
    core.ensure(not(AUDIT/'result.json').exists(),'No implicit overwrite');AUDIT.mkdir(exist_ok=True,parents=True)
    result,pins=run();(AUDIT/'input-pins-LOCAL.json').write_text(json.dumps(pins,indent=2)+'\n');(AUDIT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
