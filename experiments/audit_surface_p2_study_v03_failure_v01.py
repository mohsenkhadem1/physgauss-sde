"""Partial negative-run audit: provenance/physics only, no target scoring.

Never accesses matching_linear/NH80_stress response arrays or displacement
field arrays. Hashes raw files, reads only original prior coordinates and bound
physical-diagnostic scalar records. A controlled stop and nonzero child exits
are preserved; all16 owned slots must still be reaped and absent.
"""
import json,os,resource,signal,time
from pathlib import Path
import numpy as np
from experiments import audit_surface_p2_study_v03 as core

ROOT=core.ROOT;LOCAL=core.LOCAL;PROTOCOL=core.PROTOCOL
AUDIT=ROOT/'runs/operator-dev/surface-p2-study-v03-negative-audit-v01'
DESIGN=dict(schema='surface-p2-study-v03-negative-audit-v01',CPU=17,threads=1,GPU=False,
    wall_cap_s=900,RSS_cap_bytes=4*1024**3,objects=16,physical_records=320,
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
    began=time.monotonic();core.ensure(core.sha(base/'protocol.json')==digest,'Original protocol hash')
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
    return dict(schema=DESIGN['schema'],completed=True,audit_passed=True,scope='Original failed/stopped study: physical/raw provenance only; no held-target reads or policy comparison',protocol_SHA256=digest,execution_SHA256=core.sha(base/'execution.json'),all17_owned_PID_and_groups_absent=True,worker_slots_retained=16,nonzero_worker_exits_retained=sum(r['exit_code']!=0 for r in e['records']),original_execution_completed=e['completed'],original_execution_status=e['status'],objects_retained=16,original_prior_draws_verified=16,physical_records_verified=320,physical_records_valid=sum(d['valid']for d in physical),prefix_objects_verified=16,prefix_law_signals_verified=32,prefix_law_signals_valid=sum(d['valid']for d in prefix),minimum_prefix_nonrigid_RMS_m=min(d['RMS']for d in prefix),minimum_prefix_nonrigid_fraction=min(d['fraction']for d in prefix),minimum_recorded_J=min(d['J']for d in physical),maximum_recorded_principal_strain=max(d['strain']for d in physical),prefix_records_present=len(prefix_records),prefix_records_qualified=sum(r['qualified']for r in prefix_records),prefix_records_unqualified=sum(not r['qualified']for r in prefix_records),prefix_unresolved_levels=sum(r['unresolved_levels']for r in prefix_records),all_failures_retained=True,no_response_or_displacement_arrays_accessed=True,no_prediction_or_policy_claim=True,no_numerical_admission_claim=True,raw_arrays_LOCAL=True,wall_s=time.monotonic()-began,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),dict(raw_file_SHA256=bindings,prefix_records=prefix_records)


def main():
    core.ensure(os.sched_getaffinity(0)=={17}and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU17/noGPU guard')
    core.ensure(all(os.environ.get(k)=='1'for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']),'One-thread audit guard')
    resource.setrlimit(resource.RLIMIT_AS,(DESIGN['RSS_cap_bytes'],DESIGN['RSS_cap_bytes']))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Audit900s cap')));signal.alarm(900)
    core.ensure(not(AUDIT/'result.json').exists(),'No implicit overwrite');AUDIT.mkdir(exist_ok=True,parents=True)
    result,pins=run();(AUDIT/'input-pins-LOCAL.json').write_text(json.dumps(pins,indent=2)+'\n');(AUDIT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
