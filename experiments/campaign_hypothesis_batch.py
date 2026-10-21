"""Bounded controller for the exact96 retained-hypothesis engineering cohort.

No hypothesis sampling, response-dependent selection, retry, or physics change.
Only CPU concurrency may decrease with available resources. Raw outputs local.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
try:
    import campaign_hypothesis_dynamics as worker
    from campaign_exciting_commands import make_exciting_commands,PROGRAM_IDS
except ImportError:
    from experiments import campaign_hypothesis_dynamics as worker
    from experiments.campaign_exciting_commands import make_exciting_commands,PROGRAM_IDS

ROOT=worker.ROOT
SOURCE=Path(__file__).resolve()
BATCH_SCHEMA='finite-hypothesis-common-clock-batch-v0.1'
CELL_SCHEMA='independent-hypothesis-fixed-clock-batch-cell-v0.1'
CELL_STATUS='FROZEN_REVIEWED_FINITE_TRAIN_BATCH_CELL'


def read(path):return json.loads(Path(path).read_text())
def record(path):return {'path':Path(path).resolve().relative_to(ROOT).as_posix(),'sha256':worker.sha(path)}
def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def atomic(path,value):
    temp=Path(str(path)+'.tmp');dump(temp,value);temp.replace(path)


def validate_manifest(manifest):
    if (manifest.get('schema')!='finite-hardware-first-hypotheses-TRAIN-v0.1'
            or len(manifest.get('cells',[]))!=96 or len(manifest.get('hypotheses',[]))!=48
            or manifest.get('program_ids')!=list(PROGRAM_IDS)):
        raise ValueError('Exact retained48hypothesis/96cell manifest required')
    cells=manifest['cells'];ids=[c['id'] for c in cells]
    if len(set(ids))!=96 or any(c['split']!='train' or c['program_id'] not in PROGRAM_IDS for c in cells):
        raise ValueError('Exact unique TRAIN canonical cells required')
    counts={role:sum(c['evaluation_role']==role for c in cells) for role in ['fit','development_force_transfer','unseen_middle_material']}
    if list(counts.values())!=[32,32,32] or len({h['id'] for h in manifest['hypotheses']})!=48:
        raise ValueError('Original32/32/32 evaluation roles required')
    hypotheses={h['id']:h for h in manifest['hypotheses']};hardware={h['id']:h for h in manifest['public_hardware']}
    for c in cells:
        if c['hypothesis_id'] not in hypotheses or c['public_hardware_id'] not in hardware:
            raise ValueError('Unknown explicit hypothesis/hardware')
        if c['id']!=c['hypothesis_id']+'__'+c['program_id'] or hypotheses[c['hypothesis_id']]['public_hardware_id']!=c['public_hardware_id']:
            raise ValueError('Canonical cohort identity mismatch')
    return manifest


def prepare(manifest_path,output_prefix):
    manifest_path=worker.resolve(manifest_path);manifest=validate_manifest(read(manifest_path))
    prefix=worker.resolve(output_prefix)
    if prefix.exists():raise ValueError('Fresh canonical batch prefix required')
    prefix.mkdir(parents=True);inputs=prefix/'inputs';inputs.mkdir();protocols=prefix/'cell-protocols';protocols.mkdir()
    public_records={};hyp_records={};commands={}
    for public in manifest['public_hardware']:
        p=inputs/(public['id']+'.json');dump(p,public);public_records[public['id']]=record(p)
    for hyp in manifest['hypotheses']:
        p=inputs/(hyp['id']+'.json');dump(p,hyp['spec']);hyp_records[hyp['id']]=record(p)
    for program in PROGRAM_IDS:
        clock=make_exciting_commands(program);p=inputs/(program+'.npz')
        np.savez_compressed(p,**{k:getattr(clock,k) for k in ['endpoint_times_s','output_times_s','handle_forces_world_N',
            'handle_torques_world_Nm','nodal_forces_world_N','body_acceleration_world_m_s2']})
        commands[program]=record(p)
    declaration=dict(protocol=BATCH_SCHEMA,status='FROZEN_ROOT_AUTHORIZED_ENGINEERING96',
        manifest=record(manifest_path),integration_hz=120,settings=worker.SETTINGS,
        source_sha256=worker.runtime_sources(),controller_source=record(SOURCE),
        output_prefix=output_prefix,public_hardware=public_records,realized_hypotheses=hyp_records,
        known_commands=commands,canonical_cells=manifest['cells'],
        resources=dict(maximum_workers=8,threads_per_worker=2,GPU_used=False,
            memory_reserve_GiB=8,memory_per_worker_GiB=1),
        failure_policy='Stop new launches after any failure; drain running workers; preserve failed/queued; no redraw/retry.',
        production_dataset_qualified=False,all96_temporal_qualification=False)
    path=prefix/'batch-declaration.json';dump(path,declaration);batch_record=record(path)
    records=[]
    for cell in manifest['cells']:
        p=dict(protocol=CELL_SCHEMA,status=CELL_STATUS,split='TRAIN',settings=worker.SETTINGS,
            information_roles=worker.INFORMATION_ROLES,integration_hz=120,
            cell_id=cell['id'],evaluation_role=cell['evaluation_role'],program_id=cell['program_id'],
            output=output_prefix+'/cells/'+cell['id'],input_design=record(manifest_path),
            public_hardware=public_records[cell['public_hardware_id']],
            realized_hypothesis=hyp_records[cell['hypothesis_id']],known_commands=commands[cell['program_id']],
            source_sha256=worker.runtime_sources(),batch_declaration=batch_record,
            public_command_manifest=make_exciting_commands(cell['program_id']).public_manifest(),
            resources=dict(cpu_workers=1,threads_per_worker=2,gpu_used=False),
            qualification=worker.LIMITATION)
        cell_path=protocols/(cell['id']+'.json');dump(cell_path,p)
        records.append({**cell,'output':p['output'],'protocol':record(cell_path),'completed':False,
            'state':'queued','skip_trace':None})
    index=dict(schema='finite-hypothesis-batch-result-index-v0.1',manifest_path=str(manifest_path.relative_to(ROOT)),
        manifest_sha256=worker.sha(manifest_path),integration_hz=120,batch_declaration=batch_record,
        completed=False,complete=False,terminal=False,records=records)
    dump(prefix/'result-index.json',index)
    return batch_record


def validate_batch(path,digest):
    path=worker.checked({'path':path,'sha256':digest});batch=read(path)
    if (batch['protocol']!=BATCH_SCHEMA or batch['status']!='FROZEN_ROOT_AUTHORIZED_ENGINEERING96'
            or type(batch['integration_hz']) is not int or batch['integration_hz']!=120
            or batch['settings']!=worker.SETTINGS or batch['source_sha256']!=worker.runtime_sources()
            or batch['controller_source']!=record(SOURCE)):
        raise ValueError('Frozen canonical batch/source/settings changed')
    manifest=validate_manifest(read(worker.checked(batch['manifest'])))
    if batch['canonical_cells']!=manifest['cells']:raise ValueError('Batch cohort/roles differ from manifest')
    for group in ['public_hardware','realized_hypotheses','known_commands']:
        for artifact in batch[group].values():worker.checked(artifact)
    return batch,manifest


def resource_sample():
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
    return {'logical_cpus':os.cpu_count(),'load_average':list(os.getloadavg()),'memory_available_bytes':mem['MemAvailable']}


def resource_capacity(sample,resources):
    GiB=1024**3
    slots=max(0,int(sample['memory_available_bytes']//GiB)-resources['memory_reserve_GiB'])//resources['memory_per_worker_GiB']
    cap=min(resources['maximum_workers'],slots)
    return max(0,cap)


def completed_exact(cell):
    """Only identical own-batch artifacts; never import earlier smoke captures."""
    output=worker.resolve(cell['output'])
    if not output.exists():return False
    protocol_path=worker.checked(cell['protocol']);protocol=worker.validate_protocol(cell['protocol']['path'],cell['protocol']['sha256'])
    if not (output/'execution.json').exists():raise ValueError('Existing incomplete output retained; no retry')
    execution=read(output/'execution.json')
    if (not execution.get('terminal') or not execution.get('completed')
            or Path('/proc',str(execution['pid'])).exists() or (output/'failure.json').exists()):
        raise ValueError('Non-completed/live/failed output retained; no retry')
    if read(output/'protocol.json')!=protocol:raise ValueError('Existing completed protocol not identical')
    summary=read(output/'summary.json')
    if not summary['completed'] or summary['frames']!=61 or summary['steps']!=360:
        raise ValueError('Existing completed output contract differs')
    for name,key in [('trajectory.npz','trajectory_sha256'),('steps.jsonl','steps_sha256')]:
        if worker.sha(output/name)!=summary[key]:raise ValueError('Completed output digest changed')
    snapshots=read(output/'private-provenance/snapshots.json')
    expected={**protocol['source_sha256'],cell['protocol']['path']:cell['protocol']['sha256'],
        **{protocol[k]['path']:protocol[k]['sha256'] for k in ['public_hardware','realized_hypothesis','known_commands','batch_declaration','input_design']}}
    if {r['original_path']:r['sha256'] for r in snapshots}!=expected:
        raise ValueError('Completed output lacks exact full source/input snapshot set')
    for row in snapshots:
        if worker.sha(output/'private-provenance'/row['saved_path'])!=row['sha256']:
            raise ValueError('Existing source/input snapshot changed')
    with np.load(output/'trajectory.npz',allow_pickle=False) as raw:
        if (raw['positions_m'].shape!=(61,455,3) or raw['positions_m'].dtype!=np.float64
                or not np.array_equal(raw['times_s'],np.arange(61,dtype=np.float64)/20)
                or np.max(abs(raw['actual_state_times_s']-raw['times_s']))>1e-12
                or any(not np.isfinite(raw[k]).all() for k in raw.files)):
            raise ValueError('Completed output trajectory/clock contract differs')
    return True


def run(batch_path,batch_sha):
    batch,manifest=validate_batch(batch_path,batch_sha);prefix=worker.resolve(batch['output_prefix'])
    index_path=prefix/'result-index.json';index=read(index_path)
    if (index['manifest_sha256']!=batch['manifest']['sha256'] or index['batch_declaration']!={'path':batch_path,'sha256':batch_sha}
            or [c['id'] for c in index['records']]!=[c['id'] for c in manifest['cells']]):
        raise ValueError('Result index must retain exact96 canonical manifest IDs')
    for row,canonical in zip(index['records'],manifest['cells']):
        if any(row[k]!=v for k,v in canonical.items()):raise ValueError('Result index canonical metadata changed')
        if completed_exact(row):row.update(completed=True,state='completed',skip_trace='Identical frozen own-batch protocol/source/input/terminal/output hashes verified; no rerun.')
        elif row['state']!='queued':raise ValueError('Interrupted/failed state retained; explicit recovery required')
    live={};stopped=False;supervisor=dict(pid=os.getpid(),terminal=False,source_sha256=worker.sha(SOURCE))
    index['supervisor']=supervisor;index['resource_samples']=[];atomic(index_path,index)
    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':''}
    try:
        while True:
            for pid,(process,cell,log) in list(live.items()):
                if (worker.resolve(cell['output'])/'failure.json').exists():stopped=True
                code=process.poll()
                if code is None:continue
                process.wait();log.close();del live[pid]
                try:
                    if code!=0 or not completed_exact(cell):raise RuntimeError('Worker terminal contract failed')
                    cell.update(completed=True,state='completed',exit_code=code,terminal_pid_missing=not Path('/proc',str(pid)).exists())
                except BaseException as error:
                    stopped=True;cell.update(completed=False,state='failed',exit_code=code,error=str(error))
            sample=resource_sample();sample['time_unix_s']=time.time();index['resource_samples'].append(sample)
            # Keep bounded telemetry without dropping terminal/case evidence.
            index['resource_samples']=index['resource_samples'][-512:]
            cap=resource_capacity(sample,batch['resources'])
            for cell in index['records']:
                if stopped or len(live)>=cap:break
                if cell['state']!='queued':continue
                validate_batch(batch_path,batch_sha)
                worker.validate_protocol(cell['protocol']['path'],cell['protocol']['sha256'])
                output=worker.resolve(cell['output'])
                if output.exists():raise ValueError('Fresh pending output required; no retry')
                logs=prefix/'worker-logs';logs.mkdir(exist_ok=True);log=(logs/(cell['id']+'.log')).open('x')
                command=[sys.executable,str(worker.SOURCE),'--protocol',cell['protocol']['path'],'--protocol-sha256',cell['protocol']['sha256']]
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                cell.update(state='running',pid=process.pid,command=command)
                live[process.pid]=(process,cell,log)
            index.update(completed=all(c['completed'] for c in index['records']),complete=all(c['completed'] for c in index['records']),stopped_on_failure=stopped,
                completed_count=sum(c['completed'] for c in index['records']),live_count=len(live))
            atomic(index_path,index)
            if not live and (stopped or index['complete']):break
            if not live and cap==0:raise RuntimeError('Insufficient available memory; queued cells retained, no physics changes')
            time.sleep(2)
    except BaseException:
        # Preserve and drain owned workers on interruption/error; never kill other jobs.
        for process,cell,log in live.values():
            code=process.wait();log.close()
            try:ok=code==0 and completed_exact(cell)
            except BaseException as error:
                ok=False;cell['error']=str(error)
            cell.update(exit_code=code,state='completed' if ok else 'failed',completed=ok)
        index['controller_exception']=True;raise
    finally:
        index.update(completed=all(c['completed'] for c in index['records']),complete=all(c['completed'] for c in index['records']),terminal=True,live_count=0,
            completed_count=sum(c['completed'] for c in index['records']))
        index['supervisor']['terminal']=True;atomic(index_path,index)
    return index


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--manifest',required=True);prep.add_argument('--output-prefix',required=True)
    execute=sub.add_parser('run');execute.add_argument('--batch',required=True);execute.add_argument('--batch-sha256',required=True)
    args=parser.parse_args()
    if args.action=='prepare':print(json.dumps(prepare(args.manifest,args.output_prefix)),flush=True)
    else:
        result=run(args.batch,args.batch_sha256)
        print(json.dumps({k:result[k] for k in ['completed','complete','terminal','completed_count']}),flush=True)
        if not result['complete']:raise SystemExit(1)
