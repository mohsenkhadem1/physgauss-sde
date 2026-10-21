"""Exactly six fixed fits after complete96 capture, one child per physical GPU."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.prepare_campaign_hypothesis_dataset import prepare

PROTOCOL=ROOT/'runs/operator-dev/crossed-gaussian-training-v01/protocol.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def run():
    d=json.loads(PROTOCOL.read_text())
    def pins():
        for name,digest in d['source_sha256'].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Frozen training dependency changed: '+name)
    pins()
    assert sha(ROOT/d['input_manifest']['path'])==d['input_manifest']['sha256']
    assert sha(ROOT/d['finite_batch_declaration']['path'])==d['finite_batch_declaration']['sha256']
    index=json.loads((ROOT/d['finite_index']).read_text())
    manifest=json.loads((ROOT/d['input_manifest']['path']).read_text())
    assert index['terminal'] is True and index['completed'] is True
    assert index['manifest_sha256']==d['input_manifest']['sha256'] and index['integration_hz']==120
    assert len(index['records'])==96 and len({r['id'] for r in index['records']})==96
    assert {r['id'] for r in index['records']}=={c['id'] for c in manifest['cells']}
    assert all(r['completed'] is True for r in index['records'])
    stage=ROOT/d['output'];stage.mkdir(parents=True,exist_ok=False)
    jobs=[dict(row,state='queued',pid=None,exit_code=None) for row in d['jobs']]
    status=dict(supervisor_pid=os.getpid(),terminal=False,completed=False,jobs=jobs,
        protocol_sha256=sha(PROTOCOL),source_sha256=d['source_sha256'])
    write(stage/'execution.json',status);started=time.monotonic();active={};failed=False
    try:
        dataset_report=prepare(d['input_manifest']['path'],d['finite_index'],d['output']+'/dataset')
        dataset=stage/'dataset/dataset.npz';dataset_hash=sha(dataset)
        status.update(dataset_sha256=dataset_hash,finite_index_sha256=sha(ROOT/d['finite_index']),dataset_manifest_sha256=sha(stage/'dataset/manifest.json'))
        write(stage/'execution.json',status)
        while any(j['state']=='queued' for j in jobs) or active:
            pins()
            for gpu in [1,2]:
                if failed or gpu in active:continue
                row=next((j for j in jobs if j['physical_GPU']==gpu and j['state']=='queued'),None)
                if row is None:continue
                if sha(dataset)!=dataset_hash:raise RuntimeError('Frozen dataset changed')
                output=stage/row['id'];assert not output.exists()
                cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'experiments/train_campaign_crossed_prediction.py'),
                    '--dataset',str(dataset),'--output',str(output),'--device','cuda:0',
                    '--arm',row['arm'],'--steps','4000','--seed',str(row['seed'])]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TORCH_DISABLE_NATIVE_JIT='1',
                    OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
                log=(stage/(row['id']+'.log')).open('x')
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                row.update(state='running',pid=process.pid,command=cmd,physical_GPU=gpu,
                    CUDA_VISIBLE_DEVICES=str(gpu),started_elapsed_s=time.monotonic()-started)
                active[gpu]=(process,log,row);write(stage/'execution.json',status)
                print(json.dumps(dict(started=row['id'],pid=process.pid,physical_GPU=gpu)),flush=True)
            for gpu,(process,log,row) in list(active.items()):
                code=process.poll()
                if code is None:continue
                log.close();row.update(exit_code=code,ended_elapsed_s=time.monotonic()-started)
                output=stage/row['id'];path=output/'execution.json'
                execution=json.loads(path.read_text()) if path.exists() else {}
                complete=code==0 and execution.get('terminal') is True and execution.get('completed') is True
                if complete:
                    protocol=json.loads((output/'protocol.json').read_text())
                    complete=(protocol['dataset_sha256']==dataset_hash and protocol['steps']==4000
                        and protocol['arm']==row['arm'] and protocol['seed']==row['seed']
                        and sha(output/'final.pt')==execution['checkpoint_sha256']
                        and sha(output/'predictions.npz')==execution['predictions_sha256'])
                row['state']='completed' if complete else 'failed_preserved'
                if not complete:failed=True
                del active[gpu];write(stage/'execution.json',status)
                print(json.dumps(dict(finished=row['id'],state=row['state'],exit_code=code)),flush=True)
            if failed:
                for row in jobs:
                    if row['state']=='queued':row['state']='never_launched_after_failure'
            if active:time.sleep(1)
        status.update(terminal=True,completed=all(j['state']=='completed' for j in jobs),elapsed_s=time.monotonic()-started,live_worker_pids=[])
        pins();write(stage/'execution.json',status)
        if not status['completed']:raise RuntimeError('Preserved fit failure; no retry or dropped seed')
        return status
    except BaseException as error:
        status.update(error=type(error).__name__,message=str(error),live_worker_pids=[p.pid for p,_,_ in active.values() if p.poll() is None])
        write(stage/'execution.json',status)
        # Never launch a replacement. Preserve running peers and drain them.
        for process,log,row in active.values():
            row.update(exit_code=process.wait(),state='drained_after_stage_failure');log.close()
        for row in jobs:
            if row['state']=='queued':row['state']='never_launched_after_failure'
        status.update(terminal=True,completed=False,elapsed_s=time.monotonic()-started,live_worker_pids=[])
        write(stage/'execution.json',status);raise

if __name__=='__main__':run()
