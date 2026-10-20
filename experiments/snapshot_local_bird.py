"""Preserve exact per-worker stage checkpoints alongside the running coordinator."""
import argparse
import fcntl
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import torch


def watch(root):
    root=Path(root);seen={};saved={}
    lock=(root/'snapshot.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    destination=root/'stage-checkpoint-manifest.json'
    if destination.exists():saved=json.loads(destination.read_text())['checkpoints']
    while True:
        for index in range(4):
            path=root/'shards'/f'chain-{index}'/'chains.pt'
            if not path.exists():continue
            stat=path.stat();stamp=(stat.st_size,stat.st_mtime_ns)
            if seen.get(index)==stamp:continue
            payload=path.read_bytes();data=torch.load(io.BytesIO(payload),weights_only=True)
            seen[index]=stamp;budget=data['retained']
            if budget not in [2000,4000,8000]:continue
            key=f'{budget}/chain-{index}';digest=hashlib.sha256(payload).hexdigest()
            out=root/f'stage-checkpoints-{budget}'/f'chain-{index}';out.mkdir(parents=True,exist_ok=True)
            if (out/'chains.pt').exists() and hashlib.sha256((out/'chains.pt').read_bytes()).hexdigest()!=digest:
                raise ValueError('A preserved stage checkpoint differs')
            (out/'chains.pt').write_bytes(payload)
            for name in ['local_protocol.json','reference.pt','device-validation.json']:
                source=path.parent/name
                if source.exists() and not (out/name).exists():shutil.copyfile(source,out/name)
            saved[key]={'sha256':digest,'retained':budget,'warmup':data['completed_warmup'],
                'seed':data['protocol']['seed'],'exact_raw_bytes':True}
            temporary=destination.with_suffix('.tmp')
            temporary.write_text(json.dumps({'checkpoints':saved},indent=2)+'\n');temporary.replace(destination)
            print(json.dumps({'preserved':key,**saved[key]}),flush=True)
        execution=root/'execution.json'
        if execution.exists():
            status=json.loads(execution.read_text())['status']
            if status in ['finished','convergence_failed_at_declared_cap','worker_failure_checkpoints_preserved']:
                break
        time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='runs/bird-local-production')
    args=parser.parse_args();torch.set_num_threads(1);watch(args.run)
