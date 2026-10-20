"""Losslessly package a stopped local Bird run with file/part SHA-256 manifests.

Archives are split into small Git blobs on the separate artifact branch.
The coordinator's lock prevents hashing and copying a live, changing run.
"""
import argparse
import fcntl
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def archive(root,output,part_size=2_000_000,stage=None):
    root=Path(root).resolve();output=Path(output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise ValueError('Use a fresh archive filename')
    execution=json.loads((root/'execution.json').read_text())
    if stage is None:
        lock=(root/'coordinator.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        snapshot_lock=(root/'snapshot.lock').open('a')
        fcntl.flock(snapshot_lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        if execution['status'] not in ['finished','convergence_failed_at_declared_cap',
                'worker_failure_checkpoints_preserved']:
            raise ValueError('The run has not reached a declared stopped state')
        paths=sorted(p for p in root.rglob('*') if p.is_file() and p.name not in ['coordinator.lock','snapshot.lock'])
    else:
        # The watcher publishes each manifest entry only after its complete
        # immutable stage snapshot is copied. Current live shards are excluded.
        saved=json.loads((root/'stage-checkpoint-manifest.json').read_text())['checkpoints']
        for i in range(4):
            entry=saved[f'{stage}/chain-{i}']
            path=root/f'stage-checkpoints-{stage}'/f'chain-{i}'/'chains.pt'
            if entry['retained']!=stage or digest(path)!=entry['sha256']:
                raise ValueError('Completed stage checkpoint hash differs')
        if not (root/f'diagnostics-{stage}'/'convergence.json').exists():
            raise ValueError('Completed stage diagnostics are required')
        paths=sorted(p for directory in [root/f'stage-checkpoints-{stage}',root/f'diagnostics-{stage}']
            for p in directory.rglob('*') if p.is_file())
    files=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in paths]
    manifest={'execution':execution,'archived_stage':stage,'raw_root':root.name,'files':files,
        'inference_worker_commit':execution['commit'],
        'reassembly':f'cat {output.name}.part-* > {output.name}',
        'qualification':'Raw samples and failures are preserved; archived checkpoints are not automatically valid posteriors.'}
    content=(json.dumps(manifest,indent=2)+'\n').encode()
    with tarfile.open(output,'w:xz',preset=9) as tar:
        for path in paths:
            tar.add(path,arcname=f'{root.name}/{path.relative_to(root)}',recursive=False)
        info=tarfile.TarInfo('ARCHIVE-MANIFEST.json');info.size=len(content)
        tar.addfile(info,io.BytesIO(content))
    # Check that source bytes remained identical while packaging.
    for record in files:
        if digest(root/record['path'])!=record['sha256']:
            raise ValueError('A source file changed while archiving')
    parts=[]
    with output.open('rb') as source:
        index=0
        while block:=source.read(part_size):
            part=output.with_name(f'{output.name}.part-{index:03d}')
            if part.exists():raise ValueError('Archive part already exists')
            part.write_bytes(block)
            parts.append({'path':part.name,'bytes':len(block),'sha256':digest(part)})
            index+=1
    result={**manifest,'archive':output.name,'bytes':output.stat().st_size,
        'sha256':digest(output),'parts':parts}
    output.with_name(output.name+'-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['archive','bytes','sha256','parts']}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='runs/bird-local-production')
    parser.add_argument('--output',required=True)
    parser.add_argument('--stage',type=int,choices=[2000,4000,8000])
    args=parser.parse_args();archive(args.run,args.output,stage=args.stage)
