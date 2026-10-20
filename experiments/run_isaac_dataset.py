"""Bounded three-GPU dataset launcher. A failed episode is preserved, never replaced."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from isaac_dataset import digest_json
from validate_isaac_dataset import validate,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=Path('configs/isaac_boundary_orbit_v01.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--isaac-python',default='/home/bayespc2/vscode/deformable worldmodel/.venv/bin/python')
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.generation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads(args.plan.read_text());assert digest_json({k:v for k,v in plan.items() if k!='plan_sha256'})==plan['plan_sha256']
    frozen=root/'plan.json'
    if frozen.exists():assert frozen.read_bytes()==args.plan.read_bytes()
    else:frozen.write_bytes(args.plan.read_bytes())
    sources={p.name:sha(p) for p in [Path(__file__).with_name(n) for n in ['isaac_dataset_worker.py','isaac_dataset.py','boundary_block.py']]}
    (root/'episodes').mkdir(exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    def gpu_queue(gpu,cases):
        results=[]
        for case in cases:
            episode=root/'episodes'/case['episode_id'];started=time.time()
            try:
                assert all(sha(Path(__file__).with_name(n))==h for n,h in sources.items()),'Sources changed'
                if episode.exists():
                    audit=validate(episode,case);status='resumed_verified'
                else:
                    cmd=[args.isaac_python,'-u',str(Path(__file__).with_name('isaac_dataset_worker.py')),
                         '--plan',str(frozen),'--episode',case['episode_id'],'--gpu',str(gpu),'--output',str(episode)]
                    with (root/'logs'/f'{case["episode_id"]}.log').open('w') as log:
                        code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,
                            env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES'},timeout=600).returncode
                    assert code==0,f'Isaac worker exit {code}'
                    audit=validate(episode,case);status='generated_verified'
                (episode/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
                result={'episode_id':case['episode_id'],'status':status,'gpu':gpu,'seconds_elapsed':time.time()-started}
            except Exception as exc:
                result={'episode_id':case['episode_id'],'status':'failed','gpu':gpu,'error':repr(exc),'seconds_elapsed':time.time()-started}
            (root/'logs'/f'{case["episode_id"]}.status.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True);results.append(result)
        return results
    results=[]
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures=[pool.submit(gpu_queue,gpu,plan['episodes'][i::len(args.gpus)]) for i,gpu in enumerate(args.gpus)]
        for future in as_completed(futures):results.extend(future.result())
    results.sort(key=lambda x:x['episode_id']);failed=[r for r in results if r['status']=='failed']
    catalog={'dataset':plan['dataset'],'plan_sha256':plan['plan_sha256'],'source_sha256':sources,
             'completed':not failed,'episodes_verified':len(results)-len(failed),'episodes_failed':len(failed),'execution':results,
             'summaries':[json.loads((root/'episodes'/r['episode_id']/'summary.json').read_text()) for r in results if r['status']!='failed']}
    (root/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    files=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ['file-manifest.json','.generation.lock']]
    (root/'file-manifest.json').write_text(json.dumps({'files':files,'total_bytes':sum(f['bytes'] for f in files)},indent=2)+'\n')
    print(json.dumps({'verified':catalog['episodes_verified'],'failed':len(failed)}),flush=True)
    return bool(failed)


if __name__=='__main__':sys.exit(main())
