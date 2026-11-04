"""Bounded, resumable preprocessing of six public objects, without scoring.

GPU 0 is reserved for reconstruction; mask workers share GPUs 1 and 2.
This prepares episode zero only; it does not run a physical benchmark.
"""
import json, os, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/real-transfer-v19'
OBJECTS=['046-sponge','043-dog','047-rectangle-sponge','048-butter-sponge','049-ball','051-cube']

def main():
    destination=BASE/'cohort-preparation';destination.mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    python=str(ROOT/'runs/deform360-env/bin/python')
    for obj in OBJECTS:
        status=destination/(obj+'.json')
        steps=[
            ('align',[str(ROOT/'runs/deform360-env/bin/deform360-undistort'),
                      '--object-dir',str(BASE/'Deform360/raw'/obj),
                      '--output-dir',str(BASE/'Deform360/processed'/obj),
                      '--episodes','0','--no-overwrite']),
            ('masks',[python,'-m','scripts.real_transfer_v19.deform360_mask_queue',
                      '--object',obj,'--devices','cuda:1','cuda:2']),
            ('geometry',[python,'-m','scripts.real_transfer_v19.deform360_mask_geometry','--object',obj]),
        ]
        record=dict(object=obj,episode=0,started=time.time(),steps=[])
        for stage,command in steps:
            record['current_stage']=stage
            status.write_text(json.dumps(record,indent=2)+'\n')
            with (destination/f'{obj}-{stage}.log').open('ab') as log:
                try:
                    code=subprocess.run(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                        stdout=log,stderr=subprocess.STDOUT,timeout=7200).returncode
                except subprocess.TimeoutExpired:code=124
            record['steps'].append(dict(stage=stage,returncode=code,finished=time.time()))
            if code:break
        record.update(finished=time.time(),complete=len(record['steps'])==3 and all(s['returncode']==0 for s in record['steps']))
        status.write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)

if __name__=='__main__':main()
