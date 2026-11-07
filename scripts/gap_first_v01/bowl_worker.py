"""Durable single-GPU bowl worker; launch with explicit CUDA_VISIBLE_DEVICES.

Each integration checkpoints state. Launch under nohup/setsid; this worker never
terminates or changes any other process. GPU3 is never assigned by this study.
"""
import argparse,json,os,time
from pathlib import Path
from .bowl_campaign import generate_gpu,ROOT,REPORT,INFERENCE_N,REFERENCE_N,THETA,DEADLINE_UNIX,protocol_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['inference-and-target','reference-push','reference-rotation']);a=p.parse_args()
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if devices not in ('0','1','2'):raise RuntimeError('Assign exactly one authorized physical GPU0,1,or2')
    schedules={'inference-and-target':[(INFERENCE_N,None),(REFERENCE_N,'heldout_push')],
               'reference-push':[(REFERENCE_N,'calibration_push')],
               'reference-rotation':[(REFERENCE_N,'calibration_tilt')]}
    status=REPORT/f'worker-{a.role}.json'
    started=time.time()
    for n,interaction in schedules[a.role]:
        if time.time()>=DEADLINE_UNIX:raise RuntimeError('Campaign deadline reached')
        status.write_text(json.dumps(dict(status='running',pid=os.getpid(),physical_gpu=devices,started_unix=started,n=n,interaction=interaction,protocol_sha256=protocol_hash()),indent=2))
        generate_gpu(n,interaction)
        expected=('calibration_push','calibration_tilt','heldout_push') if interaction is None else (interaction,)
        missing=[str(ROOT/f'fem-n{n}'/f'h{h:02d}-{name}.npz') for name in expected for h in range(len(THETA)) if not (ROOT/f'fem-n{n}'/f'h{h:02d}-{name}.npz').exists()]
        if missing:
            status.write_text(json.dumps(dict(status='incomplete',missing=missing,physical_gpu=devices,requires='Finish all parallel workers then explicitly attempt missing CPU trajectories; retain failures'),indent=2))
            return
    status.write_text(json.dumps(dict(status='complete',physical_gpu=devices,elapsed_s=time.time()-started,protocol_sha256=protocol_hash()),indent=2))

if __name__=='__main__':main()
