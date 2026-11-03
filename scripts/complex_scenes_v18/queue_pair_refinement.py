"""Queue the next mesh-aligned pair check on the explicitly selected GPU."""
import json,subprocess,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


if __name__=='__main__':
    base=ROOT/'runs/complex-scenes-v18'
    while True:
        paths=[base/'pair-aligned-n3-gpu'/f'deformable_pair-v{v}-n3-dt0.00002500-s0'/'complete.json'for v in (0,1)]
        if all(p.exists()for p in paths):
            if not all(json.loads(p.read_text())['completed']for p in paths):raise RuntimeError('Previous refinement failed; inspect before proceeding')
            break
        time.sleep(10)
    for variant in (0,1):
        with open(base/f'pair-aligned-n4-gpu-v{variant}.log','ab',buffering=0)as log:
            cmd=[str(ROOT/'.venv/bin/python'),'-m','scripts.complex_scenes_v18.run_development','--case','deformable_pair','--variant',str(variant),'--n','4','--dt','.000025','--surface-samples','0','--contact-device','cuda','--seconds','3600','--label','pair-aligned-n4-gpu']
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
