"""Continue spatial qualification after the current bounded development queue."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


if __name__=='__main__':
    base=ROOT/'runs/complex-scenes-v18'
    while True:
        paths=[base/l/'supervisor.json'for l in ['temporal-01','spatial-01','release-fix-n1','release-fix-n2']]
        if all(p.exists()and json.loads(p.read_text()).get('terminal',False)for p in paths):break
        time.sleep(10)
    with open(base/'spatial-02-supervisor.log','ab',buffering=0)as log:
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.complex_scenes_v18.supervise','--label','spatial-02','--workers','2','--n','3','--dt','.000025'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
