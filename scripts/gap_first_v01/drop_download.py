"""Download official release; three lexicographically first assets, no result selection."""
from pathlib import Path
import concurrent.futures,hashlib,json,urllib.request,datetime
BASE=Path('runs/gap-first-v01/drop');OUT=Path('reports/gap-first-v01/drop');OUT.mkdir(parents=True,exist_ok=True)
assets=['00e2a405dcff4701a8f57d2e67e306b8','01b7a0ecc6124991a6c67b4be0e0218f','02030956a7624c5ebaf329207cd69dd8']
jobs=[('https://huggingface.co/chuanshuogushi/ReconPhys/resolve/main/ckpt.pt',BASE/'ckpt.pt')]
for a in assets:
 for f in ['gaussian.ply','sample_00000/physics_params.pkl','sample_00000/view0.mp4','sample_00000/view1.mp4','sample_00000/view2.mp4','sample_00000/view3.mp4','sample_00000/states/frame_0000.pt']:
  jobs.append((f'https://huggingface.co/datasets/chuanshuogushi/ReconPhys_dataset/resolve/main/{a}/{f}',BASE/'assets'/a/f))
def get(job):
 url,p=job;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():urllib.request.urlretrieve(url,p.with_suffix(p.suffix+'.partial'));p.with_suffix(p.suffix+'.partial').replace(p)
 d=dict(url=url,path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest());print(str(p),d['bytes'],flush=True);return d
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(get,jobs))
(OUT/'downloads.json').write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection='First three dataset directories in lexicographic order, sample00000. No outcome selection.',files=results),indent=2))
