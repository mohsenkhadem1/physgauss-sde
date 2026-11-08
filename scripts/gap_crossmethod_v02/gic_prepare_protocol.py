"""Freeze native GIC Torus prefix/view split; official segmentation already run."""
import hashlib,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[2]
data=R/'runs/gap-crossmethod-v02/gic-inputs/torus'
out=R/'reports/gap-crossmethod-v02/gic-protocol.json'
if out.exists():raise RuntimeError('Protocol already frozen')
p=data/'transforms_train.json';raw=p.read_bytes();(data/'transforms_native_all.json').write_bytes(raw)
a=json.loads(raw);train=dict(a);test=dict(a)
def ids(f):return tuple(map(int,f['file_path'].split('/')[-1].split('_')[1:]))
train['frames']=[f for f in a['frames'] if ids(f)[0]!=0 and ids(f)[1]<8]
test['frames']=[f for f in a['frames'] if ids(f)[0]==0]
assert len(train['frames'])==80 and len(test['frames'])==14
p.write_text(json.dumps(train,indent=2))
for name in ('test','val'):(data/f'transforms_{name}.json').write_text(json.dumps(test,indent=2))
c=json.loads((R/'runs/gap-crossmethod-v02/vendor/gic/config/pacnerf/torus.json').read_text())
c['physics']['n_frames']=8
c['gs']['data_device']='cpu';c['gs']['load2gpu_on_the_fly']=True
# Keep native40k appearance and100-iteration physical identification budgets.
config=R/'runs/gap-crossmethod-v02/gic-torus-prefix.json';config.write_text(json.dumps(c,indent=2))
files=[p,data/'transforms_test.json',config]
protocol=dict(frozen_unix=time.time(),method='native GIC b523851ed4343109ff67b0ea123e0a154af2a40c',rasterizer='b1e1cb83e27923579983a9ed19640c6031112b94',observations='Official foreground masks and RGB, fitting cameras1–10 frames0–7 only',test='Camera0 frames0–7 held camera; frames8–13 held camera and future',metrics=['PSNR','SSIM','LPIPS if dependency available','trajectory Chamfer when native groundtruth available'],changes=['Explicit camera0 holdout removes native preparation overlap','8-frame temporal prefix instead of14','CPU image storage/on-the-fly transfers to fit11GB GPU; numerical model unchanged'],native_budgets=dict(appearance=40000,velocity=100,physical=100),intervention='Native changed gravity forward diagnostic without measured intervention reference; no OOD accuracy claim',source_sha256={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},no_assumed_win=True)
out.write_text(json.dumps(protocol,indent=2));print(out)
