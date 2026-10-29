"""Numerical-recovery audit in an isolated cache, reusing immutable old levels."""
from pathlib import Path
import argparse,hashlib,json,os,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood import study
from scripts.position_ood.energy_recovery import EnergyRecoveryFixture

def configure(split,i):
 original=study.protocol();base=study.BASE;study.BASE=base/'energy-reference';study.BASE.mkdir(exist_ok=True);study.protocol=lambda:original;study.MaterialPositionFixture=EnergyRecoveryFixture
 source=base/f'references/{split}-{i}';dest=study.BASE/f'references/{split}-{i}';dest.mkdir(parents=True,exist_ok=True)
 for old in source.rglob('*.npz'):
  if '.tmp.' in old.name:continue
  target=dest/old.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():os.link(old,target)
 stamp={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('energy_recovery.py')]};path=study.BASE/'solver-source.json'
 if path.exists():assert json.loads(path.read_text())==stamp,'Recovery solver changed'
 else:path.write_text(json.dumps(stamp,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--object',type=int,default=3);a=p.parse_args();configure(a.split,a.object);study.reference(a.split,a.object)
