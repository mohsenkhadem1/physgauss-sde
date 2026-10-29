"""Isolated physical-design diagnostic for smooth Gaussian attachment patches."""
from pathlib import Path
import argparse,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood import study
from scripts.position_ood.smooth_fixture import SmoothMaterialFixture

def configure():
 original=study.protocol();base=study.BASE;study.BASE=base/'smooth-reference';study.BASE.mkdir(exist_ok=True);study.protocol=lambda:original;study.MaterialPositionFixture=SmoothMaterialFixture
 stamp={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('smooth_fixture.py')]};path=study.BASE/'physical-source.json'
 if path.exists():assert json.loads(path.read_text())==stamp,'Smooth physical model changed'
 else:path.write_text(json.dumps(stamp,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--object',type=int,default=3);a=p.parse_args();configure();study.reference(a.split,a.object)
