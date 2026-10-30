"""Predeclared stable native development pair refinement, no reserved inputs."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.native_consistent_development import job,REPORT
from scripts.coupled_contact_v13.campaign import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--dt',type=float,required=True);p.add_argument('--tag',required=True);a=p.parse_args()
    if (REPORT/'method-freeze.json').exists():raise RuntimeError('Development only before freeze')
    if a.dt<=0 or '/' in a.tag or '..' in a.tag:raise ValueError('Invalid settings')
    source=REPORT/'native-timestep-consistent-engineered-candidate4.json';original=json.loads(source.read_text())
    spec=dict(next(r['spec'] for r in original['rows'] if r['spec']['scene']=='curved' and r['spec']['scale']==1e6))
    spec.update(dt=a.dt,tag=a.tag,name='curved-1000000-refined')
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in original['source_sha256']}
    hashes[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    declaration=dict(jobs=[spec],source_sha256=hashes,source_report_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope=__doc__)
    write(REPORT/f'native-timestep-{a.tag}-declaration.json',declaration)
    row=job(spec)
    write(REPORT/f'native-timestep-{a.tag}.json',dict(**declaration,rows=[row]))
if __name__=='__main__':main()
