"""Additional development-only timestep refinement of explicitly failed pairs."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,multiprocessing,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.native_timestep_development import job,REPORT
from scripts.coupled_contact_v13.campaign import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-report',required=True);parser.add_argument('--tag',required=True);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if (REPORT/'method-freeze.json').exists():raise RuntimeError('Development refinement must precede fresh freeze')
    if '/' in args.tag or '..' in args.tag:raise ValueError('Simple tag required')
    path=REPORT/args.source_report;source=json.loads(path.read_text())
    specs=[]
    for scene,scale in [('squeeze',1e5),('squeeze',1e6),('curved',1e5),('curved',1e6)]:
        candidates=[r for r in source['rows'] if r['spec']['scene']==scene and r['spec']['scale']==scale]
        finest=min(candidates,key=lambda r:r['spec']['dt'])
        if not finest['success']:
            spec=dict(finest['spec']);spec.update(dt=spec['dt']/2,tag=args.tag,name=f'{scene}-{int(scale)}-extra-half');specs.append(spec)
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source['source_sha256']}
    hashes[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    declaration=dict(source_report=str(path.relative_to(ROOT)),source_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256=hashes,jobs=specs,
                     policy='Only finest-step failed development pairs get one additional halving; no reserved cases, model tuning or failure substitution.')
    write(REPORT/f'native-timestep-{args.tag}-declaration.json',declaration)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(job,specs))
    write(REPORT/f'native-timestep-{args.tag}.json',dict(**declaration,rows=rows))
    print(json.dumps([{k:r[k] for k in ['spec','success','error','last_successful_time_s','last_successful_min_j','stopped_state_min_j']} for r in rows],indent=2))
if __name__=='__main__':main()
