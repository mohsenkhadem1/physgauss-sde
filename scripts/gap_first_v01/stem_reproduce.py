"""Reproduce the sealed stem experiment in a fresh workspace without overwrites."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();source=Path(__file__).resolve().parents[2]
    target=args.workspace.resolve()
    if target.exists():raise SystemExit('Choose a fresh workspace; existing paths are never overwritten.')
    dest=target/'scripts/gap_first_v01';dest.mkdir(parents=True)
    for name in ['stem_campaign.py','stem_inference.py','stem_mechanics.py','stem_protocol.json','quantiles.py']:
        shutil.copy2(source/'scripts/gap_first_v01'/name,dest/name)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=str(target))
    for extra in [['--workers',str(args.workers)],['--evaluate']]:
        subprocess.run([sys.executable,'-m','scripts.gap_first_v01.stem_campaign',*extra],
                       cwd=target,env=env,check=True)
    actual=json.loads((target/'reports/gap-first-v01/stem/summary.json').read_text())
    expected_path=source/'reports/gap-first-v01/stem/summary.json'
    comparison={'workspace':str(target),'completed':True,'sources_copied_without_modification':True}
    if expected_path.exists():
        expected=json.loads(expected_path.read_text())
        comparison['summary_exactly_matches_reference']=actual==expected
        if actual!=expected:
            comparison['note']='Inspect numerical differences; reference files were not modified.'
    (target/'reproduction.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(comparison),flush=True)


if __name__=='__main__':main()
