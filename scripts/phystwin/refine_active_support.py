"""Additional development-only spatial refinement, independent of policy fitting."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.active_support_reference import protocol,solve_job


def main():
    spec=protocol();jobs=[]
    for i,r in enumerate(spec['development']):
        jobs.append((f'development-{i}-n2',r['E'],r['K'],r['layout'],2))
    # Lowest and highest n4-to-n8 development RMS discrepancy; no test labels used.
    for i in [2,11]:
        r=spec['development'][i]
        jobs.append((f'development-{i}-n12',r['E'],r['K'],r['layout'],12))
    with ProcessPoolExecutor(max_workers=2) as p:
        for result in p.map(solve_job,jobs):print(result,flush=True)

if __name__=='__main__':main()
