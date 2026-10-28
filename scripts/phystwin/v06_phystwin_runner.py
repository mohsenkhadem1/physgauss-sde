"""Finish both observation budgets and a timestep check for one bridge object."""
from pathlib import Path
import argparse,subprocess
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--object',type=int,required=True);a=p.parse_args()
 for budget in ['initial','two_probes']:
  out=ROOT/'runs/phystwin-sequential-support-v06/phystwin'/f'object-{a.object}-{budget}'
  for refinement in [1,2]:
   if (out/f'summary-dt{refinement}.json').exists():continue
   args=['--object',str(a.object),'--budget',budget,'--iterations','30','--step-refinement',str(refinement)]
   if refinement==2:args+=['--evaluate-only']
   subprocess.run([str(ROOT/'runs/phystwin-env/bin/python'),str(ROOT/'scripts/phystwin/run_v06_phystwin.py'),*args],cwd=ROOT,check=True)
 print('PHYSTWIN_OBJECT_COMPLETE',a.object,flush=True)
if __name__=='__main__':main()
