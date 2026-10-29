"""Fit qualified objects as they become available, without waiting on index zero."""
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,protocol,within_deadline
from scripts.position_ood.run_inverse import run
from scripts.phystwin.sequential_reference import write

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--method',default='extrapolated');p.add_argument('--mesh',type=int,default=3);p.add_argument('--iterations',type=int,default=80);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
 pending=set(range(a.worker,len(protocol()[a.split]),a.workers));failures={};status=BASE/f'ready-{a.split}-{a.method}-{a.worker}.json'
 while pending:
  within_deadline();progress=False
  for i in sorted(pending):
   if not (BASE/f'fields/{a.split}-{i}.npz').exists():continue
   try:run(a.split,i,a.method,a.mesh,a.iterations)
   except Exception as e:failures[str(i)]=repr(e);print('FIT_FAILED',i,repr(e),flush=True)
   pending.remove(i);progress=True;write(status,dict(pending=sorted(pending),failures=failures,time=time.time()))
  if not progress:time.sleep(20)
 write(status,dict(state='complete' if not failures else 'completed_with_failures',failures=failures,time=time.time()))
if __name__=='__main__':main()
