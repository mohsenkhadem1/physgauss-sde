"""Use CPU workers freed by primary evaluation to finish existing probe fits."""
from pathlib import Path
import argparse,fcntl,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,within_deadline
from scripts.position_ood.seal import verify
from scripts.position_ood.run_inverse import run
from scripts.phystwin.sequential_reference import write

def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);a=p.parse_args();seal=verify();directory=BASE/'active-jobs'
 while True:
  within_deadline();pending=0;progress=False
  for split,count in [('transfer_calibration',12),('transfer_test',9)]:
   for i in range(count):
    choices=directory/f'{split}-{i}-choices.json'
    if not choices.exists():pending+=1;continue
    choice=json.loads(choices.read_text());tasks=[('extrapolated','joint',action) for action in sorted(set(choice['actions'].values()))]
    if split=='transfer_test':tasks.append(('extrapolated_material_frame','material_frame',choice['actions']['adaptive']))
    for method,coarse,action in tasks:
     dest=BASE/f'inverse/{split}-{i}';tag=f'p4-0-1-2-{action}';output=dest/f'{method}-n3-{tag}.json'
     if output.exists():continue
     pending+=1
     if not (dest/f'{coarse}-n2-{tag}.json').exists() or not (dest/f'{method}-n3-p3.json').exists():continue
     with (directory/f'assist-{split}-{i}-{method}-{action}.lock').open('a') as claim:
      try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
      except BlockingIOError:continue
      with output.with_suffix('.lock').open('a') as peek:
       try:fcntl.flock(peek,fcntl.LOCK_EX|fcntl.LOCK_NB)
       except BlockingIOError:continue
       fcntl.flock(peek,fcntl.LOCK_UN)
      if output.exists():continue
      print('ASSIST',split,i,method,action,flush=True);run(split,i,method,3,seal['settings']['iterations'],(0,1,2,action));progress=True
  if pending==0:break
  if not progress:time.sleep(20)
 write(directory/f'assist-{a.worker}-done.json',dict(time=time.time()))
if __name__=='__main__':main()
