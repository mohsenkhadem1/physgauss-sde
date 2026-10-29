"""Material-indexed load ablation using the exact same selected probe response."""
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,within_deadline
from scripts.position_ood.predict_transfer import run as transfer
from scripts.position_ood.run_inverse import run as fit
p=argparse.ArgumentParser();p.add_argument('--split',default='transfer_test');p.add_argument('--kind',choices=['cpu','gpu'],required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args();original='test_id' if a.split=='transfer_test' else 'development';count=9 if original=='test_id' else 3
pending=set(range(a.worker,count,a.workers))
while pending:
 within_deadline();progress=False
 for i in sorted(pending):
  source=BASE/f'inverse/{original}-{i}';dest=BASE/f'inverse/{a.split}-{i}';choice=BASE/f'active-jobs/{a.split}-{i}-choices.json'
  if not choice.exists() or not (source/'extrapolated_material_frame-n3-p3.json').exists():continue
  action=json.loads(choice.read_text())['actions']['adaptive'];prefix=(0,1,2,action);tag=f'p4-0-1-2-{action}'
  if a.kind=='cpu':
   if not (dest/'material_frame-n2-p3.json').exists():transfer(a.split,i,'material_frame',2)
   if not (dest/'extrapolated_material_frame-n3-p3.json').exists():transfer(a.split,i,'extrapolated_material_frame',3)
   fit(a.split,i,'material_frame',2,80,prefix);pending.remove(i);progress=True
  elif (dest/f'material_frame-n2-{tag}.json').exists() and (dest/'extrapolated_material_frame-n3-p3.json').exists():
   fit(a.split,i,'extrapolated_material_frame',3,80,prefix);pending.remove(i);progress=True
 if not progress:time.sleep(20)
