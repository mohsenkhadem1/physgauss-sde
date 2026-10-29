"""Development-only end-to-end check of the matched spatial-frame ablation."""
from pathlib import Path
import argparse,sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,within_deadline
from scripts.position_ood.run_inverse import run
from scripts.position_ood.predict_transfer import run as transfer
p=argparse.ArgumentParser();p.add_argument('--object',type=int,required=True);a=p.parse_args();i=a.object
for method,n in [('material_frame',2),('extrapolated_material_frame',3)]:
 run('development',i,method,n,80);transfer('transfer_development',i,method,n)
choice=BASE/f'active-jobs/transfer_development-{i}-choices.json'
while not choice.exists():within_deadline();time.sleep(20)
action=json.loads(choice.read_text())['actions']['adaptive']
for method,n in [('material_frame',2),('extrapolated_material_frame',3)]:run('transfer_development',i,method,n,80,(0,1,2,action))
