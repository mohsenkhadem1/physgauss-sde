"""Small crash-recovery status file; no raw data is sent to the session."""
from pathlib import Path
import json,subprocess,time,datetime,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.runtime import ROOT,BASE,CONFIG
while True:
 spec=json.loads(CONFIG.read_text());deadline=datetime.datetime.fromisoformat(spec['campaign_deadline_utc'].replace('Z','+00:00')).timestamp()
 if time.time()>=deadline:break
 cohorts=json.loads((BASE/'cohorts.json').read_text());counts={}
 for split in ['development','calibration','test_id','test_material','test_environment']:
  counts[split]=dict(expected=len(cohorts[split]),references=len(list((BASE/'fields').glob(split+'-*.npz'))),joint_fits=len(list((BASE/'inverse').glob(split+'-*/joint-n2-p3.json'))),extrapolated_fits=len(list((BASE/'inverse').glob(split+'-*/extrapolated-n3-p3.json'))))
 transfer_counts={}
 transfer_path=BASE/'transfer-protocol.json'
 if transfer_path.exists():
  transfer_spec=json.loads(transfer_path.read_text())
  for split in ['transfer_development','transfer_calibration','transfer_test']:
   transfer_counts[split]=dict(expected=len(transfer_spec[split]),references=len(list((BASE/'fields').glob(split+'-*.npz'))),prefix_fits=len(list((BASE/'inverse').glob(split+'-*/extrapolated-n3-p3.json'))),postprobe_fits=sum('uncertainty' not in p.name for p in (BASE/'inverse').glob(split+'-*/extrapolated-n3-p4-*.npz')))
 jobs={}
 for path in (BASE/'jobs').glob('*.json'):
  state=json.loads(path.read_text()).get('state')
  if state:jobs[state]=jobs.get(state,0)+1
 units=subprocess.check_output(['systemctl','--user','list-units','position-ood-v09-*','--state=running','--no-legend','--plain'],text=True).splitlines()
 record=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hours_remaining=(deadline-time.time())/3600,cohorts=counts,transfer=transfer_counts,jobs=jobs,neural_bank=len(list((BASE/'neural-bank/n2').glob('*.npz'))),neural_fine_bank=len(list((BASE/'neural-bank/extrapolated-n3').glob('*.npz'))),method_sealed=(BASE/'method-seal.json').exists(),running_units=[line.split()[0] for line in units if line.strip()])
 tmp=BASE/'progress.tmp';tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(BASE/'progress.json');time.sleep(30)
