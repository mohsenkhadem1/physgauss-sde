"""Additive durable orchestration; no changes to pinned numerical sources."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from scripts.compositional_neural_mechanics_reporting.legacy_dependency_audit import closure,audit
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/compositional-neural-mechanics-v16';BASE=ROOT/'runs/compositional-neural-mechanics-v16'
PYTHON=ROOT/'runs/phystwin-env/bin/python';STATUS=REPORT/'direct-supervisor-status.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(value):
 p=STATUS.with_suffix('.pending');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(STATUS)
def active(unit):return subprocess.run(['systemctl','--user','is-active','--quiet',unit],check=False).returncode==0
def main():
 own=sha(Path(__file__));extra_path=REPORT/'legacy-dependency-initial.json';initial=json.loads(extra_path.read_text());extra_sha=sha(extra_path)
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0,1,2')
 started=time.time();limit=started+10800
 while time.time()<limit:
  if sha(Path(__file__))!=own or sha(extra_path)!=extra_sha:raise ValueError('Orchestration provenance changed')
  if any(sha(ROOT/p)!=digest for p,digest in initial['inventory_sha256'].items()):raise ValueError('Supplemental full local dependency changed')
  progress=json.loads((BASE/'training-data-n4/progress.json').read_text());counts=progress['counts']
  if any(counts.get(k,0)>0 for k in ('failed','timeout','stage_timeout')):
   write(dict(state='dataset_failure',counts=counts,source_sha256=own));raise RuntimeError('Dataset failures retained; no automatic training')
  subprocess.run([str(PYTHON),'-m','scripts.compositional_neural_mechanics.train_direct_campaign','--prepare-features'],cwd=ROOT,env=env,check=True,timeout=600)
  features=json.loads((BASE/'direct-baseline-campaign/feature-progress.json').read_text())
  write(dict(state='preparing_features',counts=counts,cached_features=features['complete'],expected=348,dataset_active=active('compositional-v16-training-data'),source_sha256=own,legacy_inventory_sha256=extra_sha))
  print('dataset',counts,'features',features['complete'],'/348',flush=True)
  if counts.get('complete')==58 and features['complete']==348 and features['missing_or_failed']==0 and not active('compositional-v16-training-data'):
   audit('dataset-complete')
   unit='compositional-v16-direct-training'
   if active(unit):raise ValueError('Existing training unit; refusing duplicate launch')
   command=['systemd-run','--user','--unit='+unit,'--property=RuntimeMaxSec=1800','--property=WorkingDirectory='+str(ROOT),'--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','--setenv=CUDA_VISIBLE_DEVICES=0,1,2',str(PYTHON),'-m','scripts.compositional_neural_mechanics.train_direct_campaign','--execute-training']
   subprocess.run(command,check=True)
   write(dict(state='training_launched',unit=unit,source_sha256=own,legacy_inventory_sha256=extra_sha,cached_features=348,command=command))
   print('launched',unit,flush=True)
   while active(unit):time.sleep(30)
   result=REPORT/'direct-training-results.json'
   if not result.exists():raise RuntimeError('Training service stopped without complete report')
   audit('training-complete')
   write(dict(state='complete',source_sha256=own,result_sha256=sha(result),checkpoint_sha256=json.loads(result.read_text())['checkpoint_sha256']))
   return
  time.sleep(60)
 raise TimeoutError('Bounded orchestration window exceeded; no training launched')
if __name__=='__main__':main()
