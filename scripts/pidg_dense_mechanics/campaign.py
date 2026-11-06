"""Resumable scene worker: matched budgets, validation-only selection, sealed tests."""
from pathlib import Path
import argparse,json,subprocess,time,hashlib,sys
ROOT=Path(__file__).resolve().parents[2];PY=Path.home()/'.local/share/physgauss-external/pidg-official-v01/env/bin/python';RUN=ROOT/'runs/pidg-dense-mechanics-v01';REPORT=ROOT/'reports/pidg-dense-mechanics-v01'
def read(p):return json.loads(p.read_text())
def completed(p):return p.exists() and read(p).get('status')=='training_complete'
def command(args,log):
 with log.open('w') as f:subprocess.run([str(PY),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();out=RUN/a.scene;out.mkdir(parents=True,exist_ok=True);report=REPORT/a.scene;report.mkdir(exist_ok=True);base=ROOT/'runs/pidg-comparison-v01/full-pidg' if a.scene=='balls_reaction' else out/'native-base';flow=a.scene!='rubber_duck';status={'scene':a.scene,'gpu':a.gpu,'state':'waiting_for_base','started':time.time()}
 def state(name,**kw):status.update(state=name,updated=time.time(),**kw);(out/'campaign.json').write_text(json.dumps(status,indent=2));print(name,kw,flush=True)
 while not completed(base/'run.json'):
  if (base/'run.json').exists() and read(base/'run.json').get('status')=='failed':raise RuntimeError('Base failed; see native-base log')
  state('waiting_for_base');time.sleep(20)
 candidates=[]
 for label,weight in [('native-refined',0.),('dense-1e-5',1e-5),('dense-1e-4',1e-4),('dense-1e-3',1e-3)]:
  run=out/label;state('training',arm=label)
  if not completed(run/'run.json'):
   args=['scripts/pidg_comparison/run.py','--gpu',a.gpu,'--scene',a.scene,'--steps',5000,'--warm-from',base,'--output',run,'--dense-weight',weight]
   if flow:args+=['--flow']
   command(args,out/(label+'.log'))
  for width in [320,640]:
   path=run/f'validation-render-{width}/metrics.json';state('validation',arm=label,width=width)
   if not path.exists():command(['scripts/pidg_comparison/evaluate.py','--gpu',a.gpu,'--run',run,'--split','validation','--render-width',width],out/f'{label}-validation-{width}.log')
  metric=read(run/'validation-render-320/metrics.json');candidates.append({'name':label,'weight':weight,'run':str(run),'validation':metric['metrics'],'validation640':read(run/'validation-render-640/metrics.json')['metrics']});(report/'validation.json').write_text(json.dumps(candidates,indent=2))
 positive=[c for c in candidates if c['weight']>0];selected=max(positive,key=lambda c:c['validation']['foreground_psnr_db']);native=candidates[0]
 hashes={}
 for c in [native,selected]:
  run=Path(c['run'])
  for f in sorted((run/'model').rglob('*')):
   if f.is_file():hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
  if (run/'dense-mechanics.pt').exists():hashes[str((run/'dense-mechanics.pt').relative_to(ROOT))]=hashlib.sha256((run/'dense-mechanics.pt').read_bytes()).hexdigest()
 seal={'scene':a.scene,'selected':selected['name'],'test_runs':[native['run'],selected['run']],'criterion':'Highest 320-width validation foreground PSNR among positive physics weights; all candidate scores retained','candidates':candidates,'checkpoint_sha256':hashes,'sealed_at':time.time(),'test_images_used_for_selection':False};(out/'selection.json').write_text(json.dumps(seal,indent=2));(report/'selection.json').write_text(json.dumps(seal,indent=2));state('sealed',selected=selected['name'])
 for c in [native,selected]:
  for width in [320,640,1600]:
   run=Path(c['run']);state('test',arm=c['name'],width=width)
   if not (run/f'test-render-{width}/metrics.json').exists():command(['scripts/pidg_comparison/evaluate.py','--gpu',a.gpu,'--run',run,'--split','test','--render-width',width],out/f'{c["name"]}-test-{width}.log')
 results={'scene':a.scene,'completed':True,'selection':seal,'metrics':{c['name']:{str(w):read(Path(c['run'])/f'test-render-{w}/metrics.json') for w in [320,640,1600]} for c in [native,selected]},'limits':'Dense PIDG renderer retained, augmented with existing frozen material law and inferred Gaussian load fields through a soft momentum constraint. Not the coarse NeuralFEM adapter; no solved-contact, SI-force or calibrated uncertainty claim. Held-frame interpolation, not changed-action forecasting.'};(report/'results.json').write_text(json.dumps(results,indent=2));state('complete',finished=time.time())
if __name__=='__main__':main()
