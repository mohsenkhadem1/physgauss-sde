"""Frozen input-only PhysTwin width fitting and conservative prefix selection.
No evaluator import, manual targets, or future object positions are used.
"""
import argparse,signal,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.benchmark_improvement_v14.common import ROOT,BASE,REPORT,read,write,sha,remaining
from scripts.benchmark_improvement_v14.provenance import snapshot_sources
from scripts.prefix_ridge_switch_v13.inputs import load_payload
from scripts.benchmark_improvement_v14.phystwin_attachment import fit,predict
from scripts.contact_benchmark_v12.object_frame import adapt,_observed_prediction

NAMES=('double_lift_sloth','double_lift_zebra','double_stretch_sloth','double_stretch_zebra','single_lift_dinosor','single_lift_sloth','single_lift_zebra','single_push_sloth')
MENU=[('none',0.)]+[(m,s) for m in ('pose','residual') for s in (.25,.5,1.)]
SETTINGS=dict(field=True,contact=True,nonconvex=True,materials=6,iterations=100,frames=36,width_bounds_mm=[7.5,30.],initial_width_mm=15.,fit_budget_seconds=1800,rollout_budget_seconds=1200)

def declared(name):
 if name not in NAMES:raise ValueError('Undeclared PhysTwin case')
 cases=[c for c in read(REPORT/'protocol.json')['historical_cases'] if c['dataset']=='PhysTwin']
 if {c['name'] for c in cases}!=set(NAMES) or len(cases)!=8:raise ValueError('Changed exact eight-case cohort')
 return next(c for c in cases if c['name']==name)

def strict(data,name=None):
 if 'evaluation' in data or data.get('dataset')!='PhysTwin' or data.get('name') not in NAMES:raise ValueError('Not declared input-only PhysTwin')
 if name is not None and data['name']!=name:raise ValueError('Payload case mismatch')
 end=int(data['train_end'])
 if len(data['observed'])!=end or len(data['observed_mask'])!=end or not 3<=end<=len(data['times']):raise ValueError('Invalid observation prefix')
 if not np.isfinite(data['nodes']).all() or not np.isfinite(data['controls']).all():raise ValueError('Invalid initial geometry/commands')
 return end

def freeze():
 p=REPORT/'phystwin-freeze.json';record=read(p)
 if record['protocol_sha256']!=sha(REPORT/'protocol.json'):raise ValueError('Protocol changed')
 for path,digest in record['source_sha256'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Frozen source changed: '+path)
 required={str(Path(__file__).relative_to(ROOT)),'scripts/benchmark_improvement_v14/phystwin_attachment.py','scripts/external_benchmark/fem.py','scripts/contact_benchmark_v12/object_frame.py'}
 if not required.issubset(record['source_sha256']):raise ValueError('Missing physics/selector freeze coverage')
 original_required={str((Path('runs/contact-benchmark-v12/base/validation')/name/file)) for name in NAMES for file in ('prediction.npz','prediction.json','complete.json','provenance.json','fit.json','best.npz','progress.json')}
 if set(record.get('original_short_sha256',{}))!=original_required:raise ValueError('Original SHORT baseline freeze incomplete')
 for path,digest in record['original_short_sha256'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Original SHORT baseline changed: '+path)
 return sha(p)

def directory(name,tag):
 if tag not in ('full','short'):raise ValueError('Unknown fit tag')
 return BASE/'phystwin-attachment'/tag/name

def contract(name,tag):
 case=declared(name);sources=snapshot_sources(case);folder=ROOT/'runs/prefix-ridge-switch-v13/inputs'/name
 paths=[folder/(tag+'.npz'),folder/(tag+'.json')]
 return dict(name=name,tag=tag,settings=SETTINGS,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},snapshot_sources=sources,freeze_sha256=freeze(),protocol_sha256=sha(REPORT/'protocol.json'))

def validate_fit(name,tag):
 expected=contract(name,tag);out=directory(name,tag);record=read(out/'complete.json')
 if record['contract']!=expected:raise ValueError('Attachment cache contract changed')
 for key in ('prediction.npz','prediction.json','fit.json','best.npz'):
  if record['artifact_sha256'].get(key)!=sha(out/key):raise ValueError('Attachment fit artifact changed: '+key)
 return record

def run(name,tag,device='cuda:0'):
 if device not in ('cpu','cuda:0','cuda:1','cuda:2'):raise ValueError('Only authorized CPU/GPU0–2 devices')
 remaining();expected=contract(name,tag);out=directory(name,tag)
 if (out/'complete.json').exists():return validate_fit(name,tag)
 if out.exists():raise ValueError('Incomplete fit directory exists; do not silently restart/overwrite')
 out.mkdir(parents=True);write(out/'provenance.json',expected)
 data=load_payload(ROOT/'runs/prefix-ridge-switch-v13/inputs'/name/(tag+'.npz'));end=strict(data,name)
 # SHORT prediction covers the full observed prefix, never the hidden suffix.
 if tag=='short':
  full_end=int(read(ROOT/'runs/prefix-ridge-switch-v13/inputs'/name/'full.json')['train_end'])
  if end!=max(3,int(.7*full_end)):raise ValueError('SHORT cutoff changed')
  for key in ('times','controls','controller_targets','attachment_active'):data[key]=np.asarray(data[key])[:full_end].copy()
 started=time.monotonic()
 def check():
  try:remaining()
  except RuntimeError as e:raise TimeoutError(str(e)) from e
 def checkpoint(theta,best,selected):
  pending=out/'best.pending.npz';np.savez_compressed(pending,theta=theta,selected_frames=selected,nodes=data['nodes']);pending.replace(out/'best.npz')
  write(out/'progress.json',dict(fit_end=end,best=best,contract=expected))
 try:
  theta,fitting=fit(data,device=device,iterations=100,frames=36,budget_seconds=min(1800,remaining()),check_callback=check,checkpoint_callback=checkpoint)
 except RuntimeError as error:
  remaining();theta=np.empty(0);fitting=dict(fit_end=end,optimizer=dict(success=False,message=str(error)),state='failed_no_valid_checkpoint')
  np.savez_compressed(out/'best.npz',theta=theta,nodes=data['nodes'],selected_frames=np.unique(np.linspace(0,end-1,36,dtype=int)))
 write(out/'fit.json',fitting);remaining()
 if len(theta):pred,meta=predict(data,theta,device=device,budget_seconds=min(1200,remaining()),check_callback=check)
 else:
  pred=np.full((len(data['times']),len(data['nodes']),3),np.nan);meta=dict(failed_frames=list(range(len(pred))),reason='No valid fitted checkpoint')
 if contract(name,tag)!=expected:raise ValueError('Inputs/source changed during fit')
 temporary=out/'prediction.pending.npz';np.savez_compressed(temporary,predicted=pred,theta=theta,nodes=data['nodes']);temporary.replace(out/'prediction.npz');write(out/'prediction.json',meta)
 record=dict(name=name,tag=tag,device=device,fit_end=end,contract=expected,elapsed_seconds=time.monotonic()-started,failed_frames=meta['failed_frames'],optimizer_converged=fitting['optimizer']['success'],artifact_sha256={p:sha(out/p) for p in ('prediction.npz','prediction.json','fit.json','best.npz')})
 write(out/'complete.json',record);return record

def prefix_score(data,prediction,cut,end):
 if len(prediction)<end or not np.isfinite(prediction[cut:end]).all():return float('inf')
 observed=_observed_prediction(data,prediction[cut:end]);truth=np.asarray(data['observed'][cut:end]);valid=np.asarray(data['observed_mask'][cut:end],bool)&np.isfinite(truth).all(2)
 return float(np.sqrt(np.mean((observed-truth)[valid]**2))) if valid.any() and np.isfinite(observed[valid]).all() else float('inf')

def select(data,short_data,full_anchor,short_anchor,original_full,original_short,original_config):
 end=strict(data);cut=strict(short_data,data['name'])
 if cut!=max(3,int(.7*end)):raise ValueError('Independent SHORT cutoff changed')
 short_data=dict(short_data)
 for key in ('times','controls','controller_targets','attachment_active'):
  if len(short_data[key])<end or not np.array_equal(np.asarray(data[key])[:end],np.asarray(short_data[key])[:end],equal_nan=True):raise ValueError('SHORT commands/window mismatch')
  short_data[key]=np.asarray(short_data[key])[:end].copy()
 if len(short_anchor)<end or len(original_short)<end:raise ValueError('Missing SHORT validation forecasts')
 short_anchor=np.asarray(short_anchor)[:end];original_short=np.asarray(original_short)[:end]
 for key in ('nodes','observed_initial','observation_matrix'):
  if not np.array_equal(data[key],short_data[key]):raise ValueError('SHORT/full geometry or marker mapping mismatch')
 for key in ('observed','observed_mask'):
  if not np.array_equal(np.asarray(data[key])[:cut],short_data[key],equal_nan=True):raise ValueError('SHORT observation mismatch')
 if np.asarray(original_full).shape!=(len(data['times']),len(data['nodes']),3):raise ValueError('Original forecast shape mismatch')
 if np.asarray(full_anchor).shape!=np.asarray(original_full).shape:raise ValueError('Width anchor shape mismatch')
 config=(original_config['method'],float(original_config.get('strength',0.)))
 if config not in MENU:raise ValueError('Unknown original adapter config')
 unavailable=bool(original_config.get('selection_unavailable'))
 if not unavailable:
  if original_config.get('independent_validation') is not True:raise ValueError('Original adapter config not independently selected')
  if original_config.get('validation_cut')!=cut or original_config.get('fit_end')!=end:raise ValueError('Original adapter prefix metadata mismatch')
 try:
  if unavailable:oldscore=float('inf')
  else:
   oldshort,_=adapt(short_data,original_short,method=config[0],strength=config[1]);oldscore=prefix_score(data,oldshort,cut,end)
 except (ValueError,RuntimeError,np.linalg.LinAlgError):oldscore=float('inf')
 if not unavailable and np.isfinite(oldscore):
  matched=[r for r in original_config.get('validation',[]) if (r['method'],float(r['strength']))==config]
  if len(matched)!=1 or not np.isclose(oldscore,matched[0]['validation_rmse_m'],atol=1e-10,rtol=1e-8):raise ValueError('Original SHORT score differs from pinned validation metadata')
 rows=[]
 for method,strength in MENU:
  error=None
  try:
   p,_=adapt(short_data,short_anchor,method=method,strength=strength);value=prefix_score(data,p,cut,end)
  except (ValueError,RuntimeError,np.linalg.LinAlgError) as failure:value=float('inf');error=str(failure)
  rows.append(dict(method=method,strength=strength,validation_rmse_m=value if np.isfinite(value) else None,error=error))
 def value(row):return float('inf') if row['validation_rmse_m'] is None else row['validation_rmse_m']
 chosen=min(rows,key=value)
 if not value(chosen)<.98*value(rows[0]):chosen=rows[0]
 validation_passed=np.isfinite(oldscore) and np.isfinite(value(chosen)) and value(chosen)<.98*oldscore
 switched=bool(validation_passed);fallback_reason=None
 if switched:
  try:
   output,adapter=adapt(data,full_anchor,method=chosen['method'],strength=chosen['strength'])
   if not np.isfinite(output).all():raise ValueError('Selected full-prefix candidate is nonfinite')
  except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
   switched=False;fallback_reason='Candidate numerical failure: '+str(error)
 if not switched:
  output=np.asarray(original_full).copy();adapter=dict(retained_original=True)
  if fallback_reason is None:fallback_reason='Original comparison unavailable' if not np.isfinite(oldscore) else 'Independent validation improvement insufficient'
 return output,dict(switched=bool(switched),validation_passed=bool(validation_passed),fallback_reason=fallback_reason,candidate=chosen,original_config=dict(method=config[0],strength=config[1]),original_validation_rmse_m=oldscore if np.isfinite(oldscore) else None,validation=rows,validation_cut=cut,fit_end=end,required_relative_gain=.02,selection='Independent SHORT refit; unchanged seven-adapter menu; >2% gain over original selected adapter AND finite complete candidate required; otherwise exact original fallback',adapter=adapter,scope='Raw empirical hybrid; strict FEM anchor saved separately; final adapter not guaranteed geometrically admissible',fallback_byte_exact=not switched and np.array_equal(output,original_full,equal_nan=True))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True,choices=NAMES);p.add_argument('--tag',required=True,choices=['short','full']);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 # Parent queue also bounds each worker. This alarm protects a standalone CLI.
 def alarm(signum,frame):raise TimeoutError('Bounded attachment CLI runtime expired')
 signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,min(remaining(),3010.))
 try:run(a.name,a.tag,a.device)
 finally:signal.setitimer(signal.ITIMER_REAL,0)
