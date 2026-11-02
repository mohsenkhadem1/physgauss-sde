"""Read-only prediction-only audit. Never opens reference arrays or scores targets."""
import argparse,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'reports/compositional-dynamics-v17'
B=ROOT/'runs/compositional-dynamics-v17/fresh-cohort-appearance-inference'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def audit(final=False):
 errors=[];count=0;evidence={}
 def check(ok,msg):
  nonlocal count
  count+=1
  if not ok:errors.append(msg)
 def bind(p):evidence[str(Path(p).relative_to(ROOT))]=sha(p)
 planpath=R/'fresh-cohort-appearance-plan.json';p=read(planpath);bind(planpath)
 protocol=read(R/'fresh-cohort-protocol.json');bind(R/'fresh-cohort-protocol.json')
 for name in ['fresh-cohort-method-freeze.json','cohort-reporting-recovery-amendment.json','cohort-appearance-inference-amendment.json']:
  f=R/name;d=read(f);bind(f)
  for field in ['source_sha256','input_sha256']:
   for name,h in d.get(field,{}).items():check(sha(ROOT/name)==h,'Changed '+name)
 for field in ['source_sha256','input_sha256']:
  for name,h in p[field].items():check(sha(ROOT/name)==h,'Plan changed '+name)
 from scripts.compositional_dynamics_v17_reporting.cohort_recovery_evaluation import committed
 _,rows,seals=committed(p['bindings']);evidence.update(seals)
 cameras=read(ROOT/protocol['metrics']['camera_design'])['cameras']
 expected_queries={(f'world-{w}',role)for w in range(4)for role in ['forecast-40mm','forecast-80mm']}
 check(len(p['queries'])==8 and {(q['world_id'],q['role'])for q in p['queries']}==expected_queries,'query census')
 for q in p['queries']:
  world,role=q['world_id'],q['role'];physical=[r for r in rows if r['world_id']==world and r['role']==role]
  methods=[{'ours':'shared_q2','native_phystwin':'native'}[r['method']]for r in physical if r['record']['completed']]
  check(q['methods']==methods,'method availability '+world+role)
  check(q['checkpoint']==protocol['metrics']['appearance_checkpoint'],'checkpoint route')
  check(sha(ROOT/q['checkpoint'])==protocol['metrics']['appearance_checkpoint_sha256'],'checkpoint hash')
  check(q['cameras']==cameras,'camera design')
  check(q['geometry_dimensions_m']==protocol['generator_settings']['dimensions_m'] and q['q2_n']==3,'geometry')
  check(q['times_s']==[i*.01 for i in range(31)],'clock')
  for row in physical:
   key='ours_directory' if row['method']=='ours' else 'native_directory'
   check(ROOT/q[key]==row['array'].parent,'mapped prediction path')
 counts=None
 if final:
  cp=B/'prediction-commit.json';c=read(cp);bind(cp);seal=read(B/'inference-context-complete.json');bind(B/'inference-context-complete.json')
  check(seal['amendment_sha256']==sha(R/'cohort-appearance-inference-amendment.json') and seal['prediction_commit_sha256']==sha(cp),'inference seal')
  check(c['plan_sha256']==sha(planpath),'image plan seal')
  expected={(w,r,m,i,k['name'])for w,r in expected_queries for m in ['native','shared_q2'] for i in range(31)for k in cameras}
  keys=[(r['world_id'],r['query'],r['method'],r['frame'],r['camera'])for r in c['rows']]
  check(len(keys)==2480 and len(set(keys))==2480 and set(keys)==expected,'2480 exact slots')
  counts={'complete':0,'failed':0}
  for row in c['rows']:
   check(type(row['completed']) is bool and row['time_s']==row['frame']*.01,'slot type/time')
   counts['complete' if row['completed'] else 'failed']+=1
   if row['completed']:
    a=row['prediction'];path=ROOT/a['path'];check(path.is_relative_to(B),'prediction containment')
    check(sha(path)==a['sha256'],'image hash '+a['path'])
    check(all(a[k]==row[k] for k in ['method','frame','camera','time_s','role']),'image identity')
   else:check(bool(row['error']) and row['prediction'] is None,'failure retained')
  check(c['reference_images_read'] is False,'reference access declaration')
  check(len(c['query_evidence'])==8,'query diagnostic census')
  for q in c['query_evidence']:
   if q['completed']:
    check(q['gaussian_count']>0,'positive Gaussian count')
    if q['initial_parity_available']:check(q['initial_rgb_max_difference']<=1e-4,'initial appearance parity')
 out={'passed':not errors,'checks':count,'errors':errors,'phase':'final_prediction_seal' if final else 'plan','counts':counts,'target_arrays_accessed':False,'scoring_run':False,'evidence_sha256':evidence,'auditor_sha256':sha(__file__),'created_epoch':time.time()}
 dest=R/('fresh-cohort-image-prediction-final-independent-audit.json' if final else 'fresh-cohort-image-plan-independent-audit.json');dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='evidence_sha256'},indent=2));return out
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--final',action='store_true');args=a.parse_args();result=audit(args.final);raise SystemExit(0 if result['passed'] else 1)
