"""Independent read-only audit of fixed v16 oracle outputs; no solver imports."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';OUT=ROOT/'runs/compositional-neural-mechanics-v16/oracle-reserved-forward'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rest_points(q,g):
 x,y,z=q.T;L,H,D=g['dimensions'];s=1+g['taper']*(2*x-1)
 return np.column_stack((L*x,H*(y-.5)*s,D*(z-.5)*s+4*g['arch']*x*(1-x)))
def nonrigid(rest,u):
 a=rest-rest.mean(0);b=rest+u;b-=b.mean(0)
 left,_,right=np.linalg.svd(a.T@b);fix=np.eye(3);fix[-1,-1]=np.sign(np.linalg.det(left@right));rotation=left@fix@right
 return b@rotation.T-a

def metrics(p,r,rest):
 diff=p['displacement']-r['displacement'];nr=nonrigid(rest,r['displacement']);nd=nonrigid(rest,p['displacement'])-nr
 rms=lambda x:float(np.sqrt(np.sum(x*x)/x.size))
 norm=np.sqrt(np.sum(diff*diff,axis=1))
 return dict(displacement_coordinate_rmse_m=rms(diff),displacement_point_error_mean_m=float(np.sum(norm)/len(norm)),displacement_point_error_max_m=float(np.max(norm)),canonical_nonrigid_coordinate_rmse_m=rms(nd),force_coordinate_rmse_N=rms(p['wrenches'][:,:3]-r['wrenches'][:,:3]),torque_coordinate_rmse_Nm=rms(p['wrenches'][:,3:]-r['wrenches'][:,3:]),reference_nonrigid_coordinate_rms_m=rms(nr))

def bootstrap(model,baseline):
 delta=np.asarray(baseline)-model;sample=np.random.default_rng(926541).integers(len(delta),size=(10000,len(delta)));values=np.sum(delta[sample],axis=1)/len(delta)
 return dict(objects=len(delta),mean_error_reduction=float(delta.sum()/len(delta)),percentile95_interval=np.quantile(values,[.025,.975]).tolist(),wins=int(np.sum(delta>0)),ties=int(np.sum(delta==0)),losses=int(np.sum(delta<0)),bootstrap_unit='object',seed=926541,resamples=10000)

class Audit:
 def __init__(self):self.checks=0;self.evidence={}
 def check(self,ok,message):
  self.checks+=1
  if not ok:raise AssertionError(message)
 def bind(self,p):self.evidence[str(Path(p).relative_to(ROOT))]=sha(p)
 def equal(self,a,b,message):self.check(np.allclose(a,b,rtol=1e-9,atol=1e-12),message)
 def run(self,live=False):
  freeze_path=R/'oracle-method-freeze.json';f=read(freeze_path);self.bind(freeze_path)
  for path,h in f['source_sha256'].items():self.check(sha(ROOT/path)==h,'source '+path)
  for path,h in f['qualification_evidence'].items():self.check(sha(R/path)==h,'qualification '+path);self.bind(R/path)
  self.check(f['prediction_n']==4 and f['reference_n']==8 and f['order']==3,'frozen mesh settings')
  kinds=['reference',*f['methods']];self.check(kinds==['reference','matched_analytic','neo_hookean','shared_laws','custom_direct','custom_direct_long'],'fixed methods')
  for item in f['checkpoints'].values():self.check(sha(ROOT/item['path'])==item['sha256'],'checkpoint hash');self.bind(ROOT/item['path'])
  manifest=read(OUT/'run-manifest.json');records_file=OUT/'reserved-records.json';records_doc=read(records_file);records=records_doc['records']
  self.check(manifest['freeze_sha256']==sha(freeze_path)==records_doc['freeze_sha256'],'runtime freeze')
  self.check(manifest['records_sha256']==sha(records_file),'record commitment');self.bind(records_file);self.bind(OUT/'run-manifest.json')
  self.check(len(records)==12 and len({r['scene']['name']for r in records})==12,'12 unique objects')
  self.check(all(sum(r['role']==role for r in records)==6 for role in ('reserved_composition','reserved_geometry')),'6+6 cohorts')
  self.check(all(len(r['actions'])==4 and len({a['name']for a in r['actions']})==4 for r in records),'48commands')
  self.check(len({json.dumps(r['scene']['geometry'],sort_keys=True)for r in records if r['role']=='reserved_geometry'})==2,'two newshapeinstances')
  expected={(r['scene']['name'],a['name'],k):(r,a)for r in records for a in r['actions']for k in kinds}
  available={}
  for record in records:
   folder=OUT/record['scene']['name'];path=folder/'complete.json'
   if not path.exists():path=folder/'progress.json'
   if not path.exists():continue
   for row in read(path)['rows']:
    key=(row['scene_id'],row['action'],row['method'])
    self.check(key in expected and key not in available,'unique declared output '+str(key));self.check(row['cohort']==record['role'],'row cohort');available[key]=row
  if live:return dict(status='pending',checks=self.checks,expected=288,present=len(available),complete=sum(x['state']=='complete'for x in available.values()),failed=sum(x['state']!='complete'for x in available.values()),pending=288-len(available))
  import torch
  checkpoints={k:torch.load(ROOT/f['checkpoints'][k]['path'],map_location='cpu',weights_only=False)for k in ('direct','direct_long')}
  a,b=checkpoints.values();self.check(a['model_config']==b['model_config'] and a['training_ids']==b['training_ids'],'baseline architecture/training')
  self.check(a['feature_contract']==b['feature_contract'],'baseline feature contracts')
  for key in a['state_dict']:
   if key.startswith('normalization.'):self.check(torch.equal(a['state_dict'][key],b['state_dict'][key]),'normalization '+key)
  for k,cp in checkpoints.items():
   self.check(cp['metadata']['steps']==(6000 if k=='direct'else 30000) and cp['metadata']['seed']==926701,'baseline budget')
   for law in ('bulk','interface'):self.check(cp['feature_contract'][law+'_law_fingerprint']['checkpoint_sha256']==f['checkpoints'][law]['sha256'],'fingerprint law')
   self.equal(cp['feature_contract']['bulk_law_fingerprint']['probes_F'],f['probes']['bulk_F'],'bulk probes')
   self.equal(cp['feature_contract']['interface_law_fingerprint']['probes_delta_m'],f['probes']['interface_delta_m'],'interface probes')
  result_path=R/'oracle-reserved-forward-results.json';result=read(result_path);self.bind(result_path);self.check(result['freeze_sha256']==sha(freeze_path),'result freeze')
  rows={(x['scene_id'],x['action'],x['method']):x for x in result['rows']}
  self.check(len(rows)==len(result['rows'])==288 and set(rows)==set(expected),'all288fixedrows')
  q=np.array(f['material_query']);arrays={};scores={}
  for key,row in rows.items():
   record,_=expected[key];self.check(row['cohort']==record['role'],'finalcohort');self.check(row['state']in ('complete','failed'),'terminalstate')
   if key in available:self.check(row['state']==available[key]['state'],'retainedterminalstate')
   else:self.check(row['state']=='failed','missing output retained failure')
   if row['state']!='complete':self.check('error'in row,'failure reason');continue
   path=ROOT/row['artifact'];self.check(sha(path)==row['sha256'],'artifact '+str(key));self.bind(path)
   with np.load(path,allow_pickle=False)as d:
    self.check(set(d.files)=={'material_query','displacement','wrenches'},'artifactfields');self.check(np.array_equal(d['material_query'],q),'querygrid');value={k:d[k].copy()for k in ('displacement','wrenches')}
   self.check(value['displacement'].shape==(len(q),3)and value['wrenches'].shape==(2,6)and all(np.isfinite(x).all()for x in value.values()),'finite arrays');arrays[key]=value
   if key[2]not in ('custom_direct','custom_direct_long'):self.check(row['diagnostics']['n']==(8 if key[2]=='reference'else 4),'actualmesh')
  for key,row in rows.items():
   if key[2]=='reference':continue
   ref=(key[0],key[1],'reference')
   if key not in arrays or ref not in arrays:self.check('scores'not in row,'no invalidscores');continue
   value=metrics(arrays[key],arrays[ref],rest_points(q,expected[key][0]['scene']['geometry']));scores[key]=value
   self.check(set(value)==set(row['scores']),'metricfields')
   for m,v in value.items():self.equal(v,row['scores'][m],'metric '+str(key)+m)
  self.check(result['failed_predictions']==sum(x['state']!='complete'for x in rows.values()),'failurecount')
  means={}
  summary_keys=[(x['cohort'],x['method'])for x in result['summaries']]
  self.check(len(summary_keys)==10 and set(summary_keys)=={(role,k)for role in ('reserved_composition','reserved_geometry')for k in f['methods']},'exact ten summaries')
  for summary in result['summaries']:
   role=summary['cohort'];kind=summary['method'];names=[x['scene']['name']for x in records if x['role']==role];keys=[k for k,(r,_)in expected.items()if r['role']==role and k[2]==kind];scored=[k for k in keys if k in scores]
   self.check(len(keys)==24 and summary['expected_commands']==24 and summary['objects']==6 and summary['scored_commands']==len(scored),'summarycounts');self.check(summary['complete']==(len(scored)==24),'fullcohort completion')
   if len(scored)==24:
    means[(role,kind)]={m:np.array([np.mean([scores[k][m]for k in keys if k[0]==name])for name in names])for m in scores[keys[0]]}
    for m,v in means[(role,kind)].items():self.equal(v.mean(),summary['object_mean_scores'][m],'summary '+m)
   else:self.check('object_mean_scores'not in summary,'no selected means')
  expected_pairs={(role,k,m)for role in ('reserved_composition','reserved_geometry')for k in f['methods']if k!='shared_laws' and (role,k)in means and (role,'shared_laws')in means for m in ('displacement_coordinate_rmse_m','force_coordinate_rmse_N','torque_coordinate_rmse_Nm')}
  self.check(len(result['paired_object_comparisons'])==len(expected_pairs),'pairedcount')
  for item in result['paired_object_comparisons']:
   key=(item['cohort'],item['baseline'],item['metric']);self.check(key in expected_pairs,'pairedidentity');expected_pairs.remove(key)
   calc=bootstrap(means[(key[0],'shared_laws')][key[2]],means[(key[0],key[1])][key[2]])
   for field,v in calc.items():
    if isinstance(v,str):self.check(item[field]==v,'bootstrapunit')
    else:self.equal(item[field],v,'bootstrap '+field)
  old=read(R/'prior-preservation.json');self.bind(R/'prior-preservation.json')
  for p,h in old['sha256'].items():self.check(sha(ROOT/p)==h,'prior preservation '+p)
  return dict(status='passed',passed=True,checks=self.checks,expected_rows=288,failed_predictions=result['failed_predictions'],scored_method_rows=len(scores),prior_files_preserved=len(old['sha256']),evidence_sha256=self.evidence,auditor_sha256=sha(Path(__file__)),scope='Oracle forward only; n8 fixed numerical targets not individually mesh-qualified;6composition and6draws conditional on2newshapes; no inverse/generalcontactclaim')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true');args=parser.parse_args();audit=Audit()
 try:out=audit.run(args.live)
 except Exception as error:out=dict(status='failed',passed=False,checks=audit.checks,error=f'{type(error).__name__}: {error}',evidence_sha256=audit.evidence)
 if not args.live:(R/'oracle-final-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out if args.live or out['status']!='passed'else{k:v for k,v in out.items()if k!='evidence_sha256'},indent=2))
 if out['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
