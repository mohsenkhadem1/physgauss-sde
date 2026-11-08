"""Read-only independent saved-array audit; no simulator/scorer imports."""
import json,hashlib,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-dynamics-v17';B=ROOT/'runs/compositional-dynamics-v17/learned-dynamic-pilot'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 errors=[];count=0;evidence={}
 def check(ok,label):
  nonlocal count
  count+=1
  if not ok:errors.append(label)
 def read(p):evidence[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
 p=read(R/'learned-dynamic-pilot-plan.json');r=read(R/'learned-dynamic-pilot-results.json');a=read(R/'learned-dynamic-pilot-analysis.json')
 for group in ['source_sha256','checkpoint_sha256']:
  for f,h in p[group].items():check(sha(ROOT/f)==h,'pin '+f)
 for group in ['input_sha256','source_sha256']:
  for f,h in a[group].items():check(sha(ROOT/f)==h,'analysis pin '+f)
 check(r['terminal'] and len(r['rows'])==8 and not r['missing'],'terminal eight')
 rows={x['task']['id']:x for x in r['rows']};check(set(rows)=={x['id']for x in p['tasks']},'exact census')
 check(sum(x['completed']for x in rows.values())==4,'four completed')
 summaries=[]
 for pair in a['pairs']:
  rr=[rows[k]for k in pair['ids']];aa=[]
  for k,row in zip(pair['ids'],rr):
   f=B/k/'trajectory.npz';check(sha(f)==row['trajectory_sha256'],'trajectory '+k);evidence[str(f.relative_to(ROOT))]=sha(f)
   with np.load(f)as z:aa.append({key:z[key]for key in z.files})
  t,q=aa;n=min(len(t['times_s']),len(q['times_s']));w=t['mass_row_sums']/t['mass_row_sums'].sum();world=[];shape=[]
  check(np.array_equal(t['times_s'][:n],q['times_s'][:n]),'timestamps')
  for i in range(n):
   x=t['rest_nodes']+t['u'][i];y=q['rest_nodes']+q['u'][i];world.append(np.sqrt(np.sum(w[:,None]*(x-y)**2)))
   xc=x-np.sum(w[:,None]*x,axis=0);yc=y-np.sum(w[:,None]*y,axis=0);u,s,vt=np.linalg.svd(yc.T@(w[:,None]*xc));D=np.eye(3);D[-1,-1]=np.linalg.det(u@vt);rot=u@D@vt;shape.append(np.sqrt(np.sum(w[:,None]*(yc@rot-xc)**2)))
   for key,value in [('world_point_rms_m',world[-1]),('shape_point_rms_m',shape[-1])]:check(np.isclose(value,pair['metrics']['rows'][i][key],rtol=1e-8,atol=1e-12),key)
  m=pair['metrics'];complete=all(z['completed']for z in rr);expected=round(p['settings']['duration'][pair['case']]/p['settings']['sample_interval_s'])+1
  check(pair['complete']==complete and m['complete']==complete,'completion flags')
  check(m['expected_frames']==expected and np.isclose(m['frame_coverage'],n/expected),'full denominator')
  for key,values in [('world_point_rms_m',world),('shape_point_rms_m',shape)]:check(np.isclose(m[key],np.sqrt(np.mean(np.square(values))),atol=1e-12),'aggregate '+key)
  check(m['physical_reference_qualified']is False,'no qualification')
  summaries.append(dict(family=pair['family'],case=pair['case'],complete=complete,frames=n,expected=expected,world_point_rms_mm=1000*np.sqrt(np.mean(np.square(world))),shape_point_rms_mm=1000*np.sqrt(np.mean(np.square(shape)))))
 out=dict(passed=not errors,errors=errors,checks=count,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),evidence_sha256=evidence,pairs=summaries,scope='Two complete drop pairs; two failed release pairs retain common-prefix diagnostics only. Same unqualified discretization and known latents; no independent physical truth, inference or native superiority claim.')
 (R/'learned-dynamic-pilot-final-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2)[:2500])
if __name__=='__main__':main()
