import json,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-dynamics-v17';B=ROOT/'runs/compositional-dynamics-v17/fresh-cohort'
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 report=R/'fresh-cohort-reference-robustness-posthoc.json';d=json.loads(report.read_text());errors=[];count=0
 def ck(x,msg):
  nonlocal count
  count+=1
  if not x:errors.append(msg)
 def eq(x,y,msg):ck(np.allclose(x,y,rtol=1e-8,atol=1e-12),msg)
 from scripts.compositional_dynamics_v17_reporting.cohort_recovery_evaluation import committed
 binding=json.loads((R/'fresh-cohort-launch-bindings.json').read_text());protocol,commits,evidence=committed(binding)
 for group in ['source_sha256','evidence_sha256']:
  for name,v in d[group].items():ck(h(ROOT/name)==v,'hash '+name)
 expected={(f'world-{i}',ref)for i in range(2)for ref in ['n4_base','n6_base','n6_halfstep']};ck(len(d['rows'])==6 and {(r['world_id'],r['reference'])for r in d['rows']}==expected,'six fixed rows');ck(d['posthoc_secondary'] and d['fixed_role']=='forecast-80mm','posthoc scope')
 from scripts.compositional_dynamics_v17.prototype import face_grid
 dense=face_grid(17);dims=np.array(protocol['generator_settings']['dimensions_m']);values=[]
 for row in d['rows']:
  world=row['world_id'];meta=json.loads((B/'public'/world/'metadata.json').read_text());q=np.array(meta['material_queries']);weights=np.zeros(len(q))
  for axis in range(3):
   others=[i for i in range(3)if i!=axis]
   for side in [0,1]:
    ids=np.flatnonzero(q[:,axis]==side);uv=q[ids][:,others];factor=np.prod(np.where((uv==0)|(uv==1),.5,1.),axis=1);weights[ids]+=np.prod(dims[others])/16*factor
  weights/=weights.sum();ck(len(weights)==98 and np.all(weights>0),'98 area weights');indices=[np.where(np.all(dense==v,axis=1))[0][0]for v in q]
  folder={'n4_base':'n4-dt0.00002500','n6_base':'n6-dt0.00002500','n6_halfstep':'n6-dt0.00001250'}[row['reference']];path=B/'references'/world/'forecast-80mm'/folder/'trajectory.npz';status=json.loads(path.with_name('complete.json').read_text());ck(status['completed']and h(path)==status['trajectory_sha256'],'reference seal')
  with np.load(path)as z:t=z['times_s'][::10];truth=z['surface_frames'][::10].reshape(31,-1,3)[:,indices].astype(float)
  eq(t,np.arange(31)*.01,'clock');result={}
  for method in ['ours','native_phystwin']:
   c=next(c for c in commits if(c['world_id'],c['role'],c['method'])==(world,'forecast-80mm',method));ck(c['record']['completed'],'prediction completed')
   with np.load(c['array'])as z:x=z['positions_m'if method=='ours'else'tracked_positions_m'].astype(float);eq(z['times_s'],t,'prediction clock')
   worlderr=float(np.sqrt(np.mean(np.sum(weights[None,:,None]*(x-truth)**2,axis=(1,2)))));shape=[]
   for a,b in zip(x,truth):
    a=a-weights@a;b=b-weights@b;u,_,vt=np.linalg.svd(a.T@(weights[:,None]*b));corr=np.eye(3);corr[-1,-1]=np.linalg.det(u@vt);shape.append(np.sum(weights[:,None]*(a@u@corr@vt-b)**2))
   result[method]=dict(world_point_rms_m=worlderr,shape_point_rms_m=float(np.sqrt(np.mean(shape))))
   for key,val in result[method].items():eq(val,row['methods'][method][key],'raw '+key)
  for key in result['ours']:
   diff=result['ours'][key]-result['native_phystwin'][key];eq(diff,row['ours_minus_native_m'][key],'paired difference');ck(diff<0,'reported ordering')
  values.append(dict(world_id=world,reference=row['reference'],methods=result))
 out=dict(passed=not errors,checks=count,errors=errors,rows=values,evidence_sha256={str(report.relative_to(ROOT)):h(report),**d['evidence_sha256']},source_sha256={str(Path(__file__).relative_to(ROOT)):h(__file__)},scope='Post-hoc fixed two-world 80mm robustness only. Independent98point rest-area/proper-Kabsch raw metrics reproduce all6pairs. All levels favor ours in world and shape; absolute errors change. Original n4 primary and failed spatial qualification unchanged; no continuum guarantee.',created_epoch=time.time());(R/'fresh-cohort-reference-robustness-final-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'passed':out['passed'],'checks':count,'errors':errors}))
if __name__=='__main__':main()
