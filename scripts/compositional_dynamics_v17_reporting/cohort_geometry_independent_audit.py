"""Released fresh-cohort independent raw geometry/physical diagnostic audit."""
import json,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-dynamics-v17';B=ROOT/'runs/compositional-dynamics-v17/fresh-cohort'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 errors=[];checks=[];evidence={}
 def ck(v,s):checks.append(s);errors.extend([]if v else[s])
 def eq(a,b,s):ck(np.allclose(a,b,rtol=1e-8,atol=1e-12),s)
 def read(p):evidence[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
 release=read(R/'fresh-cohort-reference-release.json')
 from scripts.compositional_dynamics_v17_reporting.cohort_recovery_evaluation import committed
 bindings=read(R/'fresh-cohort-launch-bindings.json');protocol,commits,seals=committed(bindings);evidence.update(seals)
 report=read(R/'fresh-cohort-geometry-recovery-results.json');ck(len(report['rows'])==8 and {(r['world_id'],r['role'])for r in report['rows']}=={(f'world-{i}',role)for i in range(4)for role in ['forecast-40mm','forecast-80mm']},'eight exact cases')
 dims=np.array(protocol['generator_settings']['dimensions_m']);summaries={};paired=[];counts={'ours':0,'native_phystwin':0};failed=[]
 from scripts.compositional_dynamics_v17.prototype import face_grid
 dense=face_grid(17)
 for row in report['rows']:
  world,role=row['world_id'],row['role'];meta=read(B/'public'/world/'metadata.json');q=np.array(meta['material_queries']);rest=q*dims+[0,-.03,-.03];w=np.zeros(len(q));tri=[]
  for axis in range(3):
   other=[j for j in range(3)if j!=axis]
   for side in [0,1]:
    grid=np.empty((5,5),int)
    for i in range(5):
     for j in range(5):
      point=np.zeros(3);point[axis]=side;point[other]=[i/4,j/4];grid[i,j]=np.where(np.all(q==point,axis=1))[0][0]
    for i in range(4):
     for j in range(4):
      a,b,c,d=grid[i,j],grid[i+1,j],grid[i+1,j+1],grid[i,j+1];w[[a,b,c,d]]+=np.prod(dims[other])/64
      for ids in [[a,b,c],[a,c,d]]:
       v=q[ids];tri.append(ids if np.cross(v[1]-v[0],v[2]-v[0])[axis]*(2*side-1)>0 else ids[::-1])
  w/=w.sum();tri=np.array(tri)
  def align(x,y):
   cx=w@x;cy=w@y;u,_,vt=np.linalg.svd((x-cx).T@(w[:,None]*(y-cy)));d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt);return (x-cx)@u@d@vt+cy
  def rms(x):return np.sqrt(np.sum(w[:,None]*x*x))
  folder=B/'references'/world/role/'n4-dt0.00002500';status=read(folder/'complete.json');target=folder/'trajectory.npz'
  ck(status['completed']and status['world_id']==world and status['role']==role and status['n']==4 and status['dt_s']==25e-6,'reference identity');ck(sha(target)==status['trajectory_sha256'],'reference hash');evidence[str(target.relative_to(ROOT))]=sha(target)
  indices=[np.where(np.all(dense==v,axis=1))[0][0]for v in q]
  with np.load(target)as z:t=z['times_s'][::10];truth=z['surface_frames'][::10].reshape(31,-1,3)[:,indices].astype(float)
  eq(t,np.arange(31)*.01,'31 common times');floor=meta['forecast_public_setups'][role]['floor_z_m']
  def diag(x):
   center=np.einsum('n,tnc->tc',w,x);v=(x-center[:,None])[:,tri];vol=np.sum(np.einsum('tni,tni->tn',v[:,:,0],np.cross(v[:,:,1],v[:,:,2])),axis=1)/6;gap=x[:,:,2].min(1)-floor;contact=gap<=1e-6;hits=np.where(contact)[0];first=int(hits[0])if len(hits)else None;sep=apex=None
   if first is not None:
    off=np.where(~contact[first+1:])[0]
    if len(off):
     sep=first+1+int(off[0]);on=np.where(contact[sep+1:])[0];end=sep+1+int(on[0])if len(on)else len(t);k=sep+int(np.argmax(center[sep:end,2]));apex=k if sep<k<end-1 else None
   return dict(signed_closed_surface_volume_m3=vol,volume_over_rest=vol/np.prod(dims),surface_centroid_z_m=center[:,2],minimum_sampled_gap_m=gap,maximum_sampled_penetration_m=max(0.,-gap.min()),first_contact_s=None if first is None else t[first],first_separation_s=None if sep is None else t[sep],first_rebound_apex_s=None if apex is None else t[apex],first_rebound_apex_surface_centroid_z_m=None if apex is None else center[apex,2])
  result={}
  for method in ['ours','native_phystwin']:
   c=next(c for c in commits if (c['world_id'],c['role'],c['method'])==(world,role,method));saved=row['methods'][method];ck(saved['completed']==c['record']['completed'],'failure status')
   if not saved['completed']:failed.append([world,role,method]);ck('geometry'not in saved,'no failed metric');continue
   with np.load(c['array'])as z:pt=z['times_s'];pred=z['positions_m'if method=='ours'else'tracked_positions_m'].astype(float)
   eq(pt,t,'prediction clock');ck(pred.shape==(31,98,3)and np.isfinite(pred).all(),'finite shape');counts[method]+=1;g=saved['geometry'];ck(g['complete']and g['frame_coverage']==1,'full coverage');values=[]
   for i,(x,y)in enumerate(zip(pred,truth)):
    v=dict(world_point_rms_m=rms(x-y),shape_point_rms_m=rms(align(x,y)-y),surface_centroid_error_m=np.linalg.norm(w@(x-y)),reference_nonrigid_rms_m=rms(y-align(rest,y)),predicted_nonrigid_rms_m=rms(x-align(rest,x)));values.append(v)
    for k,val in v.items():eq(val,g['rows'][i][k],'frame '+k)
   result[method]={k:float(np.sqrt(np.mean([v[k]**2 for v in values])))for k in values[0]}
   for k,val in result[method].items():eq(val,g[k],'aggregate '+k)
   for label,x in [('reference',truth),('prediction',pred)]:
    for k,val in diag(x).items():
     stored=saved['physical_diagnostics'][label][k]
     if val is None:ck(stored is None,'censored '+k)
     else:eq(val,stored,'physical '+k)
  summaries[world+'/'+role]=result
  if len(result)==2:paired.append((world,role,result))
 ck(counts=={'ours':8,'native_phystwin':6},'8 vs6 complete');ck(len(paired)==6,'six conditional pairs');ck(failed==[['world-3','forecast-40mm','native_phystwin'],['world-3','forecast-80mm','native_phystwin']],'explicit world3 failures')
 for method,s in report['summary'].items():
  ck(s['primary_all_eight_mean']is None and s['paired_complete_forecasts']==6 and s['expected_forecasts']==8,'primary unavailable denominator')
  for k,v in s['conditional_paired_complete_case_mean'].items():eq(np.mean([r[method][k]for _,_,r in paired]),v,'paired mean')
 for item,(world,role,r) in zip(report['paired_world_error_difference_ours_minus_native_m'],paired):
  ck((item['world_id'],item['role'])==(world,role),'paired difference identity');eq(item['difference'],r['ours']['world_point_rms_m']-r['native_phystwin']['world_point_rms_m'],'paired difference')
 ck(len(report['paired_world_error_difference_ours_minus_native_m'])==6,'paired difference census')
 for field in ['evidence_sha256','source_sha256']:
  for name,h in report[field].items():ck(sha(ROOT/name)==h,'report evidence '+name)
 out=dict(passed=not errors,checks=len(checks),errors=errors,evidence_sha256=evidence,complete_forecasts=counts,failed_forecasts=failed,paired_forecasts=len(paired),paired_worlds=len({w for w,_,_ in paired}),conditional_means_mm={m:{k:v*1000 for k,v in s['conditional_paired_complete_case_mean'].items()}for m,s in report['summary'].items()},all_eight_primary_available=False,scope='Released raw31frame/98surface-point independent area weights, proper rigid alignment and sampled volume/contact diagnostics. Conditional six forecasts from three worlds only; native world3 failures retained. Finite-resolution references; no population superiority/continuum claim.',source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__)},created_epoch=time.time())
 (R/'fresh-cohort-geometry-final-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in ['evidence_sha256','source_sha256']},indent=2));return out
if __name__=='__main__':raise SystemExit(0 if main()['passed']else 1)
