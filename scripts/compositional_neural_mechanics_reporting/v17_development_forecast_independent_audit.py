"""Development-only released target audit; never reads fresh-cohort references."""
import json,hashlib,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];D=R/'reports/compositional-dynamics-v17';B=R/'runs/compositional-dynamics-v17'
def main():
 ev={};errors=[];checks=[]
 def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 def rd(p):ev[str(p.relative_to(R))]=h(p);return json.loads(p.read_text())
 def ck(x,s):checks.append(s);errors.extend([]if x else[s])
 def eq(x,y,s):ck(np.allclose(x,y,rtol=1e-8,atol=1e-12),s)
 r=rd(D/'matched-forecast-development-results.json');meta=rd(B/'calibration-public/metadata.json');q=np.array(meta['material_queries']);dims=np.array([.12,.06,.06]);rest=q*dims+[0,-.03,-.03];w=np.zeros(98);tri=[]
 # Independently assemble area quadrature by traversing face cells.
 for axis in range(3):
  o=[j for j in range(3)if j!=axis]
  for side in [0,1]:
   grid=np.empty((5,5),int)
   for i in range(5):
    for j in range(5):
     v=np.zeros(3);v[axis]=side;v[o]=[i/4,j/4];grid[i,j]=np.where(np.all(q==v,axis=1))[0][0]
   for i in range(4):
    for j in range(4):
     a,b,c,d=grid[i,j],grid[i+1,j],grid[i+1,j+1],grid[i,j+1];w[[a,b,c,d]]+=np.prod(dims[o])/16/4
     for t in [[a,b,c],[a,c,d]]:
      v=q[t];normal=np.cross(v[1]-v[0],v[2]-v[0]);tri.append(t if normal[axis]*(side*2-1)>0 else t[::-1])
 w/=w.sum();tri=np.array(tri)
 # Public17-grid surface order mirrors stored face order; locate98 material samples.
 from scripts.compositional_dynamics_v17.prototype import face_grid
 dense=face_grid(17);idx=[np.where(np.all(dense==v,axis=1))[0][0]for v in q]
 target=B/'calibration-development/forecast/trajectory.npz';status=rd(target.with_name('complete.json'));ck(status['completed']and h(target)==status['trajectory_sha256'],'releasedtargetseal');ev[str(target.relative_to(R))]=h(target)
 with np.load(target)as z:times=z['times_s'][::10];truth=z['surface_frames'][::10].reshape(26,-1,3)[:,idx]
 def align(x,y):
  cx=w@x;cy=w@y;u,_,vt=np.linalg.svd((x-cx).T@(w[:,None]*(y-cy)));d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt);return (x-cx)@u@d@vt+cy
 def rms(x):return np.sqrt(np.sum(w[:,None]*x*x))
 def diagnostic(x,t):
  center=np.einsum('n,tnc->tc',w,x);v=(x-center[:,None])[:,tri];volume=np.sum(np.einsum('tni,tni->tn',v[:,:,0],np.cross(v[:,:,1],v[:,:,2])),axis=1)/6;gap=x[:,:,2].min(1)-meta['forecast_public_setup']['floor_z_m'];contact=gap<=1e-6;hits=np.where(contact)[0];first=int(hits[0])if len(hits)else None;sep=apex=None
  if first is not None:
   off=np.where(~contact[first+1:])[0]
   if len(off):
    sep=first+1+int(off[0]);on=np.where(contact[sep+1:])[0];end=sep+1+int(on[0])if len(on)else len(t);k=sep+int(np.argmax(center[sep:end,2]));apex=k if sep<k<end-1 else None
  return dict(signed_closed_surface_volume_m3=volume,volume_over_rest=volume/np.prod(dims),surface_centroid_z_m=center[:,2],minimum_sampled_gap_m=gap,maximum_sampled_penetration_m=max(0.,-gap.min()),first_contact_s=None if first is None else t[first],first_separation_s=None if sep is None else t[sep],first_rebound_apex_s=None if apex is None else t[apex],first_rebound_apex_surface_centroid_z_m=None if apex is None else center[apex,2])
 paths={'ours_coarse':('matched-material-forecast/surface-prediction.npz','positions_m'),'ours_fine':('matched-material-fine-forecast/surface-prediction.npz','positions_m'),'original125':('native-phystwin-graph-recovery/forecast.npz','tracked_positions_m'),'capacity729':('native-phystwin-capacity-forecast/prediction.npz','tracked_positions_m'),'extended125':('native-phystwin-budget-forecast/prediction.npz','tracked_positions_m')};preds={};summaries={}
 for name,(path,key)in paths.items():
  path=B/path;ev[str(path.relative_to(R))]=h(path)
  with np.load(path)as z:x=z[key].astype(float);t=z['times_s'];eq(t,times,name+'clock')
  preds[name]=x;g=r['methods'][name]['geometry'];ck(x.shape==(26,98,3)and g['complete']and g['frame_coverage']==1,name+'coverage');vals=[]
  for i,(a,b)in enumerate(zip(x,truth)):
   row=dict(world_point_rms_m=rms(a-b),shape_point_rms_m=rms(align(a,b)-b),surface_centroid_error_m=np.linalg.norm(w@(a-b)),reference_nonrigid_rms_m=rms(b-align(rest,b)),predicted_nonrigid_rms_m=rms(a-align(rest,a)));vals.append(row)
   for k,v in row.items():eq(v,g['rows'][i][k],name+'frame'+k)
  for k in vals[0]:eq(np.sqrt(np.mean([v[k]**2 for v in vals])),g[k],name+'aggregate'+k)
  summaries[name]={k:g[k]*1000 for k in ['world_point_rms_m','shape_point_rms_m','reference_nonrigid_rms_m']}
  for which,xx in [('reference',truth),('prediction',x)]:
   raw=diagnostic(xx,times);saved=r['methods'][name]['physical_diagnostics'][which]
   for k,v in raw.items():
    if v is None:ck(saved[k]is None,name+which+k)
    else:eq(v,saved[k],name+which+k)
 # Stored render aggregate coverage: all26frames x3cameras, no dropped images.
 for name in paths:
  rr=r['methods'][name]['render_rows'];ck(len(rr)==78 and len({(x['frame'],x['camera'])for x in rr})==78,name+'rendercensus')
  for k,s in r['methods'][name]['render_summary'].items():
   values=[x['metrics'][k]for x in rr if x['metrics'][k]is not None];ck(s['available']==len(values)and s['expected']==78,name+'rendercoverage');eq(np.mean(values),s['mean'],name+'rendermean')
 for group in ('source_sha256','input_sha256'):
  for n,v in r[group].items():ck(h(R/n)==v,'reportedpin '+n)
 out=dict(passed=not errors,checks=len(checks),errors=errors,created_unix=time.time(),evidence_sha256=ev,methods_mm=summaries,scope='Independent all5methods x26frames rest-area world/properrigidshape/deformation/centroid plus closed-volume/contact/apex diagnostics. Render census and aggregates checked; image metric pixels assessed in separate appearance audit. One development object, finite n3 teacher, no population or fresh-cohort conclusion. No fresh-cohort targets accessed.',source_sha256={str(Path(__file__).relative_to(R)):h(__file__)})
 (D/'matched-forecast-development-final-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ['passed','checks','errors','methods_mm']}),flush=True)
if __name__=='__main__':main()
