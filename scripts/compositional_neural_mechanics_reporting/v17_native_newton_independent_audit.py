"""Independent saved-array native Newton audit; does not import production scorer."""
import json,hashlib,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-dynamics-v17';B=ROOT/'runs/compositional-dynamics-v17/native-newton-dynamics'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 errors=[];count=0;evidence={};data={};summaries=[]
 def ck(ok,msg):
  nonlocal count
  count+=1
  if not ok:errors.append(msg)
 def read(p):evidence[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
 p=read(R/'native-newton-dynamics-plan.json');r=read(R/'native-newton-dynamics-results.json');t=read(B/'terminal.json')
 for f,h in p['source_sha256'].items():ck(sha(f)==h,'source '+f)
 ck(sha(ROOT/'scripts/compositional_dynamics_v17/native_newton_analysis.py')==r['source_sha256'],'analysis source')
 for f,h in r['evidence_sha256'].items():ck(sha(ROOT/f)==h,'evidence '+f)
 names=[j['name']for j in p['jobs']];ck(len(names)==len(set(names))==12,'12 unique');ck([x['job']['name']for x in t['rows']]==names,'exact terminal census');ck(r['qualified']is False,'unqualified')
 for job,entry,reported in zip(p['jobs'],t['rows'],r['rows']):
  name=job['name'];s=read(B/name/'complete.json');f=B/name/'trajectory.npz';ck(s['complete']and entry['state']=='complete','successful '+name);ck(s['job']==job,'job identity');ck(sha(f)==s['trajectory_sha256']==entry['trajectory.npz_sha256'],'arrayhash');evidence[str(f.relative_to(ROOT))]=sha(f)
  with np.load(f)as z:a={k:z[k]for k in z.files}
  data[name]=a
  for key,value in a.items():ck(np.isfinite(value).all(),'finite '+name+'/'+key)
  duration=p['duration_s'][job['case']];ck(len(a['times_s'])==round(duration/.001)+1,'frame census');ck(np.allclose(a['times_s'],np.arange(len(a['times_s']))*.001,atol=1e-12),'clock');ck(abs(a['masses'].sum()-.27)<1e-7,'mass')
  L=a['ledger'];W=a['contact_wrench_on_soft_about_world_origin'];impulse=job['dt']*W[1:,:3].sum(0);gravity=np.array([0,0,-9.8 if job['case']=='drop'else 0]);dp=np.sum(a['masses'][:,None]*(a['velocities'][-1]-a['velocities'][0]),axis=0);balance=dp-impulse-gravity*a['masses'].sum()*duration
  ck(np.allclose(impulse,s['contact_impulse_Ns'],atol=1e-12),'impulse');ck(np.allclose(balance,s['momentum_balance_residual_Ns'],atol=1e-12),'momentum')
  if job['case']=='drop':ck(impulse[2]>=0,'floor impulse upward')
  E=L[:,1:5].sum(1);d=reported['diagnostics'];ck(np.isclose(d['max_relative_energy_change'],max(abs(E-E[0]))/max(abs(E[0]),1e-12),atol=1e-12),'energy change');ck(np.isclose(d['minimum_J'],L[:,5].min()),'J');ck(np.isclose(d['max_penetration_m'],L[:,6].max()),'depth')
  summaries.append(dict(name=name,energy_loss_fraction=float((E[0]-E[-1])/E[0]),momentum_balance_norm_Ns=float(np.linalg.norm(balance))))
 comparisons=[]
 for c in r['comparisons']:
  x=data[c['left']]['surface_frames'].reshape(-1,1734,3);y=data[c['right']]['surface_frames'].reshape(-1,1734,3);world=np.sqrt(np.mean((x-y)**2));sq=[]
  for xx,yy in zip(x,y):
   xx=xx-xx.mean(0);yy=yy-yy.mean(0);U,S,Vt=np.linalg.svd(yy.T@xx);D=np.eye(3);D[-1,-1]=np.linalg.det(U@Vt);sq.append(np.mean((yy@(U@D@Vt)-xx)**2))
  shape=np.sqrt(np.mean(sq));ck(np.isclose(world,c['world_coordinate_RMS_m'],atol=1e-12),'world '+c['left']);ck(np.isclose(shape,c['shape_coordinate_RMS_m'],atol=1e-12),'shape '+c['left']);ck(c['world_le_0_25mm']==bool(world<=.00025)and c['shape_le_0_25mm']==bool(shape<=.00025),'thresholds');comparisons.append(dict(kind=c['kind'],left=c['left'],world_mm=world*1000,shape_mm=shape*1000))
 out=dict(passed=not errors,errors=errors,checks=count,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),evidence_sha256=evidence,rows=summaries,comparisons=comparisons,scope='All12 native development runs complete, but numerical reference remains unqualified. Energy loss is numerical with declared zero physical damping; not evidence of material viscosity. Different mass/contact discretization and drop gravity prevent direct superiority inference. Contact sign checked against installed native static-plane kernel in prefit review.')
 (R/'native-newton-dynamics-final-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],errors=errors,checks=count,comparisons=comparisons),indent=2))
if __name__=='__main__':main()
