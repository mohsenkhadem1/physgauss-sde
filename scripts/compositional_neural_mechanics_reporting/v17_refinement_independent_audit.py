"""Independent saved-field refinement audit, no rollout or production metrics."""
import json,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-dynamics-v17'
def main():
 read=lambda p:json.loads(Path(p).read_text());h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();p=read(R/'refinement-plan.json');old=read(R/'prototype-plan.json');d=read(R/'refinement-results.json');a=read(R/'refinement-analysis.json');errors=[];checks=0
 def ck(v,msg):
  nonlocal checks
  checks+=1
  if not v:errors.append(msg)
 def eq(x,y,msg):ck(np.allclose(x,y,rtol=1e-8,atol=2e-12),msg)
 for obj in (p,a):
  for table in ('source_sha256','input_sha256'):
   for f,v in obj[table].items():ck(h(ROOT/f)==v,'pin '+f)
 ck(p['gates']==old['gates'],'unchanged gates');ck(d['plan_sha256']==h(R/'refinement-plan.json')and d['terminal']and d['completed']==21 and not d['missing_keys'],'terminal21');rows={x['task_id']:x for x in d['rows']};ck(set(rows)=={t['id']for t in p['tasks']}and len(rows)==21,'census');data={};event={};reused=0
 for key,row in rows.items():
  path=ROOT/row['path']/'trajectory.npz';ck(h(path)==row['trajectory_sha256'],'arrayhash');ck(row['task']==next(t for t in p['tasks']if t['id']==key),'tasksettings')
  if 'reuse'in row:
   reused+=1;s=ROOT/row['reuse']['source'];ck(h(s/'trajectory.npz')==h(path)==row['reuse']['trajectory_sha256']and h(s/'complete.json')==row['reuse']['complete_sha256'],'exactreuse')
  with np.load(path)as raw:data[key]={k:raw[k].copy()for k in raw.files}
  x=data[key];ck(all(np.isfinite(v).all()for v in x.values()),'finite');led=x['ledger'];eq(led[:,7].min(),row['minimum_J'],'J');eq(x['saved_dense_penetration_m'].max(),row['maximum_dense_penetration_m'],'depth');eq(np.max(abs(led[:,1]-row['initial_energy_J']))/row['energy_scale_J'],row['maximum_relative_energy_drift'],'energy')
  mass=x['mass_row_sums'].sum();v=led[:,15:18]/mass;g=np.array([0,0,-9.81])if row['case']=='drop'else np.zeros(3);dt=row['dt'];c0=np.average(x['rest_nodes']+x['u'][0],axis=0,weights=x['mass_row_sums']);com=np.vstack([c0,c0+np.cumsum(dt*v[:-1]+.5*dt*dt*(g-led[:-1,9:12]/mass),axis=0)]);stride=round(.001/dt);eq(com[::stride],np.einsum('tnc,n->tc',x['rest_nodes'][None]+x['u'],x['mass_row_sums'])/mass,'COM recurrence');imp=-np.sum(.5*(led[1:,9:12]+led[:-1,9:12])*np.diff(led[:,0])[:,None],axis=0);active=led[:,4]>0;on=np.where(active&~np.r_[False,active[:-1]])[0];ap=[]
  for j in range(1,len(v)):
   if len(on)and j>on[0]and v[j-1,2]>0>=v[j,2]and not(active[j]or active[j-1]):
    tau=-v[j-1,2]/g[2];ap.append((led[j-1,0]+tau,com[j-1,2]+v[j-1,2]*tau+.5*g[2]*tau*tau))
  event[key]=dict(imp=imp,impact=float(led[on[0],0])if len(on)else None,ap=ap);stored=a['events'][key];eq(imp,stored['contact_impulse_on_soft_Ns'],'impulse');eq(np.array(ap).reshape(-1,2),np.array([(z['time_s'],z['com_z_m'])for z in stored['free_flight_apices']]).reshape(-1,2),'apices')
 ck(reused==2,'two exactreuse')
 def align(x,y):
  xx=x-x.mean(0);yy=y-y.mean(0);u,_,v=np.linalg.svd(xx.T@yy);s=np.eye(3);s[-1,-1]=np.linalg.det(u@v);return xx@u@s@v+y.mean(0)
 for c in a['comparisons']:
  x,y=data[c['a']],data[c['b']];eq(x['times_s'],y['times_s'],'times');xx=x['surface_frames'].reshape(len(x['times_s']),-1,3);yy=y['surface_frames'].reshape(len(y['times_s']),-1,3);world=np.sqrt(np.mean((xx-yy)**2));shape=np.sqrt(np.mean([np.mean((align(q,z)-z)**2)for q,z in zip(xx,yy)]));eq(world,c['world_coordinate_RMS_m'],'world');eq(shape,c['shape_coordinate_RMS_m'],'shape');ea,eb=event[c['a']],event[c['b']];imp=np.linalg.norm(ea['imp']-eb['imp']);tol=.0001+.05*np.linalg.norm(eb['imp']);eq(imp,c['impulse_difference_Ns'],'impdifference');eq(tol,c['impulse_tolerance_Ns'],'imptol');ra,rb=rows[c['a']],rows[c['b']];g=dict(world=world<=.005*.06,shape=shape<=.0025*.06,impulse=imp<=tol,finite_J=min(ra['minimum_J'],rb['minimum_J'])>.1,penetration=max(ra['maximum_dense_penetration_m'],rb['maximum_dense_penetration_m'])<=.02*.06,energy=max(ra['maximum_relative_energy_drift'],rb['maximum_relative_energy_drift'])<=.01)
  if c['case']=='drop':
   g['impact_present']=ea['impact']is not None and eb['impact']is not None
   if g['impact_present']:v=abs(ea['impact']-eb['impact']);eq(v,c['impact_time_difference_s'],'impacttiming');g['impact_time']=v<=.01*np.sqrt(.06/9.81)
   g['first_apex_present']=bool(ea['ap']and eb['ap'])
   if g['first_apex_present']:
    dt=abs(ea['ap'][0][0]-eb['ap'][0][0]);dz=abs(ea['ap'][0][1]-eb['ap'][0][1]);eq(dt,c['first_apex_time_difference_s'],'apextime');eq(dz,c['first_apex_height_difference_m'],'apexheight');g.update(first_apex_height=dz<=.02*.06,first_apex_time=dt<=.01*np.sqrt(.06/9.81))
  ck({k:bool(v)for k,v in g.items()}==c['checks']and all(g.values())==c['passed'],'all gates')
 for case in ('release','drop'):
  req=[c for c in a['comparisons']if c['case']==case and((c['type']in ('temporal','volume','contact_quadrature')and c['n']in (3,4))or(c['type']=='spatial'and c['pair']==[3,4]))];ck(a['cases'][case]['passed']==all(c['passed']for c in req),'latestconjunction');ck(len(a['cases'][case]['required'])==len(req),'requiredcount')
 ck(not a['all_candidate_gates_pass']and not any(c['passed']for c in a['cases'].values()),'bothunqualified')
 out=dict(passed=not errors,checks=checks,errors=errors,created_unix=time.time(),evidence_sha256={str(R/n):h(R/n)for n in ['refinement-plan.json','refinement-results.json','refinement-analysis.json']},scientific_qualification=a['cases'],scope='All21rawarrays and23comparisons recomputed independently includingproperrigidshape,energy,COM,impulse andairborneevents. Bothlatest3/4spatial gates fail; allotherrequiredchecks pass. Integritypass doesnotqualifiedynamics. q4diagnosticcannotrescueoriginalacceptance.')
 (R/'refinement-final-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(checks,errors)
if __name__=='__main__':main()
