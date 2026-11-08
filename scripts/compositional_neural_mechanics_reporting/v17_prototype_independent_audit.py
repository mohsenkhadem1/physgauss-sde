"""Read-only v17 feasibility audit; no dynamics replay."""
import json,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-dynamics-v17';B=ROOT/'runs/compositional-dynamics-v17/prototype'
def main():
 rd=lambda p:json.loads(Path(p).read_text());h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();p=rd(R/'prototype-plan.json');d=rd(R/'prototype-results.json');a=rd(R/'prototype-analysis.json');errors=[];checks=0
 def ck(v,msg):
  nonlocal checks
  checks+=1
  if not v:errors.append(msg)
 def eq(x,y,msg):ck(np.allclose(x,y,rtol=1e-8,atol=2e-12),msg)
 for f,v in p['source_sha256'].items():ck(h(ROOT/f)==v,'source '+f)
 for table in ('source_sha256','input_sha256'):
  for f,v in a[table].items():ck(h(ROOT/f)==v,'analysis pin '+f)
 ck(d['plan_sha256']==h(R/'prototype-plan.json')and d['terminal']and d['completed']==8 and not d['missing_keys'],'8terminalcomplete')
 rows={r['key']:r for r in d['rows']};ck(set(rows)==set(d['expected_keys'])and len(rows)==8,'exact census');arrays={};events={}
 for key,row in rows.items():
  path=B/key/'trajectory.npz';ck(h(path)==row['trajectory_sha256'],'array seal')
  with np.load(path)as x:arr={k:x[k].copy()for k in x.files}
  arrays[key]=arr;ck(all(np.isfinite(v).all()for v in arr.values()),'finite');led=arr['ledger'];mass=arr['mass_row_sums'].sum();vel=led[:,15:18]/mass;g=np.array([0,0,-9.81])if row['case']=='drop'else np.zeros(3);dt=row['dt'];c0=np.average(arr['rest_nodes']+arr['u'][0],axis=0,weights=arr['mass_row_sums']);acc=g-led[:,9:12]/mass;com=np.vstack([c0,c0+np.cumsum(dt*vel[:-1]+.5*dt**2*acc[:-1],axis=0)]);stride=round(.001/dt);direct=np.einsum('tnc,n->tc',arr['rest_nodes'][None]+arr['u'],arr['mass_row_sums'])/mass;eq(com[::stride],direct,'COM ledger vs saved positions')
  eq(np.max(abs(led[:,1]-row['initial_energy_J']))/row['energy_scale_J'],row['maximum_relative_energy_drift'],'energy drift');eq(led[:,7].min(),row['minimum_J'],'minJ');eq(arr['saved_dense_penetration_m'].max(),row['maximum_dense_penetration_m'],'dense max')
  impulse=-np.sum((led[:-1,9:12]+led[1:,9:12])*.5*np.diff(led[:,0])[:,None],axis=0);e=a['events'][key];eq(impulse,e['contact_impulse_on_soft_Ns'],'impulse');eq(com[-1],e['final_COM_m'],'finalCOM');eq(np.linalg.norm(led[-1,15:18]-led[0,15:18]-mass*g*led[-1,0]-impulse),e['momentum_balance_Ns'],'momentumbalance')
  active=led[:,4]>0;ons=np.where(active & ~np.r_[False,active[:-1]])[0];offs=np.where(~active & np.r_[False,active[:-1]])[0];eq(led[ons,0],e['contact_on_times_s'],'onsets');eq(led[offs,0],e['contact_off_times_s'],'offsets');peaks=[]
  for j in range(1,len(vel)):
   if len(ons)and j>ons[0]and vel[j-1,2]>0>=vel[j,2]and not(active[j]or active[j-1]):
    tau=-vel[j-1,2]/g[2];peaks.append([led[j-1,0]+tau,com[j-1,2]+vel[j-1,2]*tau+.5*g[2]*tau*tau])
  eq(np.array(peaks).reshape(-1,2),np.array([[v['time_s'],v['com_z_m']]for v in e['free_flight_apices']]).reshape(-1,2),'airborne apices');events[key]=e
 def aligned(x,y):
  xc=x-x.mean(0);yc=y-y.mean(0);u,_,v=np.linalg.svd(xc.T@yc);s=np.eye(3);s[-1,-1]=np.linalg.det(u@v);return xc@u@s@v+y.mean(0)
 for c in a['comparisons']:
  x,y=arrays[c['a']],arrays[c['b']];eq(x['times_s'],y['times_s'],'same samples');pred=x['surface_frames'].reshape(len(x['times_s']),-1,3);ref=y['surface_frames'].reshape(len(y['times_s']),-1,3);world=np.sqrt(np.mean((pred-ref)**2));shape=np.sqrt(np.mean([np.mean((aligned(v,w)-w)**2)for v,w in zip(pred,ref)]));eq(world,c['world_coordinate_RMS_m'],'world RMS');eq(shape,c['shape_coordinate_RMS_m'],'shape RMS');imp=np.linalg.norm(np.array(events[c['a']]['contact_impulse_on_soft_Ns'])-events[c['b']]['contact_impulse_on_soft_Ns']);tol=.0001+.05*np.linalg.norm(events[c['b']]['contact_impulse_on_soft_Ns']);eq(imp,c['impulse_difference_Ns'],'impulse difference');eq(tol,c['impulse_tolerance_Ns'],'impulse tolerance');L=min(p['dimensions']);cr=dict(world=world<=.005*L,shape=shape<=.0025*L,impulse=imp<=tol,finite_J=rows[c['a']]['minimum_J']>.1 and rows[c['b']]['minimum_J']>.1,penetration=max(rows[c['a']]['maximum_dense_penetration_m'],rows[c['b']]['maximum_dense_penetration_m'])<=.02*L)
  if c['case']=='drop':
   ea,eb=events[c['a']],events[c['b']];impact=abs(ea['first_impact_time_s']-eb['first_impact_time_s']);eq(impact,c['impact_time_difference_s'],'impact');cr['impact_time']=impact<=.01*np.sqrt(L/9.81);aa,bb=ea['free_flight_apices'],eb['free_flight_apices'];cr['first_apex_present']=bool(aa and bb)
   if aa and bb:
    height=abs(aa[0]['com_z_m']-bb[0]['com_z_m']);timing=abs(aa[0]['time_s']-bb[0]['time_s']);eq(height,c['first_apex_height_difference_m'],'apexheight');eq(timing,c['first_apex_time_difference_s'],'apextime');cr.update(first_apex_height=height<=.02*L,first_apex_time=timing<=.01*np.sqrt(L/9.81))
  ck({k:bool(v)for k,v in cr.items()}==c['checks']and all(cr.values())==c['passed'],'all comparison gates')
 out=dict(passed=not errors,checks=checks,errors=errors,created_unix=time.time(),evidence_sha256={str(f.relative_to(ROOT)):h(f)for f in [R/'prototype-plan.json',R/'prototype-results.json',R/'prototype-analysis.json']},numerically_qualified=False,scope='Independent all8 rawtrajectory finite/hash/energy/COM/momentum/events and6comparison metrics/gates. Both spatial1/2 comparisons fail; no finaldynamicqualification. Analysiscomparisons do not include energy acceptance in their passed flag; energy drift remains separately reported. SurfaceuniformcoordinateRMS notmassnorm. Contacttoggles notallbounces.')
 (R/'prototype-final-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(checks,errors)
if __name__=='__main__':main()
