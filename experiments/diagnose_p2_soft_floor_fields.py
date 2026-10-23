"""Read-only exact-mass rigid/nonrigid fields for the NEW softer floor screen."""
import argparse,json,os,time
import numpy as np
from experiments import run_p2_soft_floor_screen as screen
from experiments.diagnose_p2_contact_rigid_elastic import rigid_fit,contribution
from experiments.campaign_static_p2_block import shapes
from experiments.campaign_p2_keast_quadrature import positive_keast24
ROOT=screen.ROOT;OP=screen.OP;SOURCE='experiments/diagnose_p2_soft_floor_fields.py'

def prepare():
 p=dict(schema='NEW-softer-floor-readonly-fields-v01',source_sha256={SOURCE:screen.sha(ROOT/SOURCE),'experiments/diagnose_p2_contact_rigid_elastic.py':screen.sha(ROOT/'experiments/diagnose_p2_contact_rigid_elastic.py')},screen_protocol_sha256=screen.sha(OP/'protocol.json'),metrics='All nine clocks/all three meshes: positive exact Keast24 rho*dV proper Kabsch and instantaneous velocity least squares; exact consistent-M COM/P versus saved net impulse. Common81 rigid/nonrigid decomposition including cross terms; no projection of physical fields.',CPU_affinity_cores=[0,1],no_new_physics=True,no_gate_changes=True,no_time_surface_or_dataset_admission=True)
 f=OP/'field-diagnosis-protocol.json'
 if f.exists():raise FileExistsError(f)
 screen.write(f,p);return screen.sha(f)

def run(digest):
 f=OP/'field-diagnosis-protocol.json';p=json.loads(f.read_text());start=time.monotonic()
 if screen.sha(f)!=digest or any(screen.sha(ROOT/k)!=v for k,v in p['source_sha256'].items()) or screen.sha(OP/'protocol.json')!=p['screen_protocol_sha256']:raise ValueError('Changed frozen diagnosis recipe')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or set(os.sched_getaffinity(0))!={0,1}:raise ValueError('CPU0,1 only')
 screen.validate(p['screen_protocol_sha256']);execution=json.loads((OP/'execution.json').read_text())
 if not execution['terminal'] or not execution['completed']:raise ValueError('All three complete frozen fields required')
 from experiments.run_p2_contact_validation import build
 from experiments.campaign_p2_consistent_transient import scalar_mass
 b,w=positive_keast24();N=shapes(b);rows=[];fields=[];evidence=[]
 for job in screen.JOBS:
  summary_file=screen.OUT/job['id']/screen.CONFIG['name']/'summary.json';summary=json.loads(summary_file.read_text());capture=summary['capture'];cache=next(r['summary'] for r in execution['records'] if r['job']['id']==job['id'])
  if not summary['completed'] or screen.sha(summary_file)!=cache['sha256'] or screen.sha(ROOT/capture['path'])!=capture['sha256']:raise ValueError('Changed completed field')
  evidence.append(dict(summary=cache,capture=capture));m=build(job['cells'],300000.,9.81);mw=np.asarray(scalar_mass(m)@np.ones(len(m.rest)));mass=mw.sum();W=(m.density*m.volumes[:,None]*w[None]).ravel();X=np.einsum('qa,eai->eqi',N,m.rest[m.elements]).reshape(-1,3);fits=[];outputs=[]
  with np.load(ROOT/capture['path'],allow_pickle=False) as z:
   q=z['positions_m'];v=z['velocities_m_s'];times=z['times_s'];probes=z['material_probes_m'];pq=z['query_positions_m'];pv=z['query_velocities_m_s'];state=z['accepted_states'];size=m.rest.size;P=np.einsum('n,nct->tc',mw,state[size:2*size].reshape(len(m.rest),3,-1));P0=mass*np.array([0.,0.,-9.81/12]);Perror=float(np.linalg.norm(P-P0-state[2*size:].T,axis=1).max());contact_impulse=state[2*size:].T-mass*(z['accepted_times_s']-1/12)[:,None]*np.array([0.,0.,-9.81])
   for k,t in enumerate(times):
    Q=np.einsum('qa,eai->eqi',N,q[k,m.elements]).reshape(-1,3);V=np.einsum('qa,eai->eqi',N,v[k,m.elements]).reshape(-1,3);fit=rigid_fit(X,Q,V,W,probes,pq[k]);fits.append(fit);outputs.append(dict(time_s=float(t),COM_m=fit['COM_m'].tolist(),COM_velocity_m_s=fit['COM_velocity_m_s'].tolist(),angular_velocity_rad_s=fit['angular_velocity_rad_s'].tolist(),bulk_nonrigid_coordinate_RMS_m=fit['bulk_nonrigid_coordinate_RMS_m'],bulk_nonrigid_coordinate_velocity_RMS_m_s=fit['bulk_nonrigid_coordinate_velocity_RMS_m_s'],analytic_face_penetration_m=float(z['output_analytic_face_penetrations_m'][k])))
  fields.append(dict(q=pq,v=pv,rq=np.array([f['rigid_query_positions_m'] for f in fits]),rv=np.array([f['rigid_query_velocities_m_s'] for f in fits]),times=times));rows.append(dict(id=job['id'],mass_kg=float(mass),quadrature_mass_kg=float(W.sum()),accepted_P_vs_saved_net_impulse_max_Ns=Perror,final_contact_impulse_Ns=contact_impulse[-1].tolist(),outputs=outputs))
 comparisons=[]
 for a,b in zip(fields[:-1],fields[1:]):
  np.testing.assert_array_equal(a['times'],b['times']);comparisons.append(dict(position=contribution(a['q']-b['q'],a['rq']-b['rq']),velocity=contribution(a['v']-b['v'],a['rv']-b['rv'])))
 result=dict(completed=True,protocol_sha256=digest,evidence=evidence,meshes=rows,common81_pair_decompositions=comparisons,seconds=time.monotonic()-start,diagnostic_only=True,contact_dataset_admitted=False);screen.write(OP/'field-diagnosis.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha);print(json.dumps(r if isinstance(r,str) else dict(completed=r['completed'],seconds=r['seconds'])))
