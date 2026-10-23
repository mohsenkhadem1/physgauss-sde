"""Read-only physical mass/rigid/elastic diagnosis of completed root128 fields."""
import argparse,json,os,signal,sys,time
from pathlib import Path
import numpy as np
from experiments import run_p2_contact_spatial_pairs as pair
from experiments.campaign_static_p2_block import shapes
from experiments.campaign_p2_keast_quadrature import positive_keast24
ROOT=pair.ROOT;META=ROOT/'runs/operator-dev/p2-contact-rigid-elastic-diagnosis-v01'
SOURCE='experiments/diagnose_p2_contact_rigid_elastic.py';TEST='tests/test_diagnose_p2_contact_rigid_elastic.py'

def rigid_fit(X,q,v,weights,query_X,query_q):
 """Positive exact-rest-volume quadrature; velocities use instantaneous rigid LS."""
 w=np.asarray(weights);mass=w.sum();Xc=w@X/mass;qc=w@q/mass;vc=w@v/mass;A=X-Xc;B=q-qc;C=A.T@(w[:,None]*B);U,_,Vh=np.linalg.svd(C);D=np.eye(3);D[-1,-1]=np.linalg.det(U@Vh);Rrow=U@D@Vh
 I=np.eye(3)*np.sum(w*np.sum(B*B,axis=1))-(B.T@(w[:,None]*B));L=np.sum(w[:,None]*np.cross(B,v-vc),axis=0);omega=np.linalg.solve(I,L);rq=(query_X-Xc)@Rrow+qc;rv=vc+np.cross(omega,query_q-qc)
 return dict(COM_m=qc,COM_velocity_m_s=vc,rotation=Rrow.T,angular_velocity_rad_s=omega,rigid_query_positions_m=rq,rigid_query_velocities_m_s=rv,bulk_nonrigid_coordinate_RMS_m=float(np.sqrt(np.sum(w[:,None]*(q-(A@Rrow+qc))**2)/(3*mass))),bulk_nonrigid_coordinate_velocity_RMS_m_s=float(np.sqrt(np.sum(w[:,None]*(v-(vc+np.cross(omega,B)))**2)/(3*mass))))

def contribution(total,rigid):
 elastic=total-rigid;den=float(np.sum(total*total));out=dict(total_coordinate_RMS=float(np.sqrt(np.mean(total**2))),rigid_coordinate_RMS=float(np.sqrt(np.mean(rigid**2))),nonrigid_coordinate_RMS=float(np.sqrt(np.mean(elastic**2))),total_max_vector=float(np.linalg.norm(total,axis=-1).max()),rigid_max_vector=float(np.linalg.norm(rigid,axis=-1).max()),nonrigid_max_vector=float(np.linalg.norm(elastic,axis=-1).max()))
 if den:out.update(rigid_L2_norm_fraction=float(np.linalg.norm(rigid)/np.sqrt(den)),nonrigid_L2_norm_fraction=float(np.linalg.norm(elastic)/np.sqrt(den)),cross_term_over_total_squared=float(2*np.sum(rigid*elastic)/den))
 return out

def prepare():
 records=[]
 paths=[('96tets',[4,2,2],ROOT/'runs/gaussian-campaign-v01/p2-corrected-contact-order128-trajectory-v01/outer128_accuracy_1e10/summary.json')]+[(name,pair.PHASES[name]['cells'],ROOT/f'runs/gaussian-campaign-v01/p2-contact-spatial-pairs-v01/{name}/accuracy_1e10/summary.json') for name in pair.PHASES]
 for name,cells,path in paths:
  d=json.loads(path.read_text())
  if not d['completed'] or not d['passed'] or pair.sha(ROOT/d['capture']['path'])!=d['capture']['sha256']:raise ValueError('Completedmechanicalfields only')
  records.append(dict(name=name,cells=cells,summary=dict(path=str(path.relative_to(ROOT)),sha256=pair.sha(path)),capture=d['capture']))
 base=json.loads((pair.META/'768tets/protocol-v02.json').read_text());pins={**base['source_sha256'],SOURCE:pair.sha(ROOT/SOURCE),TEST:pair.sha(ROOT/TEST)}
 p=dict(schema='completed-contact-rigid-elastic-readonly-v01',status='FROZEN_BEFORE_DIAGNOSIS',source_sha256=pins,records=records,CPU_affinity_cores=[0,1],hard_cap_s=180,quadrature='POSITIVEKeast24 exact degree6: fixedreference rho*dV; P2positions/velocities quadratic, covariance/kinetic degree4 exact',rigid_position_fit='Massweighted proper Kabsch from referencequadrature tocurrentquadrature. AppliedtoSAME81materialqueries.',rigid_velocity_fit='Instantaneousmassweighted LS v=Vcom+omega cross(q-COM) onCURRENTquadrature; notderivativeofKabschR.',accepted_COM_P='ExactconsistentP2 nodalINTEGRALweights M@ones (maybenegative, NOTphysicalnegativepointmasses), equivalentlypositivequadraturelinearfieldintegrals; comparePtoSAVEDauxnetimpulse.',contact_onset_proxy='Firstsavedaccepted cumulativeupwardcontactimpulse>1e-8Ns, thresholdfixedbeforeanalysis; outputcontactE/analyticpenetrationreportedonly9clocks. Not exacteventtime.',operator_comparison='ALL27outputstates: CPUroot128/CPUroot64/independentNumPy+strictadaptiveEg. Everyreferencefailureexplicit; no partialsumreference or statuswaiver.',no_physics_GPU_or_gate_changes=True,nonclaims=['Rigid/nonrigidL2componentscanexceedtotalwhentheyanti-correlate; notadditivepercentages','Decompositionisdiagnosticnotuniquedynamicalcausalidentification','Operatorfixedstateagreementdoesnotboundtrajectoryerror','141Jpointguardsarenotwholeelementtimeintervalcertificate','No contactdatasetadmission ornewphysicalscenarioexecution'])
 path=META/'protocol.json'
 if path.exists():raise FileExistsError('Preserve recipe')
 pair.write(path,p);return pair.sha(path)

def run(digest):
 path=META/'protocol.json'
 if pair.sha(path)!=digest:raise ValueError('Changedrecipe')
 p=json.loads(path.read_text())
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or set(os.sched_getaffinity(0))!=set(p['CPU_affinity_cores']):raise ValueError('CPU2ONLY')
 if any(pair.sha(ROOT/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Changedsource')
 sys.path.append(str(pair.ENV_SITE));import warp as wp
 from experiments.campaign_p2_warp_reference_order128 import WarpReferenceForce128
 from experiments.campaign_p2_warp_reference_force import WarpReferenceForce
 from experiments.campaign_contact_geometry_controls import independent_adaptive_response,differences
 from experiments.run_p2_contact_validation import build
 from experiments.campaign_p2_consistent_transient import scalar_mass
 def cap(*args):raise TimeoutError('FrozenCPU180s cap')
 signal.signal(signal.SIGALRM,cap);signal.alarm(180);wp.config.kernel_cache_dir=str(META/'warp-cache-CPU');start=time.monotonic();s=dict(completed=False,protocol_sha256=digest,meshes=[]);allfields=[]
 try:
  b,w=positive_keast24();N=shapes(b)
  for record in p['records']:
   for rec in [record['summary'],record['capture']]:
    if pair.sha(ROOT/rec['path'])!=rec['sha256']:raise ValueError('Changedevidence')
   m=build(record['cells'],300000.,9.81);M=scalar_mass(m);mw=np.asarray(M@np.ones(len(m.rest)));mass=float(mw.sum());W=(m.density*m.volumes[:,None]*w[None]).ravel();X=np.einsum('qa,eai->eqi',N,m.rest[m.elements]).reshape(-1,3);obj128=WarpReferenceForce128(m,device='cpu');obj64=WarpReferenceForce(m,device='cpu')
   with np.load(ROOT/record['capture']['path'],allow_pickle=False) as z:
    np.testing.assert_array_equal(z['rest_m'],m.rest);q=z['positions_m'];v=z['velocities_m_s'];t=z['times_s'];probes=z['material_probes_m'];pq=z['query_positions_m'];pv=np.array([m.query(a,probes) for a in v]);accepted_t=z['accepted_times_s'];states=z['accepted_states'];Paux=states[2*m.rest.size:].T;uA=states[:m.rest.size].reshape(len(m.rest),3,-1);vA=states[m.rest.size:2*m.rest.size].reshape(len(m.rest),3,-1);COMA=np.einsum('n,nct->tc',mw,uA)/mass+(mw@m.rest)/mass;PA=np.einsum('n,nct->tc',mw,vA);P0=mass*GRAVITY_T0();contact_impulse=Paux-mass*(accepted_t-1/12)[:,None]*np.array([0.,0.,-9.81]);impidx=np.flatnonzero(contact_impulse[:,2]>1e-8);Pdef=float(np.linalg.norm(PA-P0-Paux,axis=1).max());energies=z['output_mechanical_energies_J'];penetrations=z['output_analytic_face_penetrations_m'];accepted_E=z['accepted_mechanical_energies_J']
   row=dict(name=record['name'],cells=record['cells'],physical_mass_kg=mass,positive_quad_mass_kg=float(W.sum()),mass_COM_quad_agreement_max_m=0.,mass_velocity_quad_agreement_max_m_s=0.,accepted_count=len(accepted_t),accepted_P_vs_saved_netimpulse_max_Ns=Pdef,accepted_contact_impulse_crosses_1e8_time_s=float(accepted_t[impidx[0]]) if len(impidx) else None,accepted_final_contact_impulse_Ns=contact_impulse[-1].tolist(),accepted_COM_range_m=[COMA.min(0).tolist(),COMA.max(0).tolist()],accepted_energy_range_J=[float(accepted_E.min()),float(accepted_E.max())],outputs=[]);s['meshes'].append(row);fits=[]
   for k,clock in enumerate(t):
    Q=np.einsum('qa,eai->eqi',N,q[k,m.elements]).reshape(-1,3);V=np.einsum('qa,eai->eqi',N,v[k,m.elements]).reshape(-1,3);fit=rigid_fit(X,Q,V,W,probes,pq[k]);fits.append(fit);com=mw@q[k]/mass;vc=mw@v[k]/mass;row['mass_COM_quad_agreement_max_m']=max(row['mass_COM_quad_agreement_max_m'],float(np.linalg.norm(com-fit['COM_m'])));row['mass_velocity_quad_agreement_max_m_s']=max(row['mass_velocity_quad_agreement_max_m_s'],float(np.linalg.norm(vc-fit['COM_velocity_m_s'])));u=np.ascontiguousarray(q[k]-m.rest);a=obj128.evaluate(u);c=obj64.evaluate(u);out=dict(time_s=float(clock),COM_m=fit['COM_m'].tolist(),COM_velocity_m_s=fit['COM_velocity_m_s'].tolist(),angular_velocity_rad_s=fit['angular_velocity_rad_s'].tolist(),bulk_nonrigid_coordinate_RMS_m=fit['bulk_nonrigid_coordinate_RMS_m'],bulk_nonrigid_coordinate_velocity_RMS_m_s=fit['bulk_nonrigid_coordinate_velocity_RMS_m_s'],contact_energy_J=a['contact_energy'],root128_total_contact_force_proxy_N=(-a['gradient'].sum(0)).tolist(),root64_vs128=differences(a,c),mechanical_energy_J=float(energies[k]),analytic_maximum_face_penetration_m=float(penetrations[k]))
    try:r=independent_adaptive_response(m,u,obj128.arrays);out['root128_vs_independent_adaptive']=differences(a,r);out['adaptive_reference_completed']=True
    except (ValueError,RuntimeError) as e:out.update(adaptive_reference_completed=False,adaptive_error=repr(e))
    row['outputs'].append(out);pair.write(META/'result.json',s)
   allfields.append(dict(times=t,queries=pq,query_velocity=pv,rigid_q=np.array([f['rigid_query_positions_m'] for f in fits]),rigid_v=np.array([f['rigid_query_velocities_m_s'] for f in fits]),COM=np.array([f['COM_m'] for f in fits]),COM_velocity=np.array([f['COM_velocity_m_s'] for f in fits])))
  comparisons=[]
  for a,b in zip(allfields[:-1],allfields[1:]):
   np.testing.assert_array_equal(a['times'],b['times']);comparisons.append(dict(position=contribution(a['queries']-b['queries'],a['rigid_q']-b['rigid_q']),velocity=contribution(a['query_velocity']-b['query_velocity'],a['rigid_v']-b['rigid_v']),COM_maximum_difference_m=float(np.linalg.norm(a['COM']-b['COM'],axis=1).max()),COM_velocity_maximum_difference_m_s=float(np.linalg.norm(a['COM_velocity']-b['COM_velocity'],axis=1).max())))
  s.update(completed=True,common81_pair_decompositions=comparisons,no_new_physics=True,old_gates_unchanged=True)
 except BaseException as error:s['error']=repr(error)
 finally:signal.alarm(0);s['seconds']=time.monotonic()-start;pair.write(META/'result.json',s)
 return s

def GRAVITY_T0():return np.array([0.,0.,-9.81/12])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha);print(json.dumps(r if isinstance(r,str) else dict(completed=r['completed'],seconds=r['seconds'],error=r.get('error'))))
