"""Read-only, fixed-clock diagnosis. Never loads full-node trajectories."""
import json
from pathlib import Path
import numpy as np
from experiments.run_campaign_c38_graded_family_warp import record,sha,write
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/diagnose_supported_gravity_common_queries.py'
META='runs/operator-dev/p2-supported-gravity-query-diagnosis-v01'
OUT='runs/gaussian-campaign-v01/p2-supported-gravity-fine-v01'
KEYS=('times_s','material_probes_m','probe_positions_m','probe_velocities_m_s')

def prepare():
 p=dict(schema='completed-common-query-diagnosis-v01',status='FROZEN_BEFORE_COMMON_QUERY_ANALYSIS_NO_PHYSICS',source=record(ROOT/SOURCE),summary=record(ROOT/OUT/'summary.json'),results=[record(ROOT/OUT/(str(n)+'tets-refined')/'result.json') for n in [12000,20736]],allowed_array_keys=list(KEYS),metrics=['Exact identical clocks/material probes; fixed-face displacement/velocity/mismatch','Full unfiltered q/v difference and physical amplitudes; per-x-section maxima','Single least-squares first-order time-shift proxy delta_u=tau*v_fine, globally and independently each saved clock; residual fraction NOT registration gate','Time-constant discrepancy versus fluctuating discrepancy; no frequency identification from25 clocks'],limits=['25clocks over0.1s cannot resolve/identify high-frequency modes','No filtering/retiming/rescaling changes any old gate','Phase fit is descriptive tangent proxy, not proof of a unique cause','No new mechanics/material/load/boundary reads or solves'])
 path=ROOT/META/'protocol.json';path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
 return record(path)

def run(protocol,digest):
 path=ROOT/protocol
 if sha(path)!=digest:raise ValueError('Changed frozen diagnostic')
 p=json.loads(path.read_text())
 if sha(ROOT/SOURCE)!=p['source']['sha256']:raise ValueError('Changed source')
 data=[];evidence=[]
 for rec in p['results']:
  if sha(ROOT/rec['path'])!=rec['sha256']:raise ValueError('Changed result')
  r=json.loads((ROOT/rec['path']).read_text())
  if not r.get('completed') or not r.get('passed'):raise ValueError('Completed mechanical result required')
  capture=r['capture']
  if sha(ROOT/capture['path'])!=capture['sha256']:raise ValueError('Changed capture')
  with np.load(ROOT/capture['path']) as f:data.append({key:f[key] for key in KEYS})
  evidence.append(dict(result=rec,capture=capture))
 a,b=data;np.testing.assert_array_equal(a['times_s'],b['times_s']);np.testing.assert_array_equal(a['material_probes_m'],b['material_probes_m'])
 t=a['times_s'];X=a['material_probes_m'];ua=a['probe_positions_m']-X;ub=b['probe_positions_m']-X;va=a['probe_velocities_m_s'];vb=b['probe_velocities_m_s'];dq=ua-ub;dv=va-vb
 boundary=np.isclose(X[:,0],0.)|np.isclose(X[:,0],.24);interior=~boundary
 def normmax(x):return float(np.linalg.norm(x,axis=-1).max(initial=0.))
 def phase(delta,velocity):
  den=float(np.sum(velocity**2));tau=float(np.sum(delta*velocity)/den) if den else 0.;norm=float(np.linalg.norm(delta));return dict(tau_s=tau,residual_norm_fraction=float(np.linalg.norm(delta-tau*velocity)/norm) if norm else 0.)
 perclock=[dict(time_s=float(clock),**phase(dq[k,interior],vb[k,interior])) for k,clock in enumerate(t)]
 mean=dq.mean(axis=0,keepdims=True)
 report=dict(completed=True,new_physics=False,source=record(ROOT/SOURCE),protocol=record(path),evidence=evidence,original_summary=p['summary'],unfiltered=True,clock_count=len(t),query_count=len(X),boundary=dict(query_count=int(boundary.sum()),coarse_displacement_max_m=normmax(ua[:,boundary]),fine_displacement_max_m=normmax(ub[:,boundary]),coarse_velocity_max_m_s=normmax(va[:,boundary]),fine_velocity_max_m_s=normmax(vb[:,boundary]),position_mismatch_max_m=normmax(dq[:,boundary]),velocity_mismatch_max_m_s=normmax(dv[:,boundary])),physical_query_amplitudes=dict(coarse_u_max_m=normmax(ua),fine_u_max_m=normmax(ub),coarse_v_max_m_s=normmax(va),fine_v_max_m_s=normmax(vb)),raw_discrepancies=dict(q_max_m=normmax(dq),v_max_m_s=normmax(dv),q_coordinate_RMS_m=float(np.sqrt(np.mean(dq**2))),v_coordinate_RMS_m_s=float(np.sqrt(np.mean(dv**2))),q_time_constant_component_norm_fraction=float(np.linalg.norm(np.broadcast_to(mean,dq.shape))/np.linalg.norm(dq)),q_fluctuating_component_norm_fraction=float(np.linalg.norm(dq-mean)/np.linalg.norm(dq))),first_order_time_shift_proxy=phase(dq[:,interior],vb[:,interior]),time_shift_per_clock=perclock,sections=[dict(x_m=float(x),q_max_m=normmax(dq[:,np.isclose(X[:,0],x)]),v_max_m_s=normmax(dv[:,np.isclose(X[:,0],x)])) for x in np.unique(X[:,0])],frequency_or_modal_cause_identified=False,old_spatial_failure_preserved=True,original_gates_changed=False)
 write(ROOT/META/'diagnosis.json',report);return report

if __name__=='__main__':
 import argparse
 q=argparse.ArgumentParser();q.add_argument('action',choices=['prepare','run']);q.add_argument('--protocol');q.add_argument('--sha');a=q.parse_args()
 print(json.dumps(prepare() if a.action=='prepare' else run(a.protocol,a.sha),indent=2))
