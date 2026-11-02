"""Combine exact completed original jobs with prescribed retries for reporting only."""
import json,importlib.util
import numpy as np
from pathlib import Path
from scripts.compositional_dynamics_v17 import viscous_recovery as recovery
from scripts.compositional_dynamics_v17 import viscous_development as original
ROOT,REPORT,sha,write=recovery.ROOT,recovery.REPORT,recovery.sha,recovery.write

def main():
 p=json.loads(recovery.PLAN.read_text());recovery.verify(p);old=json.loads(recovery.OLDRESULT.read_text());rp=REPORT/'viscous-recovery-results.json';new=json.loads(rp.read_text());mp=recovery.RAW/'runtime-input-freeze.json';manifest=json.loads(mp.read_text())
 if not new.get('terminal')or new['plan_sha256']!=sha(recovery.PLAN)or new['runtime_freeze_sha256']!=sha(mp):raise ValueError('Terminal pinned recovery required')
 if manifest['original_result_sha256']!=sha(recovery.OLDRESULT)or any(sha(ROOT/f)!=h for f,h in manifest['original_artifacts'].items()):raise ValueError('Original provenance changed')
 recovery.validate_original_assets(p,old);selected=[c['name']for c in recovery.select_cases(p,old)]
 if selected!=manifest['selected_names']or selected!=new['selected_names']:raise ValueError('Changed recovery selection')
 oldrows={r['name']:r for r in old['rows']};newrows={r['name']:r for r in new['rows']}
 if len(newrows)!=len(new['rows'])or set(newrows)-set(selected):raise ValueError('Undeclared/duplicate recovery records')
 merged=[];origins={};prefix_replay={}
 for c in p['cases']:
  name=c['name'];r=newrows.get(name)if name in selected else oldrows.get(name)
  if r is None:continue
  expected=sha(recovery.PLAN)if name in selected else p['recovery']['original_plan_sha256']
  if r['case']!=c or r['plan_sha256']!=expected:raise ValueError('Selected artifact identity mismatch')
  merged.append(r);origins[name]='recovery'if name in selected else'original'
  if name in selected and name in oldrows:
   oldfile=ROOT/oldrows[name]['path']/'trajectory.npz';newfile=ROOT/r['path']/'trajectory.npz'
   if oldfile.exists()and newfile.exists():
    with np.load(oldfile)as x,np.load(newfile)as y:
     values={}
     for key in ['times_s','u','velocities','surface_frames','sphere_centers_m','sphere_velocities_m_s','sphere_omega_rad_s','tool_wrenches_N_Nm','ledger']:
      if key not in x or key not in y:continue
      covered=y[key].shape[0]>=x[key].shape[0]and y[key].shape[1:]==x[key].shape[1:]
      values[key]=dict(prefix_covered=covered,exact_equal=bool(covered and np.array_equal(x[key],y[key][:len(x[key])])),max_abs_difference=float(np.max(abs(x[key]-y[key][:len(x[key])]),initial=0))if covered else None)
     prefix_replay[name]=values
 derived=REPORT/'viscous-recovery-combined-inputs.json';write(derived,dict(plan_sha256=sha(recovery.PLAN),terminal=True,rows=merged,source_results=dict(original=sha(recovery.OLDRESULT),recovery=sha(rp)),source_for_cases=origins,original_incomplete_names=selected,scope='Original completed jobs unchanged; recovery used only for originally incomplete jobs. Original failures retained separately.'))
 class ReportRoutes:
  def __truediv__(self,name):
   if name=='viscous-development-results.json':return derived
   if name=='viscous-development-analysis.json':return REPORT/'viscous-recovery-analysis.json'
   raise ValueError('Unexpected reporting path')
 class RawRoutes:
  def __truediv__(self,name):
   return ROOT/next(r['path']for r in merged if r['name']==name)
 src=ROOT/'scripts/compositional_dynamics_v17/viscous_analysis.py';spec=importlib.util.spec_from_file_location('private_recovery_reporting',src);impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl);impl.PLAN=recovery.PLAN;impl.REPORT=ReportRoutes();impl.RAW=RawRoutes();impl.main()
 out=REPORT/'viscous-recovery-analysis.json';report=json.loads(out.read_text());report['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(__file__)
 for f in [recovery.OLDRESULT,rp,mp]:report['input_sha256'][str(f.relative_to(ROOT))]=sha(f)
 report.update(restart_prefix_replay=prefix_replay,source_for_cases=origins,original_incomplete_names=selected,original_failures_preserved=True,recovery_scope='Same fixed physical model and integrator from initial state; original incomplete attempts preserved. No best-of numerical or outcome selection.');write(out,report)
if __name__=='__main__':main()
