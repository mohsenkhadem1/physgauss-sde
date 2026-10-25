"""Terminal metadata/scalar audit; no GPU, model, NPZ or reference replay."""
import argparse
import numpy as np
from experiments import run_layout_grouped_gpu_parity_v01 as run
ROOT=run.ROOT;LOCAL=run.LOCAL
SOURCE='experiments/audit_layout_grouped_gpu_parity_v01.py';TEST='tests/test_audit_layout_grouped_gpu_parity_v01.py'
EXPECTED_PROTOCOL='1b041ee35d5d314d1b0c5c9840e97d17aaf22242f2f7188c711e33722735fb8c'
def need(v,message):
    if not v:raise ValueError(message)
def le(v,b):return isinstance(v,(int,float))and not isinstance(v,bool)and np.isfinite(v)and 0<=v<=b
def check_device(row,value,p,saved):
    tol=p['tolerances'];need(value.get('completed')is True and value.get('qualified')is True,'Completed qualified device')
    need(value['slot']==row['slot']and value['pid']==row['pid']and value['physical_UUID']==row['physical_UUID']==p['devices'][row['slot']],'Fixed slot/PID/UUID')
    need(value['protocol_SHA256']==EXPECTED_PROTOCOL and value['release']==run.record(LOCAL/'release.json'),'Device protocol/release')
    need(all(value.get(k)is True for k in('oneaction_only','all8_ranks_pending','no_new_RNG','no_new_observation','no_reference_requalification')),'Same one-action scope')
    b=value['bank_replay_errors'];need(le(b['logZ'],tol['bank_logZ'])and le(b['layout_probability'],tol['bank_layout_probability']),'Same saved decision bank')
    c=value['conditional_parity'];bound=tol['risk_absolute_m2']+tol['risk_relative']*saved[0]['prior_target_variance'];need(c['passed']is True and le(c['risk_error_m2'],bound)and le(c['log_density_error'],tol['conditional_log_density']),'Conditional centered risk/logdensity parity')
    levels=value['levels'];need([r['order']for r in levels]==[8,16]and len(levels)==2,'Both fixed outcome orders')
    checks=[]
    for s,r in zip(saved,levels):
        check=run.compare(s,r['result'],tol);need(check==r['parity']and check['passed']is True,'Recomputed saved CPU parity');checks.append(check)
    need(le(value['torch_peak_allocated_bytes'],p['resources']['torch_allocated_bytes'])and le(value['peak_RSS_bytes'],p['resources']['aggregate_host_RSS_bytes']),'Device memory limits')
    need(le(value['wall_s'],p['resources']['worker_wall_s'])and all(le(value['utility_wall_s'][str(n)],value['wall_s'])for n in(8,16)),'Measured bounded device timings')
    return checks

def audit():
    output=LOCAL/'independent-terminal-review.json';need(not output.exists(),'No audit overwrite')
    p=run.validate(EXPECTED_PROTOCOL);e=run.read(LOCAL/'execution.json')
    need(e.get('completed')is True and e.get('qualified')is True and e.get('terminal')is True and'failure'not in e,'Clean complete controller')
    need(e['protocol_SHA256']==EXPECTED_PROTOCOL and e['release']==run.record(LOCAL/'release.json')and e.get('all3_devices_retained')is True and e.get('all8_ranks_pending')is True,'Controller commitments')
    need(le(e['wall_s'],p['resources']['global_wall_s']),'Global wall cap')
    rows=e['devices'];need(len(rows)==3 and[r['slot']for r in rows]==[0,1,2]and[r['physical_UUID']for r in rows]==p['devices']and len({r['pid']for r in rows})==3,'All three unique owned devices')
    absent=run.old.old.mechanical_screen.shared.old.absent;absent(e['controller_pid']);saved=run.read(ROOT/p['origin']['utility']['path'])['levels'];values=[];checks=[]
    for row in rows:
        need(row.get('exit_code')==0 and row.get('cleanup_exit_code')==0 and row.get('reaped')is True and row['status']=='qualified','Every worker exit/reap')
        absent(row['pid']);path=ROOT/row['result']['path'];need(run.record(path)==row['result']and path==LOCAL/f"worker-{row['slot']}/result-LOCAL.json",'Exact worker result bytes/path')
        value=run.read(path);checks.extend(check_device(row,value,p,saved));values.append(value)
        need(le(row['peak_host_RSS_bytes'],p['resources']['aggregate_host_RSS_bytes'])and le(row['peak_owned_VRAM_bytes'],p['resources']['owned_VRAM_bytes']),'External owned memory caps')
    summary=dict(integrity_passed=True,completed=True,all3_GPU_parity_qualified=True,all8_ranks_pending=True,policy_use_blocked=True,oneaction_only=True,no_model_GPU_or_array_replay=True,wall_s=e['wall_s'],maximum_risk_error_m2=max(c['risk_error_m2']for c in checks),maximum_prior_variance_error_m2=max(c['prior_variance_error_m2']for c in checks),maximum_predictive_mass_error=max(c['mass_error']for c in checks),maximum_conditional_log_density_error=max(v['conditional_parity']['log_density_error']for v in values),maximum_torch_allocated_bytes=max(v['torch_peak_allocated_bytes']for v in values),maximum_owned_VRAM_bytes=max(r['peak_owned_VRAM_bytes']for r in rows),maximum_host_RSS_bytes=max(r['peak_host_RSS_bytes']for r in rows),timing_ranges_s={key:[min(v[key]for v in values),max(v[key]for v in values)]for key in('model_and_bank_replay_s','GPU_initialization_s','wall_s')},utility_timing_ranges_s={str(n):[min(v['utility_wall_s'][str(n)]for v in values),max(v['utility_wall_s'][str(n)]for v in values)]for n in(8,16)},scope='Same qualified saved LINEAR one-action bank only; no nonlinear, all-action ranking or active advantage admission')
    run.write(output,dict(summary=summary,source_SHA256={s:run.sha(ROOT/s)for s in(SOURCE,TEST)},protocol=run.record(LOCAL/'protocol.json'),release=run.record(LOCAL/'release.json'),execution=run.record(LOCAL/'execution.json'),worker_results=[r['result']for r in rows]));return summary

def inventory():
    review=run.read(LOCAL/'independent-terminal-review.json');s=review['summary'];need(s['integrity_passed']and s['all3_GPU_parity_qualified'],'Audited qualification')
    report=ROOT/'reports/layout-grouped-gpu-parity-v01';report.mkdir(parents=True,exist_ok=True)
    files=[]
    for path,data in((report/'summary.json',s),(report/'source-provenance.json',dict(protocol=review['protocol'],release=review['release'],audit_source_SHA256=review['source_SHA256'],worker_result_SHA256=[v['sha256']for v in review['worker_results']]))):
        need(not path.exists(),'No public report overwrite');run.write(path,data);files.append(run.record(path))
    files.extend(run.record(ROOT/v)for v in(SOURCE,TEST));path=LOCAL/'publication-inventory-completed.json';need(not path.exists(),'No inventory overwrite');run.write(path,dict(files=files,raw_arrays_observations_risks_or_category_probabilities_included=False));return run.record(path)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['audit','inventory']);a=ap.parse_args();print(audit()if a.action=='audit'else inventory())
