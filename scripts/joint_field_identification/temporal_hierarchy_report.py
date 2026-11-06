"""Audit and summarize the complete conditional temporal-prior stress cohort."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
from .temporal_hierarchy import REPORT,MODELS,verify

def main(wait_seconds=0):
 deadline=time.time()+wait_seconds
 while not all(len(list((REPORT/s).glob('case-*.json')))==32 for s in ['block','step']):
  if time.time()>=deadline:raise RuntimeError('Full64case cohort required')
  time.sleep(15)
 protocol,sha=verify();docs=[]
 for shape in ['block','step']:
  for case in range(32):
   d=json.loads((REPORT/shape/f'case-{case:03d}.json').read_text());assert d['case']==case and d['shape']==shape and d['protocol_sha256']==sha
   assert len(d['rows'])==12 and len({(r['model'],r['condition']) for r in d['rows']})==12
   for r in d['rows']:
    for k,v in r.items():
     if isinstance(v,float):assert math.isfinite(v),(shape,case,k)
    assert abs(r['conditional_force_covariance_trace_N2']+r['between_component_force_covariance_trace_N2']-r['total_force_covariance_trace_N2'])<1e-12
    assert abs(sum(r['posterior_tau_weights'])-1)<1e-10
    if r['model']=='scale-correlation':assert abs(sum(r['correlation_posterior_weights'])-1)<1e-10
   docs.append(d)
 keys=['interval_width_mm','hidden_interval_width_mm','changed_interval_width_mm','rmse_mm','hidden_rmse_mm','changed_load_rmse_mm','material_relative_rmse','force_rmse_N','coverage90','hidden_coverage90','changed_coverage90','force_coverage90','force_interval_width_N','material_coverage90','material_interval_relative_width','support_center_error_mm','load_center_error_mm']
 def rows(model,condition):return [next(r for r in d['rows'] if r['model']==model and r['condition']==condition) for d in docs]
 means=[]
 for condition in ['initial','views','probes']:
  for model in MODELS:
   rr=rows(model,condition);entry=dict(model=model,condition=condition,n=64,**{key:float(np.mean([r[key] for r in rr])) for key in keys})
   if model=='scale-correlation':entry.update(true_length_posterior_mass=float(np.mean([r['true_length_posterior_mass'] for r in rr])))
   means.append(entry)
 rng=np.random.default_rng(260603);indices=np.concatenate([rng.integers(0,32,(2000,32)),32+rng.integers(0,32,(2000,32))],axis=1);comparisons=[]
 for condition in ['initial','views','probes']:
  ours=rows('scale-correlation',condition)
  for control in MODELS[1:]:
   other=rows(control,condition)
   for key in keys:
    delta=np.array([a[key]-b[key] for a,b in zip(ours,other)]);ci=np.quantile(delta[indices].mean(1),[.025,.975]);comparisons.append(dict(condition=condition,control=control,metric=key,extension_minus_control=float(delta.mean()),paired95=ci.tolist()))
 summary=dict(complete=True,cases=64,protocol_sha256=sha,means=means,comparisons=comparisons,scope=protocol['scope'])
 (REPORT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['Temporal correlation stress test:64fresh cases; true length0.10 fixed.','Conditional stress coverage is NOT matched-prior calibration.','model | evidence | hidden mm | new-load mm | material rel | force N | pred90 | force90 | forcewidth N']
 for r in means:lines.append(f"{r['model']} | {r['condition']} | {r['hidden_rmse_mm']:.6f} | {r['changed_load_rmse_mm']:.6f} | {r['material_relative_rmse']:.4f} | {r['force_rmse_N']:.5f} | {r['coverage90']:.3f} | {r['force_coverage90']:.3f} | {r['force_interval_width_N']:.5f}")
 lines+=['All cases retained. Same independent fixed4view and4probe branches. Full mixture intervals and evidence updates.','Normalized force-profile score remains a longitudinal basis proxy, not complete physical surface traction.']
 (REPORT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 (REPORT/'completion-audit.json').write_text(json.dumps(dict(passed=True,cases=64,rows=768,all_metrics_finite=True,covariance_decomposition_verified=True,hyper_weights_normalized=True,source_hashes_verified=True,protocol_sha256=sha,report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
 print('\n'.join(lines),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--wait-seconds',type=int,default=0);args=ap.parse_args();main(args.wait_seconds)
