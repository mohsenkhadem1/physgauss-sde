"""No actual model/data/draws: deterministic manufactured27/54dim cost only."""
import json,time,resource,os
from pathlib import Path
import numpy as np
from experiments import campaign_surrogate_sbc as s
coeff={}
for i,c in enumerate(s.CATEGORIES):
 a=np.zeros((3,27));a[1]=np.linspace(.0001,.001,27)*(1+.2*i);b=a.copy();b[2]=np.linspace(-.0001,.0001,27);t=np.zeros((3,189));t[1]=np.linspace(1e-12,.001,189);coeff[c]=dict(prefix=a,SIDE=b,target=t)
E=100000.;c=s.CATEGORIES[0];obs={k:s.evaluate_polynomial(coeff[c][k],E)+np.linspace(-1e-4,1e-4,27)for k in s.PHASES};truth=dict(young_Pa=E,support_category=c,target_displacement_m=s.evaluate_polynomial(coeff[c]['target'],E).reshape(63,3));timings={}
for phase in s.PHASES:
 start=time.perf_counter();r=s.infer(coeff,obs,phase);infer_s=time.perf_counter()-start
 start=time.perf_counter();reconstructed,posts=s.compile_inference(coeff,obs,phase);reconstruct_s=time.perf_counter()-start
 start=time.perf_counter();scored=s.score_laws(coeff,obs,phase,truth,np.full(28,.3),posts=posts);PIT_s=time.perf_counter()-start
 assert r['completed']and reconstructed['numerical_controls_passed']and scored['numerical_controls_passed']
 timings[phase]=dict(inference_s=infer_s,pre_truth_reconstruction_s=reconstruct_s,private_PIT_s=PIT_s)
result=dict(completed=True,scope='DETERMINISTIC_MANUFACTURED_ONLY_NO_CAMPAIGN_MODEL_DATA_OR_DRAWS',timings=timings,pid=os.getpid(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,phases=2,integration_methods=3,quantities_per_phase=28,coordinate_quantile_roots=0,material_quantile_roots=0,scalar_right_or_material_CDF_queries=168,additional_coordinate_left_queries=162)
p=Path(__file__).with_name('manufactured-cost-receipt.json')
with p.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
