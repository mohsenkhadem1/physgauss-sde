"""Refine saved v08 predictions to 97^3 per layout without changing actions/scales."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.v08_study import BASE,OUT,Experiment,interpolators,fit,POLICIES,write
from scripts.phystwin.sequential_policy import verify
import fcntl
_audit_lock=(BASE/'prediction-audit.lock').open('a');fcntl.flock(_audit_lock,fcntl.LOCK_EX)
seal=json.loads((BASE/'seal.json').read_text());verify(seal)
if (OUT/'prediction-refinement.json').exists():
 print('AUDIT_ALREADY_COMPLETE',flush=True);sys.exit(0)
interp=interpolators();bias,C=fit(interp);ex=Experiment(interp,97,bias,C);ex.cache_limit=2;z=np.load(BASE/'test-observations.npz');y=z['measurements'];summary={}
for policy in POLICIES:
 dest=BASE/f'test-{policy}-97.npz'
 if not dest.exists():
  plans=json.loads((BASE/'plans'/f'test-{policy}.json').read_text());mean=np.zeros((len(y),3,243));sd=mean.copy();groups={}
  for i,p in enumerate(plans):
   for b in range(3):groups.setdefault(tuple(sorted(p['history'][:b])),[]).append((i,b))
  for history,rows in groups.items():
   for start in range(0,len(rows),2):
    batch=rows[start:start+2];pred=ex.model(history).infer(y[[i for i,b in batch]][:,ex.observed(history)])
    for j,(i,b) in enumerate(batch):mean[i,b]=pred['mean'][j];sd[i,b]=pred['sd'][j]
  np.savez_compressed(dest,mean=mean,sd=sd,truth=z['truth'])
 fine=np.load(dest);coarse=np.load(BASE/f'test-{policy}-65.npz');summary[policy]=dict(max_mean_change_mm=float(1000*abs(fine['mean']-coarse['mean']).max()),rms_mean_change_mm=float(1000*np.sqrt(np.mean((fine['mean']-coarse['mean'])**2))),max_sd_change_mm=float(1000*abs(fine['sd']-coarse['sd']).max()),refined_rmse_mm=(1000*np.sqrt(np.mean((fine['mean']-fine['truth'][:,None])**2,axis=(0,2)))).tolist(),refined_object_coverage=(abs(fine['mean']-fine['truth'][:,None])<=np.array(seal['scales'][policy])[None,:,None]*1.644853626951*fine['sd']).reshape(24,3,3,243).all((1,3)).mean(0).tolist());print('AUDIT97',policy,summary[policy],flush=True)
write(OUT/'prediction-refinement.json',dict(scope='Post-hoc saved-history prediction audit, 65 to 97 cubed per layout. Calibration multipliers and selected actions unchanged; primary results stay on 65. No policy retuning.',policies=summary));verify(seal)
