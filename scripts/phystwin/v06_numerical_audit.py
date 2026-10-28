"""Post-evaluation refinement with saved actions fixed; never retunes a policy."""
from pathlib import Path
import argparse,gc,itertools,json,subprocess,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.sequential_policy import Experiment,interpolators,fit,observations,verify,Z90
from scripts.phystwin.sequential_reference import BASE,write,protocol

def audit(mode,size):
 directory=BASE/('analysis' if mode=='direct' else 'analysis-rgbd');seal=json.loads((directory/'seal.json').read_text());verify(seal)
 interp=interpolators();bias,C,_=fit(interp);sigma=protocol()['noise_sigma_m'] if mode=='direct' else .0004
 if mode=='direct':y,truth=observations('test')
 else:
  with np.load(BASE/'vision/test.npz') as z:y=z['measurements'];truth=z['truth']
 old={}
 for policy in ['fixed','random','heuristic','adaptive']:
  with np.load(directory/f'test-{policy}.npz') as z:old[policy]={k:z[k] for k in z.files}
 ex=Experiment(interp,size,bias,C,sigma);ex.cache_limit=2;groups={}
 for policy,data in old.items():
  for i,choices in enumerate(data['choices']):
   for budget in range(3):groups.setdefault(tuple(sorted(choices[:budget])),set()).add(i)
 pred={}
 for history,indices in groups.items():
  model=ex.model(history);indices=sorted(indices)
  for start in range(0,len(indices),4):
   rows=indices[start:start+4];result=model.infer(y[np.ix_(rows,ex.observed(history))])
   for j,i in enumerate(rows):pred[history,i]={k:v[j] for k,v in result.items()}
  print('REFINE',mode,size,list(history),len(indices),flush=True)
 result={}
 for policy,data in old.items():
  mean=[];sd=[];q=[]
  for i,choices in enumerate(data['choices']):
   values=[pred[tuple(sorted(choices[:budget])),i] for budget in range(3)];mean.append([v['mean'] for v in values]);sd.append([v['sd'] for v in values]);q.append([v['log_parameter_quantiles'] for v in values])
  mean=np.array(mean);sd=np.array(sd);q=np.array(q);delta=mean-data['mean'];err=(mean-truth[:,None])**2
  result[policy]=dict(rmse_mm=(1000*np.sqrt(err.mean((0,2)))).tolist(),maximum_mean_change_mm=float(1000*abs(delta).max()),rms_mean_change_mm=float(1000*np.sqrt(np.mean(delta**2))),maximum_sd_change_mm=float(1000*abs(sd-data['sd']).max()),maximum_log_quantile_change=float(abs(q-data['quantiles']).max()),object_mse=err.mean(2).reshape(-1,protocol()['noise_repeats'],3).mean(1).tolist())
  np.savez_compressed(directory/f'test-{policy}-audit{size}.npz',mean=mean,sd=sd,quantiles=q,choices=data['choices'],truth=truth)
 write(directory/f'posthoc-refinement-{size}.json',dict(scope='Diagnostic only. Saved test actions, calibration multiplier, priors, likelihood and discrepancy remain unchanged. These refined predictions do not select or rerun actions.',grid_size=size,policies=result));verify(seal)
 print('AUDIT_COMPLETE',mode,{k:v['rmse_mm'] for k,v in result.items()},flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--wait-unit');p.add_argument('--size',type=int,default=97);a=p.parse_args()
 if a.wait_unit:
  while subprocess.run(['systemctl','--user','is-active','--quiet',a.wait_unit]).returncode==0:time.sleep(10)
 for mode in ['direct','rgbd']:audit(mode,a.size);gc.collect()
if __name__=='__main__':main()
