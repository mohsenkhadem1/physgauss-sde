"""Observed-prefix and known-command diagnostics only, across fixed PhysTwin cases."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.prefix_ridge_switch_v13.inputs import load_payload
from scripts.prefix_ridge_switch_v13.selector import ridge
from scripts.external_benchmark.evaluate import features
from scripts.contact_benchmark_v12.object_frame import _observed_prediction,adapt
from scripts.benchmark_improvement_v14.phystwin_production import NAMES,declared
from scripts.benchmark_improvement_v14.phystwin_gaussian_production import anchor_folder
from scripts.benchmark_improvement_v14.common import REPORT,read,write,sha

def err(d,p,start,stop):
 y=_observed_prediction(d,p[start:stop]);truth=d['observed'][start:stop];valid=d['observed_mask'][start:stop]&np.isfinite(truth).all(2)&np.isfinite(y).all(2)
 return float(np.sqrt(np.mean((y-truth)[valid]**2))*1000) if valid.any() else None

def leverage(d,end):
 F=features(d);mu=F[:end].mean(0);sd=F[:end].std(0);sd[sd<1e-5]=1;X=np.c_[np.ones(len(F)),(F-mu)/sd];M=np.linalg.inv(X[:end].T@X[:end]+.1*np.eye(X.shape[1]));h=np.einsum('ti,ij,tj->t',X,M,X)
 return dict(prefix_median=float(np.median(h[:end])),future_median=float(np.median(h[end:])),future_max=float(np.max(h[end:])),future_to_prefix_median_ratio=float(np.median(h[end:])/np.median(h[:end])))

def main():
 rows=[]
 for name in NAMES:
  c=declared(name);folder=ROOT/'runs/prefix-ridge-switch-v13/inputs'/name;d=load_payload(folder/'full.npz');s=load_payload(folder/'short.npz');end=d['train_end'];cut=s['train_end']
  for k in ('times','controls','controller_targets','attachment_active'):s[k]=s[k][:end]
  raw,_=ridge(d);sr,_=ridge(s)
  with np.load(anchor_folder(c)/'prediction.npz') as f:base=f['predicted']
  oldmeta=read(ROOT/'runs/contact-benchmark-v12/evaluation'/c['cohort']/name/'prediction-metadata.json')['object_frame']
  original_validation=next((r['validation_rmse_m']*1000 for r in oldmeta.get('validation',[]) if (r['method'],r['strength'])==(oldmeta['method'],oldmeta.get('strength',0))),None)
  row=dict(name=name,full_end=end,short_end=cut,original_SHORT_mm=original_validation,raw_nodal_ridge_SHORT_mm=err(d,sr,cut,end),last_prefix_ridge_mm=err(d,raw,end-1,end),last_prefix_FEM_mm=err(d,base,end-1,end),short_validation_command_leverage=leverage(s,cut),full_future_command_leverage=leverage(d,end));rows.append(row);print(json.dumps(row),flush=True)
 write(REPORT/'phystwin-prefix-reliability-diagnosis.json',dict(scope='Only observed object prefixes and known controller commands; no future object target/scoring reads',cases=rows,source_sha256=sha(Path(__file__))))
if __name__=='__main__':main()
