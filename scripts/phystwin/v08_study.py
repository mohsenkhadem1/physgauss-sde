"""Stronger-load acquisition study and exhaustive development action diagnosis."""
from pathlib import Path
import argparse, hashlib, itertools, json, sys, gc
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scipy.interpolate import RegularGridInterpolator, BSpline
from scipy.sparse.linalg import spsolve
from scipy.special import ndtri
from scipy.stats import qmc
from scripts.phystwin import v08_reference as ref
from scripts.phystwin import sequential_policy as old
from scripts.phystwin.sequential_fixture import LAYOUTS,points
from scripts.phystwin.v06_camera import render,track
ROOT=ref.ROOT; SOURCE=ref.BASE; BASE=ref.BASE; OUT=ref.OUT
write=ref.write

protocol=ref.protocol

def indices():
 ix=old.field_indices();return np.r_[ix[:27],np.arange(216,243),ix[27:]]

def observations(splits=None):
 spec=protocol();rest=points();ix=old.field_indices()
 for split in (splits or ['development','calibration','test']):
  path=BASE/f'{split}-observations.npz'
  if path.exists():continue
  rng=np.random.default_rng(dict(development=1810,calibration=1811,test=1812)[split]);rows=[];truth=[];exact=[]
  for obj in range(len(spec[split])):
   field=np.load(BASE/'fields'/f'{split}-{obj}-reference.npz')['field'];ims=[]
   for a in range(8):
    ids=(np.arange(72,81) if a==7 else (ix[:9]//3)[::3]) if a in [0,7] else ((ix[9+3*(a-1):12+3*(a-1)]%243)//3)[::3]
    side=(0 if a==0 else 1) if a in [0,7] else (-1 if rest[ids[0],0]==0 else 1)
    rgb,depth,cam=render(rest+field[0 if a==7 else a],side,ids);ims.append((rgb,depth,cam,ids))
   selected=field.ravel()[indices()]
   for repeat in range(spec['noise_repeats']):
    rows.append(np.concatenate([(track(rgb,dep,cam,ids,rng)-rest[ids]).ravel() for rgb,dep,cam,ids in ims]));truth.append(selected[54:]);exact.append(selected[:54])
  np.savez_compressed(path,measurements=rows,truth=truth,exact_observations=exact)
  print('OBSERVATIONS',split,flush=True)

def interpolators():
 axes=[np.linspace(*b,4) for b in ref.BOUNDS];ix=indices();result=[]
 for layout in LAYOUTS:
  data=np.stack([np.load(SOURCE/'fields'/f'bank-{layout}-{i}-{j}-{k}-n4-q3.npz')['field'].ravel()[ix] for i,j,k in itertools.product(range(4),repeat=3)])
  result.append(RegularGridInterpolator(axes,data.reshape(4,4,4,-1),method='cubic',solver=spsolve))
 return result

def fit(interp):
 spec=protocol();z=np.load(BASE/'development-observations.npz');pred=np.repeat(np.array([interp[LAYOUTS.index(r['layout'])](np.log(r['parameters'])[None])[0] for r in spec['development']]),3,axis=0)
 residual=np.c_[z['measurements'],z['truth']]-pred;bias=residual.mean(0);emp=np.cov(residual,rowvar=False)
 C=.5*emp+.5*np.diag(np.diag(emp))+np.eye(297)*1e-12
 # Observation covariance includes actual camera quantization and depth errors;
 # target covariance includes forward error only. This is a development fit.
 return bias,C

class Experiment(old.Experiment):
 def __init__(self,interp,size,bias,C,device='cuda:0'):
  import torch
  self.size=size;self.bias=bias;self.C=C;self.device=device;self.target=np.arange(54,297);self.cache={};self.cache_limit=32 if size<=33 else 4
  axes=[np.linspace(*b,size) for b in ref.BOUNDS];nodes=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
  self.fields=np.concatenate([f(nodes) for f in interp]);w=np.ones((size,)*3)
  for a in range(3):
   ix=[slice(None)]*3;ix[a]=[0,-1];w[tuple(ix)]*=.5
  self.prior=np.tile(w.ravel()/w.sum()/3,3);self.shared=torch.as_tensor(self.fields,dtype=torch.float64,device=device)
  self.bases=[BSpline.design_matrix(a,t,k).toarray() for a,t,k in zip(axes,interp[0]._spline.t,interp[0]._spline.k)];self.coefficients=np.stack([f._spline.c for f in interp]).reshape(-1,297)
 def action_indices(self,a):return np.arange(27,54) if a==6 else np.arange(9+3*a,12+3*a)
 def observed(self,history):return np.r_[np.arange(9),*[self.action_indices(a) for a in sorted(history)]]
 def choose(self,history,y,samples=128,allow_view=True):
  obs=self.observed(history);m=self.model(history);w=m.weights(y[None,obs]).cpu().numpy()[0];risks=np.full(7,np.inf)
  u=qmc.Sobol(28,scramble=True,seed=2731+len(history)).random_base2(int(np.log2(samples)));idx=np.minimum(np.searchsorted(np.cumsum(w),u[:,0]),len(w)-1)
  for a in range(7 if allow_view else 6):
   if a in history:continue
   future=self.action_indices(a);gain=np.linalg.solve(self.C[np.ix_(obs,obs)],self.C[np.ix_(obs,future)]).T
   mu=self.fields[idx[:,None],future]+self.bias[future]+(y[obs]-self.fields[idx[:,None],obs]-self.bias[obs])@gain.T
   cov=self.C[np.ix_(future,future)]-gain@self.C[np.ix_(obs,future)];full=np.broadcast_to(y,(samples,54)).copy();full[:,future]=mu+ndtri(u[:,1:1+len(future)])@np.linalg.cholesky(cov).T
   risks[a]=self.model([*history,a]).risk(full[:,self.observed([*history,a])]).mean()
  current=float(m.risk(y[None,obs])[0]);return risks,current

def select(risks,current,costs,penalty):
 score=np.asarray(risks)+penalty*np.asarray(costs);a=int(score.argmin());return -1 if current<=score[a] else a

POLICIES=['fixed','random','adaptive_pokes','cost_aware']
def plan(split,device):
 spec=protocol();interp=interpolators();bias,C=fit(interp);ex=Experiment(interp,spec['planning_grid'],bias,C,device);z=np.load(BASE/f'{split}-observations.npz')
 risk_cache={}
 for policy in POLICIES:
  dest=BASE/'plans'/f'{split}-{policy}.json'
  if dest.exists():continue
  rows=[]
  for i,y in enumerate(z['measurements']):
   checkpoint=BASE/'plans'/f'{split}-{policy}-{i}.json'
   if checkpoint.exists():rows.append(json.loads(checkpoint.read_text()));continue
   history=[];decisions=[]
   for step in range(2):
    if policy=='fixed':a=spec['fixed_sequence'][step]
    elif policy=='random':a=int(np.random.default_rng(2901+i*2+step).choice([a for a in range(6) if a not in history]))
    else:
     key=(i,tuple(sorted(history)))
     if key not in risk_cache:risk_cache[key]=ex.choose(history,y,spec['utility_samples'],True)
     risks,current=risk_cache[key];risks=risks.copy()
     if policy=='adaptive_pokes':risks[6]=np.inf
     a=select(risks,current,spec['costs'],spec['cost_penalty_m2']) if policy=='cost_aware' else int(risks.argmin())
     decisions.append(dict(current_variance_m2=current,expected_variances_m2=[None if not np.isfinite(v) else float(v) for v in risks],chosen=a))
    if a<0:break
    history.append(a)
   row=dict(history=history,decisions=decisions);rows.append(row);write(checkpoint,row)
   if i%6==0:print('PLAN',split,policy,i,flush=True)
  write(dest,rows)

def predict(split,device,size=65):
 spec=protocol();interp=interpolators();bias,C=fit(interp);ex=Experiment(interp,size,bias,C,device);z=np.load(BASE/f'{split}-observations.npz');y=z['measurements']
 for policy in POLICIES:
  dest=BASE/f'{split}-{policy}-{size}.npz'
  if dest.exists():continue
  plans=json.loads((BASE/'plans'/f'{split}-{policy}.json').read_text());means=np.zeros((len(y),3,243));sds=means.copy();history_rows={}
  for i,p in enumerate(plans):
   for budget in range(3):history_rows.setdefault(tuple(sorted(p['history'][:budget])),[]).append((i,budget))
  for history,rows in history_rows.items():
   for start in range(0,len(rows),4):
    batch=rows[start:start+4];idx=[i for i,b in batch];pred=ex.model(history).infer(y[idx][:,ex.observed(history)])
    for j,(i,b) in enumerate(batch):means[i,b]=pred['mean'][j];sds[i,b]=pred['sd'][j]
  np.savez_compressed(dest,mean=means,sd=sds,truth=z['truth']);print('PREDICT',split,policy,size,flush=True)

def seal():
 paths=[BASE/'protocol.json',BASE/'development-observations.npz',BASE/'calibration-observations.npz',Path(__file__),ROOT/'scripts/phystwin/sequential_policy.py']+sorted((BASE/'plans').glob('calibration-*.json'))+sorted(BASE.glob('calibration-*-65.npz'))
 paths += sorted((SOURCE/'fields').glob('bank*-n4-q3.npz')) + sorted((BASE/'fields').glob('development-*-reference.npz')) + sorted((BASE/'fields').glob('calibration-*-reference.npz'))
 paths += [ROOT/'scripts/phystwin'/name for name in ['v08_reference.py','v08_physbridge.py','run_v08_phystwin.py','v08_ensemble.py','sequential_fixture.py','sequential_reference.py','consistent_fixture.py','tensor_fixture.py','v06_camera.py','active_support_policy.py','uncertainty_posterior.py']]
 scales={}
 for p in POLICIES:
  z=np.load(BASE/f'calibration-{p}-65.npz');scores=(abs(z['mean']-z['truth'][:,None])/np.maximum(1.644853626951*z['sd'],1e-12)).reshape(24,3,3,243).max((1,3));rank=int(np.ceil(.9*25))-1;scales[p]=np.maximum(1,np.sort(scores,axis=0)[rank]).tolist()
 value=dict(hashes={str(p.relative_to(ROOT)):old.sha(p) for p in paths},scales=scales,method='Separate object-max split-conformal scale for each policy and budget; 24 calibration objects, 3 repeats, 243 coordinates; rank 23/24 for nominal 90%. Policy fixed before calibration. No uniform per-layout or physical-parameter guarantee.')
 path=BASE/'seal.json'
 if path.exists():assert json.loads(path.read_text())==value,'Refusing to replace existing seal'
 else:write(path,value)

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','observations','plan','predict','seal']);p.add_argument('--worker',type=int,default=0);p.add_argument('--split',choices=['development','calibration','test'],default='calibration');p.add_argument('--device',default='cuda:0');p.add_argument('--size',type=int,default=65);a=p.parse_args();protocol()
 if a.stage=='observations':observations([a.split])
 elif a.stage=='plan':
  if a.split=='test':old.verify(json.loads((BASE/'seal.json').read_text()))
  plan(a.split,a.device)
 elif a.stage=='predict':
  if a.split=='test':old.verify(json.loads((BASE/'seal.json').read_text()))
  predict(a.split,a.device,a.size)
 elif a.stage=='seal':seal()
if __name__=='__main__':main()
