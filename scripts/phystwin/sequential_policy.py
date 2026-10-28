"""Continuous three-parameter posterior and two-step reset probing, v06.

Uses exact mixture expected posterior variance. A cubic tensor contraction speeds
means without deleting hypotheses. No true layout or test target enters choices.
"""
from pathlib import Path
import argparse,functools,gc,hashlib,itertools,json,sys,time
import numpy as np
from scipy.interpolate import RegularGridInterpolator,BSpline
from scipy.sparse.linalg import spsolve
from scipy.special import ndtri
from scipy.stats import qmc
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.sequential_reference import BASE,ROOT,OUT,BOUNDS,protocol,write
from scripts.phystwin.sequential_fixture import points,LAYOUTS
from scripts.phystwin.active_support_policy import Mixture
from scripts.phystwin.uncertainty_posterior import marginal_quantiles,conditional_components,Z90
ANALYSIS=BASE/'analysis'

def field_indices():
 x=points();initial=[np.argmin(np.linalg.norm(x-p,axis=1)) for p in [[.12,-.04,z] for z in [-.04,0,.04]]]
 probes=[[0,-.04,-.04],[.24,-.04,.04],[0,-.04,.04],[.24,-.04,-.04],[0,-.04,.04],[.24,-.04,0]]
 initial=(np.array(initial)[:,None]*3+np.arange(3)).ravel()
 return np.r_[initial,*[243*(i+1)+3*np.argmin(np.linalg.norm(x-p,axis=1))+np.arange(3) for i,p in enumerate(probes)],np.arange(1701,1944)]

def interpolators(mesh=4):
 size=protocol()['bank_grid'];axes=[np.linspace(*b,size) for b in BOUNDS];ix=field_indices();result=[]
 for layout in LAYOUTS:
  data=np.stack([np.load(BASE/'fields'/f'bank-{layout}-{i}-{j}-{k}-n{mesh}-q3.npz')['field'].ravel()[ix] for i,j,k in itertools.product(range(size),repeat=3)])
  result.append(RegularGridInterpolator(axes,data.reshape(size,size,size,-1),method='cubic',solver=spsolve))
 return result

def fit(interp):
 spec=protocol();truth=[];pred=[];coarse=[];low=interpolators(2)
 for i,r in enumerate(spec['development']):
  l=LAYOUTS.index(r['layout']);theta=np.log(r['parameters'])[None]
  truth.append(np.load(BASE/'fields'/f'development-{i}-reference.npz')['field'].ravel()[field_indices()]);pred.append(interp[l](theta)[0]);coarse.append(low[l](theta)[0])
 truth=np.array(truth);pred=np.array(pred);residual=truth-pred;bias=residual.mean(0);emp=np.cov(residual,rowvar=False)
 C=.5*emp+.5*np.diag(np.diag(emp))+np.eye(emp.shape[0])*1e-12
 diagnostic=dict(development_target_rms_mm=(1000*np.sqrt(np.mean(residual[:,27:]**2,axis=1))).tolist(),bank_mesh_target_change_mm=(1000*np.sqrt(np.mean((pred[:,27:]-np.array(coarse)[:,27:])**2,axis=1))).tolist(),discrepancy='Development-only mean plus 50% empirical/50% diagonal covariance and 1um floor. Six development objects; calibration separately audits predictive bands.')
 return bias,C,diagnostic

class Model(Mixture):
 def __init__(self,fields,prior,bias,C,observed,target,sigma,device,shared=None):
  import torch
  self.torch=torch;self.device=device;self.prior=prior;self.tensor_basis=None;self.weight_batch_size=max(1,(3*2**24)//len(prior))
  self.observed=observed;self.target=target;self.shared=self.tensor(fields) if shared is None else shared
  S=C[np.ix_(observed,observed)]+np.eye(len(observed))*sigma**2
  gain=np.linalg.solve(S,C[np.ix_(observed,target)]).T;self.chol=np.linalg.cholesky(S);self.gain=self.tensor(gain)
  self.bias_obs=self.tensor(bias[observed]);self.bias_target=self.tensor(bias[target]);self.f=(self.shared[:,observed]+self.bias_obs)@self.tensor(np.linalg.inv(self.chol).T)
  self.normf=(self.f*self.f).sum(1);self.logprior=self.tensor(np.log(prior));self.V=self.tensor(np.maximum(np.diag(C[np.ix_(target,target)]-gain@C[np.ix_(observed,target)]),0))
  self.normM=torch.cat([(v*v).mean(1) for v in self.adjusted_chunks()])
 def adjusted_chunks(self):
  for part in self.shared.split(self.moment_chunk_size):
   yield part[:,self.target]+self.bias_target-(part[:,self.observed]+self.bias_obs)@self.gain.T
 @property
 def M(self):return self.torch.cat(list(self.adjusted_chunks()))
 def target_mean(self,p):
  if self.tensor_basis is None:return p@self.M
  bases,coef,offset,size=self.tensor_basis
  mass=p.reshape(len(p),3,size,size,size)
  mass=self.torch.einsum('nlijk,ia->nlajk',mass,bases[0])
  mass=self.torch.einsum('nlajk,jb->nlabk',mass,bases[1])
  mass=self.torch.einsum('nlabk,kc->nlabc',mass,bases[2])
  return mass.reshape(len(p),-1)@coef+p.sum(1,keepdim=True)*offset
 def infer(self,y):
  p=self.weights(y);m=self.target_mean(p);second=self.torch.zeros_like(m)
  start=0
  for v in self.adjusted_chunks():
   second+=p[:,start:start+len(v)]@(v*v);start+=len(v)
  sd=self.torch.sqrt(self.torch.clamp(second-m*m+self.V,min=1e-24));mean=m+self.tensor(y)@self.gain.T
  w=p.reshape(len(y),3,self.size,self.size,self.size);marginal=w.sum(1);quant=[]
  for j in range(3):
   axes=tuple(k+1 for k in range(3) if k!=j);mass=marginal.sum(axes).cpu().numpy();quant.append(marginal_quantiles(mass,np.linspace(*BOUNDS[j],self.size)))
  return dict(mean=mean.cpu().numpy(),sd=sd.cpu().numpy(),layout_probability=w.sum((2,3,4)).cpu().numpy(),log_parameter_quantiles=np.stack(quant,axis=1))

class Experiment:
 def __init__(self,interp,size,bias,C,sigma,device='cuda:0'):
  self.size=size;self.bias=bias;self.C=C.copy();self.C[:27,:27]+=np.eye(27)*sigma**2;self.device=device
  axes=[np.linspace(*b,size) for b in BOUNDS];nodes=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
  self.fields=np.concatenate([f(nodes) for f in interp]);w=np.ones((size,size,size))
  for axis in range(3):
   ix=[slice(None)]*3;ix[axis]=[0,-1];w[tuple(ix)]*=.5
  self.prior=np.tile(w.ravel()/w.sum()/3,3);self.target=np.arange(27,270);self.cache={};self.cache_limit=22
  import torch
  self.shared=torch.as_tensor(self.fields,dtype=torch.float64,device=device)
  self.bases=[BSpline.design_matrix(a,t,k).toarray() for a,t,k in zip(axes,interp[0]._spline.t,interp[0]._spline.k)]
  self.coefficients=np.stack([f._spline.c for f in interp]).reshape(-1,270)
 def observed(self,history):return np.r_[np.arange(9),*[np.arange(9+3*a,12+3*a) for a in sorted(history)]]
 def model(self,history):
  key=tuple(sorted(history))
  if key in self.cache:return self.cache[key]
  if len(self.cache)>=self.cache_limit:self.cache.pop(next(iter(self.cache)))
  obs=self.observed(key);m=Model(self.fields,self.prior,self.bias,self.C,obs,self.target,0.,self.device,self.shared);m.size=self.size
  gain=m.gain.cpu().numpy();coef=self.coefficients[:,self.target]-self.coefficients[:,obs]@gain.T;offset=self.bias[self.target]-self.bias[obs]@gain.T
  m.tensor_basis=([m.tensor(b) for b in self.bases],m.tensor(coef),m.tensor(offset),self.size);self.cache[key]=m;return m
 def predict(self,history,measurement):return self.model(history).infer(measurement[None,self.observed(history)])
 def choose(self,history,measurement,samples=64):
  obs=self.observed(history);y=measurement[obs];m=self.model(history);weights=m.weights(y[None]).cpu().numpy()[0]
  u=qmc.Sobol(4,scramble=True,seed=731+len(history)).random_base2(int(np.log2(samples)));idx=np.minimum(np.searchsorted(np.cumsum(weights),u[:,0]),len(weights)-1);noise=ndtri(u[:,1:]);risks=np.full(6,np.inf)
  for action in range(6):
   if action in history:continue
   future=np.arange(9+action*3,12+action*3);S=self.C[np.ix_(obs,obs)];gain=np.linalg.solve(S,self.C[np.ix_(obs,future)]).T
   mean=self.fields[idx[:,None],future]+self.bias[future]+(y-self.fields[idx[:,None],obs]-self.bias[obs])@gain.T
   cov=self.C[np.ix_(future,future)]-gain@self.C[np.ix_(obs,future)];draw=mean+noise@np.linalg.cholesky(cov).T
   nextobs=self.observed([*history,action]);full=np.broadcast_to(measurement,(samples,27)).copy();full[:,future]=draw
   risks[action]=float(self.model([*history,action]).risk(full[:,nextobs]).mean())
  return int(risks.argmin()),risks
 def heuristic(self,history,measurement):
  # Geometry-free response heuristic: probe where the posterior predicts most motion.
  obs=self.observed(history);w=self.model(history).weights(measurement[None,obs]).cpu().numpy()[0]
  amplitudes=np.array([w@np.sum(self.fields[:,9+3*a:12+3*a]**2,axis=1) for a in range(6)]);amplitudes[list(history)]=-np.inf
  return int(amplitudes.argmax())

def observations(split,sigma=None):
 spec=protocol();sigma=spec['noise_sigma_m'] if sigma is None else sigma;rng=np.random.default_rng({'development':810,'calibration':811,'test':812}[split]);data=[];truth=[]
 for i in range(len(spec[split])):
  field=np.load(BASE/'fields'/f'{split}-{i}-reference.npz')['field'].ravel()[field_indices()]
  for repeat in range(spec['noise_repeats']):data.append(field[:27]+rng.normal(0,sigma,27));truth.append(field[27:])
 return np.array(data),np.array(truth)

def run_policy(ex,measurements,policy,fixed=(0,1),samples=64):
 means=[];sds=[];quantiles=[];choices=[];stability=[]
 for i,y in enumerate(measurements):
  history=[];mm=[];ss=[];qq=[]
  for budget in range(3):
   pred=ex.predict(history,y);mm.append(pred['mean'][0]);ss.append(pred['sd'][0]);qq.append(pred['log_parameter_quantiles'][0])
   if budget==2:break
   if policy=='adaptive':
    action,risk=ex.choose(history,y,samples)
    # Fourfold sampling refinement is recorded; not a scientific all-or-nothing gate.
    refined,rr=ex.choose(history,y,samples*4);stability.append(dict(observation=i,budget=budget+1,agreement=action==refined,risks=rr.tolist()));action=refined
   elif policy=='heuristic':action=ex.heuristic(history,y)
   elif policy=='fixed':action=fixed[budget]
   elif policy=='random':action=int(np.random.default_rng(901+i*2+budget).choice([a for a in range(6) if a not in history]))
   else:raise ValueError(policy)
   history.append(action)
  means.append(mm);sds.append(ss);quantiles.append(qq);choices.append(history)
  if i%6==0:print('POLICY',policy,i,len(measurements),flush=True)
 return dict(mean=np.array(means),sd=np.array(sds),quantiles=np.array(quantiles),choices=np.array(choices)),stability

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def seal_inputs():
 paths=[BASE/'protocol.json',*[ROOT/'scripts/phystwin'/p for p in ['sequential_fixture.py','sequential_reference.py','sequential_policy.py']],ROOT/'scripts/phystwin/consistent_fixture.py',ROOT/'scripts/phystwin/tensor_fixture.py',ROOT/'scripts/phystwin/active_support_policy.py',ROOT/'scripts/phystwin/uncertainty_posterior.py',ROOT/'scripts/phystwin/gpu_multigrid.py']
 paths+=sorted((BASE/'fields').glob('bank*.npz'))+sorted((BASE/'fields').glob('development*-reference.npz'))+sorted((BASE/'fields').glob('calibration*-reference.npz'))
 return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def verify(seal):
 for p,h in seal['hashes'].items():assert sha(ROOT/p)==h,f'Sealed input changed: {p}'

def main():
 global ANALYSIS
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','calibration','test'],required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--observations',choices=['direct','rgbd'],default='direct');a=p.parse_args();spec=protocol()
 if a.observations=='rgbd':ANALYSIS=BASE/'analysis-rgbd'
 ANALYSIS.mkdir(parents=True,exist_ok=True)
 if a.stage=='test':seal=json.loads((ANALYSIS/'seal.json').read_text());verify(seal)
 interp=interpolators();bias,C,diagnostic=fit(interp);write(ANALYSIS/'forward-diagnostic.json',diagnostic);y,truth=observations(a.stage)
 sigma=spec['noise_sigma_m']
 if a.observations=='rgbd':
  with np.load(BASE/'vision'/f'{a.stage}.npz') as z:y=z['measurements'];truth=z['truth']
  sigma=.0004
 # Monitor all one-probe and initial predictive moments, and physical quantiles.
 previous=None;checks=[]
 for size in spec['posterior_levels']:
  ex=Experiment(interp,size,bias,C,sigma,a.device);current=[]
  for history in [[],*[[i] for i in range(6)],*itertools.combinations(range(6),2)]:
   current.append(ex.model(history).infer(y[:,ex.observed(history)]))
  if previous is not None:
   check={k:float(max(abs(c[k]-old[k]).max() for c,old in zip(current,previous))) for k in ['mean','sd','log_parameter_quantiles']}
   check.update(coarse=oldsize,fine=size);check['precision_met']=check['mean']<1e-5 and check['sd']<1e-5 and check['log_parameter_quantiles']<.05;checks.append(check);print('POSTERIOR',check,flush=True)
   if check['precision_met']:break
  previous=current;oldsize=size
  if size!=spec['posterior_levels'][-1]:del ex;gc.collect()
 write(ANALYSIS/f'{a.stage}-quadrature.json',checks)
 if a.stage=='development':
  # Fixed sequence selected only from development truth; same two-observation budget.
  scores=[]
  for pair in itertools.combinations(range(6),2):
   pred=ex.model(pair).infer(y[:,ex.observed(pair)])['mean'];scores.append((float(np.mean((pred-truth)**2)),pair))
  best=min(scores)[1]
  # Order chosen by development one-probe prediction error.
  best=sorted(best,key=lambda action:float(np.mean((current[action+1]['mean']-truth)**2)))
  write(ANALYSIS/'design.json',dict(fixed_sequence=best,grid_size=size,quadrature=checks,forward=diagnostic));print('DEVELOPMENT_COMPLETE',best,flush=True);return
 fixed=json.loads((ANALYSIS/'design.json').read_text())['fixed_sequence'];allresult={};stabilities={}
 for policy in ['fixed','random','heuristic','adaptive']:
  data,stable=run_policy(ex,y,policy,fixed,spec['utility_samples'][0]);allresult[policy]=data;stabilities[policy]=stable
  np.savez_compressed(ANALYSIS/f'{a.stage}-{policy}.npz',truth=truth,**data)
 if a.stage=='calibration':
  scores=np.stack([abs(v['mean']-truth[:,None,:])/(Z90*v['sd']) for v in allresult.values()]).reshape(4,len(spec[a.stage]),spec['noise_repeats'],3,243).max((0,2,3,4));scale=max(1.,float(np.sort(scores)[min(len(scores)-1,int(np.ceil(.9*(len(scores)+1)))-1)]))
  seal=dict(hashes=seal_inputs(),scale=scale,fixed_sequence=fixed,calibration_scores=scores.tolist(),scope='Frozen direct-point study before held-out policy scoring. Test references may already exist; no test prediction scores used for selection.')
  seal['observations']=a.observations
  if a.observations=='rgbd':
   seal['hashes'].update({str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/phystwin/v06_visual.py',BASE/'vision/development.npz',BASE/'vision/calibration.npz']})
  write(ANALYSIS/'seal.json',seal)
 else:scale=seal['scale']
 summary={}
 for policy,v in allresult.items():
  err=(v['mean']-truth[:,None])**2;inside=(abs(v['mean']-truth[:,None])<=scale*Z90*v['sd']).all(2).reshape(-1,spec['noise_repeats'],3).all(1)
  summary[policy]=dict(rmse_mm=(1000*np.sqrt(err.mean((0,2)))).tolist(),object_mse=err.mean(2).reshape(-1,spec['noise_repeats'],3).mean(1).tolist(),object_coverage=inside.mean(0).tolist(),mean_band_width_mm=(2000*scale*Z90*v['sd'].mean((0,2))).tolist())
 write(ANALYSIS/f'{a.stage}-summary.json',dict(policies=summary,grid_size=size,quadrature=checks,scale=scale,stability=stabilities));print('COMPLETE',a.stage,{k:v['rmse_mm'] for k,v in summary.items()},flush=True)
 if a.stage=='test':verify(seal)
if __name__=='__main__':main()
