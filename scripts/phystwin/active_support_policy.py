"""Exact finite-quadrature Bayes prediction and expected-variance probe selection.

The categorical fixture side/extent is latent. No true fixture label enters a
policy. The joint discrepancy is conditioned on both prefix and probe data.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.interpolate import RegularGridInterpolator,BSpline
from scipy.special import ndtri
from scipy.sparse.linalg import spsolve
from scipy.stats import qmc
from scripts.phystwin.active_support_reference import ROOT,BASE,BOUNDS,LAYOUTS,ActiveBlock,protocol
from scripts.phystwin.uncertainty_posterior import conditional_components,discrepancy,Z90

RISK_EXECUTOR=ThreadPoolExecutor(max_workers=2)


def mean_risk(model,observations):return float(model.risk(observations).mean())


def interpolators(size=9):
    axes=[np.linspace(*b,size) for b in BOUNDS]
    return [RegularGridInterpolator(axes,np.stack([np.load(BASE/'fields'/f'bank{size}-{l}-{i}-{j}.npz')['fields']
             for i in range(size) for j in range(size)]).reshape(size,size,972),method='cubic',solver=spsolve) for l in range(4)]


def nodes_and_fields(interp,size):
    axes=[np.linspace(*b,size) for b in BOUNDS]
    nodes=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,2)
    w=np.ones((size,size));w[[0,-1],:]*=.5;w[:,[0,-1]]*=.5
    return np.concatenate([f(nodes) for f in interp]),np.tile(w.ravel()/w.sum()/4,4)


def indices():
    rest=ActiveBlock(2,3000,400,'left_full').rest
    center=np.flatnonzero(np.isclose(rest[:,0],.12));prefix=(center[:,None]*3+np.arange(3)).ravel()
    markers=[np.argmin(np.linalg.norm(rest-[x,0,0],axis=1)) for x in [0.,.24]]
    probes=[243*(a+1)+3*marker+np.arange(3) for a,marker in enumerate(markers)]
    return prefix,probes,np.arange(729,972)


class Mixture:
    moment_chunk_size=65536
    def __init__(self,fields,prior,bias,C,observed,target,sigma,device):
        import torch
        self.torch=torch;self.device=device;self.prior=prior
        self.tensor_basis=None
        self.weight_batch_size=max(1,(3*2**24)//len(prior))
        f,M,gain,V,chol=conditional_components(fields,bias,C,observed,target,sigma)
        self.chol=chol
        self.f=self.tensor(np.linalg.solve(chol,f.T).T);self.normf=(self.f*self.f).sum(1)
        self.M=self.tensor(M)
        self.normM=torch.cat([(part*part).mean(1) for part in self.M.split(self.moment_chunk_size)])
        self.gain=self.tensor(gain);self.V=self.tensor(V);self.logprior=self.tensor(np.log(prior))
    def tensor(self,x):return self.torch.as_tensor(x,dtype=self.torch.float64,device=self.device)
    def weights(self,y):
        yt=self.tensor(np.linalg.solve(self.chol,np.asarray(y).T).T)
        # Keep at most two batch-by-quadrature arrays alive. This preserves the
        # whitening/likelihood formula while avoiding multiple large temporaries.
        product=(2*yt)@self.f.T
        logp=(yt*yt).sum(1)[:,None]+self.normf[None,:]
        logp.sub_(product);del product
        logp.mul_(-.5).add_(self.logprior[None,:])
        return self.torch.softmax(logp,dim=1)
    def target_mean(self,p):
        if self.tensor_basis is None:return p@self.M
        left,right,coefficients,offset,size=self.tensor_basis
        mass=p.reshape(len(p),4,size,size)
        projected=self.torch.matmul(self.torch.matmul(left.T,mass),right).reshape(len(p),-1)
        return projected@coefficients+p.sum(1,keepdim=True)*offset
    def infer(self,y):
        # Bound likelihood temporaries on refined grids and oracle noise batches.
        batch=self.weight_batch_size
        if len(y)>batch:
            parts=[self.infer(y[start:start+batch]) for start in range(0,len(y),batch)]
            return {key:np.concatenate([part[key] for part in parts]) for key in parts[0]}
        p=self.weights(y);m=self.target_mean(p)
        second=self.torch.zeros_like(m)
        for start in range(0,len(self.prior),self.moment_chunk_size):
            part=self.M[start:start+self.moment_chunk_size]
            second+=p[:,start:start+self.moment_chunk_size]@(part*part)
        v=self.torch.clamp(second-m*m+self.V,min=1e-24)
        mean=m+self.tensor(y)@self.gain.T
        return dict(mean=mean.cpu().numpy(),sd=self.torch.sqrt(v).cpu().numpy(),
                    layout_probability=p.reshape(len(y),4,-1).sum(2).cpu().numpy())
    def risk(self,y,batch=128):
        # Predictive variance is invariant to the common gain@y mean shift.
        batch=min(batch,self.weight_batch_size)
        risks=[]
        for start in range(0,len(y),batch):
            p=self.weights(y[start:start+batch]);m=self.target_mean(p)
            v=p@self.normM-(m*m).mean(1)+self.V.mean()
            risks.append(self.torch.clamp(v,min=0).cpu().numpy())
        return np.concatenate(risks)


class Experiment:
    def __init__(self,interp,size,bias,C,sigma,device,tensor_mean=True):
        self.fields,self.prior=nodes_and_fields(interp,size);self.bias=bias;self.C=C;self.sigma=sigma
        self.prefix,self.probes,self.target=indices()
        devices=device.split(',')
        self.models=[Mixture(self.fields,self.prior,bias,C,ix,self.target,sigma,devices[i%len(devices)])
                     for i,ix in enumerate([self.prefix]+[np.r_[self.prefix,q] for q in self.probes])]
        splines=[getattr(f,'_spline',None) for f in interp]
        if tensor_mean and all(s is not None for s in splines):
            first=splines[0]
            compatible=all(tuple(s.k)==tuple(first.k) and s.c.shape==first.c.shape and
                all(np.array_equal(a,b) for a,b in zip(s.t,first.t)) for s in splines)
            if compatible:
                # Exact reassociation of the existing tensor-product spline.
                # No rank truncation, hypothesis pruning or new interpolation.
                basis=[BSpline.design_matrix(np.linspace(*bounds,size),t,k).toarray() for bounds,t,k in zip(BOUNDS,first.t,first.k)]
                coefficients=np.stack([s.c for s in splines]).reshape(-1,972)
                for m,observed in zip(self.models,[self.prefix]+[np.r_[self.prefix,q] for q in self.probes]):
                    gain=m.gain.cpu().numpy()
                    adjusted=coefficients[:,self.target]-coefficients[:,observed]@gain.T
                    offset=bias[self.target]-bias[observed]@gain.T
                    m.tensor_basis=(*[m.tensor(v) for v in [*basis,adjusted,offset]],size)
        action_devices=[m.device for m in self.models[1:]]
        self.parallel_actions=len(set(action_devices))==2 and all(d.startswith('cuda') for d in action_devices)
        self.prefix_model=self.models[0]
        S=C[np.ix_(self.prefix,self.prefix)]+sigma**2*np.eye(len(self.prefix))
        self.future=[]
        for q in self.probes:
            gain=np.linalg.solve(S,C[np.ix_(self.prefix,q)]).T
            mean=self.fields[:,q]+bias[q]-(self.fields[:,self.prefix]+bias[self.prefix])@gain.T
            cov=C[np.ix_(q,q)]+sigma**2*np.eye(3)-gain@C[np.ix_(self.prefix,q)]
            self.future.append((mean,gain,np.linalg.cholesky(cov)))
    def choose(self,y0,samples=1024):
        weights=self.prefix_model.weights(y0[None]).cpu().numpy()[0]
        u=qmc.Sobol(4,scramble=True,seed=731).random_base2(int(np.log2(samples)))
        ix=np.searchsorted(np.cumsum(weights),u[:,0]);ix=np.minimum(ix,len(weights)-1)
        noise=ndtri(u[:,1:]);risks=[]
        for a,(mean,gain,chol) in enumerate(self.future):
            y=mean[ix]+y0@gain.T+noise@chol.T
            full=np.c_[np.broadcast_to(y0,(samples,len(y0))),y]
            risks.append(RISK_EXECUTOR.submit(mean_risk,self.models[a+1],full) if self.parallel_actions else mean_risk(self.models[a+1],full))
        if self.parallel_actions:risks=[future.result() for future in risks]
        return np.array(risks)
    def predict(self,ys):return [m.infer(y) for m,y in zip(self.models,ys)]


def data(spec,split):
    rng=np.random.default_rng(dict(development=501,calibration=502,test=503)[split])
    prefix,probes,target=indices();ys=[[],[],[]];truth=[]
    for i in range(len(spec[split])):
        f=np.load(BASE/'fields'/f'{split}-{i}-n8.npz')['fields'].ravel()
        for repeat in range(spec['noise_repeats']):
            noise=rng.normal(0,spec['noise_sigma_m'],972)
            for a,ix in enumerate([prefix]+[np.r_[prefix,q] for q in probes]):ys[a].append(f[ix]+noise[ix])
            truth.append(f[target])
    return [np.array(y) for y in ys],np.array(truth)


def fit_discrepancy(spec,interp):
    pred=[];fine=[];direct=[];deformation=[]
    for i,r in enumerate(spec['development']):
        pred.append(interp[LAYOUTS.index(r['layout'])](np.log([[r['E'],r['K']]]))[0])
        a=np.load(BASE/'fields'/f'development-{i}-n8.npz');b=np.load(BASE/'fields'/f'development-{i}-n4.npz')
        fine.append(a['fields'].ravel());direct.append(b['fields'].ravel())
        c=a['checks'][1:3];best=int(np.argmax(c[:,2]));deformation.append(dict(object=i,
            best_probe_nonrigid_rms_m=float(c[best,2]),best_probe_nonrigid_fraction=float(c[best,2]/c[best,3])))
    pred=np.array(pred);fine=np.array(fine);direct=np.array(direct)
    error=np.sqrt(np.mean((pred-direct)**2,axis=1));mesh=np.sqrt(np.mean((fine-direct)**2,axis=1))
    diagnostic=dict(interpolation_rms_m=error.tolist(),mesh_difference_rms_m=mesh.tolist(),deformation=deformation,
        interpolation_passed=bool(max(error)<spec['interpolation_gate_rms_m']),
        deformation_passed=all(r['best_probe_nonrigid_rms_m']>=spec['probe_nonrigid_rms_gate_m'] and
            r['best_probe_nonrigid_fraction']>=spec['probe_nonrigid_fraction_gate'] for r in deformation))
    return *discrepancy(fine-pred),diagnostic


def calibration(pred,truth,repeats):
    scores=np.stack([np.max(abs(truth-p['mean'])/(Z90*p['sd']),axis=1) for p in pred])
    scores=scores.reshape(3,-1,repeats).max((0,2));rank=int(np.ceil(.9*(len(scores)+1)))
    return max(1.,float(np.sort(scores)[rank-1])),scores


def score(pred,truth,choices,scale,repeats):
    # Choices are generated from the prefix alone, before probe outcomes are scored.
    mse=np.stack([np.mean((p['mean']-truth)**2,axis=1) for p in pred])
    rawcov=np.stack([np.mean(abs(p['mean']-truth)<=Z90*p['sd'],axis=1) for p in pred])
    cover=np.stack([np.all(abs(p['mean']-truth)<=scale*Z90*p['sd'],axis=1) for p in pred])
    width=np.stack([np.mean(2*scale*Z90*p['sd'],axis=1) for p in pred]);out={};object_mse={}
    for name,choice in choices.items():
        if name=='random':
            m=mse[1:].mean(0);c=rawcov[1:].mean(0);w=width[1:].mean(0)
            complete=cover[1:].mean(0).reshape(-1,repeats).prod(1)
        else:
            col=np.arange(len(truth));m=mse[choice,col];c=rawcov[choice,col];w=width[choice,col]
            complete=cover[choice,col].reshape(-1,repeats).all(1)
        out[name]=dict(rmse_m=float(np.sqrt(m.mean())),raw_coordinate_coverage=float(c.mean()),
            calibrated_mean_width_m=float(w.mean()),calibrated_object_all_repeats_coverage=float(complete.mean()))
        object_mse[name]=m.reshape(-1,repeats).mean(1).tolist()
    return out,object_mse


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','calibrate','test'],default='development')
    p.add_argument('--device',default='cuda:0');p.add_argument('--bank-grid',type=int,default=9,choices=[9,17]);a=p.parse_args()
    spec=protocol();out=BASE/'analysis';out.mkdir(exist_ok=True);interp=interpolators(a.bank_grid)
    mu,C,diagnostic=fit_discrepancy(spec,interp)
    (out/'development-checks.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    print('DEVELOPMENT',json.dumps(diagnostic),flush=True)
    assert diagnostic['interpolation_passed'] and diagnostic['deformation_passed'],'Forward/deformation gate failed'
    split={'development':'development','calibrate':'calibration','test':'test'}[a.stage]
    if a.stage!='development':
        dev=json.loads((out/'development-summary.json').read_text())
        assert dev['utility_stable']==dev['utility_trials']==dev['development_quadrature_choice_agreement'], 'Development policy refinement failed'
    if a.stage=='test':
        cal=json.loads((out/'calibration-summary.json').read_text())
        assert cal['utility_stable']==cal['utility_trials'], 'Calibration policy utility refinement failed'
        seal=json.loads((out/'seal.json').read_text());assert seal['protocol']==spec
        for path,digest in seal['hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    ys,truth=data(spec,split);previous=None;checks=[]
    for size in spec['quadrature_sizes']:
        ex=Experiment(interp,size,mu,C,spec['noise_sigma_m'],a.device);pred=ex.predict(ys)
        if previous is not None:
            check=dict(coarse=old_size,fine=size,mean_m=max(float(np.max(abs(x['mean']-y['mean']))) for x,y in zip(pred,previous)),
                       sd_m=max(float(np.max(abs(x['sd']-y['sd']))) for x,y in zip(pred,previous)))
            check['passed']=max(check['mean_m'],check['sd_m'])<spec['quadrature_gate_mean_sd_m'];checks.append(check)
            print('QUADRATURE',json.dumps(check),flush=True)
            if check['passed']:break
        previous=pred;old_size=size;del ex
    assert checks[-1]['passed'],'Quadrature refinement failed'
    # A low/high quadrature action audit is performed on all development prefixes.
    coarse_ex=Experiment(interp,checks[-1]['coarse'],mu,C,spec['noise_sigma_m'],a.device) if split=='development' else None
    choices=[];risks=[];utility_checks=[]
    for i,y0 in enumerate(ys[0]):
        low=ex.choose(y0,spec['utility_samples'][0]);high=ex.choose(y0,spec['utility_samples'][1])
        final=high;samples=spec['utility_samples'][1]
        if np.argmin(low)!=np.argmin(high):
            final=ex.choose(y0,spec['utility_refinement_samples']);samples=spec['utility_refinement_samples']
            stable=bool(np.argmin(high)==np.argmin(final))
        else:stable=True
        choices.append(int(np.argmin(final))+1);risks.append(final.tolist())
        quad_risk=coarse_ex.choose(y0,samples) if coarse_ex is not None else final
        utility_checks.append(dict(stable=stable,samples=samples,max_risk_difference=float(max(abs(final-low))),
            quadrature_choice_agreement=bool(np.argmin(quad_risk)==np.argmin(final))))
        if i%8==0:print('POLICY',split,i,len(ys[0]),choices[-1],stable,flush=True)
    adaptive=np.array(choices);layout=pred[0]['layout_probability'];heuristic=np.where(layout[:,:2].sum(1)>=.5,2,1)
    count=len(truth)
    if a.stage=='calibrate':
        scale,cal_scores=calibration(pred,truth,spec['noise_repeats'])
        fixed=1+int(np.argmin([np.mean((p['mean']-truth)**2) for p in pred[1:]]))
    elif a.stage=='test':scale=seal['scale'];fixed=seal['fixed_action']
    else:scale=1.;fixed=1;cal_scores=[]
    policies=dict(initial=np.zeros(count,dtype=int),fixed=np.full(count,fixed),random=None,
                  adaptive=adaptive,fixture_side_heuristic=heuristic,left=np.ones(count,dtype=int),right=np.full(count,2))
    scores,object_mse=score(pred,truth,policies,scale,spec['noise_repeats'])
    summary=dict(split=split,grid_size=size,quadrature=checks,policies=scores,scale=scale,fixed_action=fixed,
        adaptive_action_counts={str(j):int(np.sum(adaptive==j)) for j in [1,2]},
        heuristic_agreement=float(np.mean(adaptive==heuristic)),utility_stable=sum(r['stable'] for r in utility_checks),
        utility_trials=len(utility_checks),development_quadrature_choice_agreement=sum(r['quadrature_choice_agreement'] for r in utility_checks) if split=='development' else None,maximum_utility_refinement_difference_m2=max(r['max_risk_difference'] for r in utility_checks))
    if a.stage=='calibrate':
        files=[ROOT/'scripts/phystwin/active_support_reference.py',ROOT/'scripts/phystwin/active_support_policy.py',BASE/'protocol.json']
        files+=list((BASE/'fields').glob('bank*.npz'))+list((BASE/'fields').glob('development-*.npz'))
        seal=dict(protocol=spec,scale=scale,calibration_scores=cal_scores.tolist(),fixed_action=fixed,
                  hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)})
        sp=out/'seal.json'
        if sp.exists():assert json.loads(sp.read_text())==seal
        else:sp.write_text(json.dumps(seal,indent=2)+'\n')
    np.savez_compressed(out/f'{split}-predictions.npz',truth=truth,adaptive=adaptive,heuristic=heuristic,
        **{f'{a}_{k}':v for a,prediction in enumerate(pred) for k,v in prediction.items()})
    (out/f'{split}-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/f'{split}-object-mse.json').write_text(json.dumps(object_mse,indent=2)+'\n')
    (out/f'{split}-utility.json').write_text(json.dumps(dict(risks=risks,checks=utility_checks),indent=2)+'\n')
    print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
