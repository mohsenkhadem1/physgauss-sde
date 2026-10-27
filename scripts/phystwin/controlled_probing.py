"""Finite nonlinear physical-hypothesis probing reference, NOT PhysTwin uncertainty.

Nine explicit E/K states; independent fine-mesh truth; same observation/action
budget, prior-frozen fixed action, exact average over random candidate actions.
"""
from pathlib import Path
import sys,json,itertools,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.special import logsumexp
from numpy.polynomial.hermite import hermgauss
from scripts.phystwin.controlled_reference import ControlledBlock,ACTIONS,MATERIALS

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/phystwin-transfer-v01/probing'
STATES=np.array(list(itertools.product([2000.,4000.,8000.],[64.,250.,1000.])))
SIGMA=.0003


def normalize(logp):return np.exp(logp-logsumexp(logp,axis=-1,keepdims=True))

def utility(p,probes,targets,order):
    nodes,w=hermgauss(order);grid=np.array(list(itertools.product(range(order),repeat=3)))
    noise=np.sqrt(2)*SIGMA*nodes[grid];weights=np.prod(w[grid]/np.sqrt(np.pi),axis=1)
    norm=np.mean(targets*targets,axis=1);G=targets@targets.T/targets.shape[1]
    risk=[]
    for obs in probes:
        y=obs[:,None,:]+noise[None,:,:]
        likelihood=-.5*np.sum(((y[:,:,None,:]-obs[None,None,:,:])/SIGMA)**2,axis=-1)
        posterior=normalize(np.log(np.maximum(p,1e-300))[None,None,:]+likelihood)
        variance=posterior@norm-np.einsum('...i,ij,...j->...',posterior,G,posterior)
        risk.append(float(np.sum(p[:,None]*weights[None,:]*variance)))
    return np.array(risk)


def field(E,K,action,n=4):
    model=ControlledBlock(n,E,K);u=np.zeros_like(model.rest)
    for alpha in np.linspace(0,1,9)[1:]:
        model.target=ACTIONS[action]*alpha
        u=model.solve(u)['displacement_local_m']
    return u.reshape(4*n+1,n+1,n+1,3)[::n//2,::n//2,::n//2].reshape(-1,3)


def main():
    base = ROOT / 'runs/phystwin-transfer-v01'
    assert json.loads((base / 'reference/qualification.json').read_text())['passed'], 'Reference gate failed'
    for name in MATERIALS:
        assert json.loads((base / name / 'summary.json').read_text())['gate_passed'], 'Held prediction gate failed'
    OUT.mkdir(parents=True,exist_ok=True)
    # Freeze this design before generating hypothesis responses or noisy outcomes.
    protocol=dict(states_E_K=STATES.tolist(),prior='uniform nine states',sigma_m=SIGMA,
        initial='27 axial right-end displacement coordinates',probe='3 coordinates at mid-length upper corner; reset interaction',
        candidates=['axial','sideways'],target='all 243 coordinates under oblique held interaction',
        fixed='minimum prior expected posterior target variance; frozen before noisy observations',
        adaptive='minimum expected posterior target variance given prefix',random='exact mean over both reset actions',
        noise_repeats_per_truth=32,seed=20261027,truth='Two original objects, fine n8 FEM',
        forward='Nine n4 FEM hypotheses; separate from PhysTwin',quadrature_orders=[7,11],
        scope='Finite-prior controlled reference only; not a learned posterior or population calibration')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rest=np.load(ROOT/'runs/phystwin-transfer-v01/reference/soft_firm-axial-n2.npz')['rest']
    end=np.isclose(rest[:,0],.24);marker=np.argmin(np.linalg.norm(rest-[.12,.04,.04],axis=1))
    bank=OUT/'bank.npz'
    if not bank.exists():
        responses=[]
        for E,K in STATES:
            responses.append(np.stack([field(E,K,a) for a in ['axial','sideways','oblique']]))
            print('BANK',E,K,flush=True)
        np.savez_compressed(bank,fields=np.stack(responses),states=STATES)
    fields=np.load(bank)['fields'];prefix=fields[:,0,end].reshape(9,-1);probes=fields[:,:2,marker].transpose(1,0,2);targets=fields[:,2].reshape(9,-1)
    prior=np.ones(9)/9;priorrisk=utility(prior,probes,targets,11);fixed=int(np.argmin(priorrisk))
    (OUT/'fixed-choice.json').write_text(json.dumps(dict(index=fixed,action=['axial','sideways'][fixed],risks=priorrisk.tolist()),indent=2)+'\n')
    rng=np.random.default_rng(20261027);records=[];alloutcomes=[]
    for name in MATERIALS:
        truth=np.stack([np.load(ROOT/f'runs/phystwin-transfer-v01/reference/{name}-{act}-n8.npz')['sampled_positions'][-1]-rest for act in ['axial','sideways','oblique']])
        trueprefix=truth[0,end].ravel();truetarget=truth[2].ravel()
        for repeat in range(32):
            y0=trueprefix+rng.normal(0,SIGMA,size=27)
            yprobe=truth[:2,marker]+rng.normal(0,SIGMA,size=(2,3))
            p=normalize(-.5*np.sum(((prefix-y0)/SIGMA)**2,axis=1))
            r7=utility(p,probes,targets,7);r11=utility(p,probes,targets,11);chosen=int(np.argmin(r11))
            def score(weights):
                mean=weights@targets;var=weights@(targets*targets)-mean*mean
                # Discrete central equal-tail 90% interval per target coordinate.
                order=np.argsort(targets,axis=0);values=np.take_along_axis(targets,order,axis=0)
                cum=np.cumsum(weights[order],axis=0)
                lo=values[np.argmax(cum>=.05,axis=0),np.arange(len(truetarget))]
                hi=values[np.argmax(cum>=.95,axis=0),np.arange(len(truetarget))]
                return dict(mse=float(np.mean((mean-truetarget)**2)),variance=float(np.mean(var)),
                            coordinate_coverage90=float(np.mean((truetarget>=lo-1e-12)&(truetarget<=hi+1e-12))),
                            interval_width_m=float(np.mean(hi-lo)))
            post=[normalize(np.log(np.maximum(p,1e-300))-.5*np.sum(((probes[j]-yprobe[j])/SIGMA)**2,axis=1)) for j in range(2)]
            scores=[score(q) for q in post]
            records.append(dict(object=name,repeat=repeat,choice=chosen,prior_weights=p.tolist(),
                before=score(p),adaptive=scores[chosen],fixed=scores[fixed],
                random={k:float(np.mean([s[k] for s in scores])) for k in scores[0]},
                quadrature_choice_agreement=int(np.argmin(r7))==chosen,
                quadrature_max_risk_difference=float(np.max(abs(r7-r11)))))
            alloutcomes.append(dict(object=name,repeat=repeat,prefix=y0.tolist(),reset_probes=yprobe.tolist()))
    summary={'fixed_action':['axial','sideways'][fixed],'trials':len(records),'physical_objects':2,
        'quadrature_choice_agreement':sum(r['quadrature_choice_agreement'] for r in records),
        'max_quadrature_risk_difference':max(r['quadrature_max_risk_difference'] for r in records),
        'action_counts':{name:sum(r['choice']==i for r in records) for i,name in enumerate(['axial','sideways'])}}
    for policy in ['before','adaptive','fixed','random']:
        summary[policy]=dict(rmse_m=float(np.sqrt(np.mean([r[policy]['mse'] for r in records]))),
            rms_sd_m=float(np.sqrt(np.mean([r[policy]['variance'] for r in records]))),
            coordinate_coverage90=float(np.mean([r[policy]['coordinate_coverage90'] for r in records])),
            mean_interval_width_m=float(np.mean([r[policy]['interval_width_m'] for r in records])))
    summary['scope']=protocol['scope'];summary['gate']='Reference only: do not claim calibrated uncertainty or superior active probing unless evidence supports it.'
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'trials.json').write_text(json.dumps(records)+'\n');(OUT/'observations.json').write_text(json.dumps(alloutcomes)+'\n')
    print('PROBING_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
