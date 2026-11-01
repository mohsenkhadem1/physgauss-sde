"""Development-only budget-two policy diagnostic; never a reserved evaluator.

Fixed baselines are selected and measured on the SAME development worlds and
are explicitly training diagnostics. Random means an exact uniform expectation
over all unordered distinct pairs, not an averaged forecast. Acquisition sees
only model predictions; actual development responses are used solely for scoring.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse, hashlib, itertools, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.physical_refinement_v15.belief import choose
from scripts.physical_refinement_v15.posterior import Likelihood, posterior, prediction_metrics
from scripts.physical_refinement_v15.banks import (BASE,ROOT,assemble,expand,mechanical_design,
                                                  checked_npz,nonrigid,sha)
from scripts.physical_refinement_v15.development_data import (development_worlds,menu,
                                                             prediction_queries,observation)


def update(prior, initial_ll, query_ll, history):
    """Rebuild every update from original prior; reject duplicate evidence."""
    if len(history)!=len(set(history)) or any(name not in query_ll for name in history):
        raise ValueError('Only distinct available observations can be consumed')
    return posterior(prior,[initial_ll]+[query_ll[name] for name in history])[0]


def select_sequence(prior, initial_ll, query_ll, queries, design, policy, *, seed, fantasies):
    """Outcome-blind selector; only the chosen likelihood is consumed each step."""
    history=[]; diagnostics=[]
    weights=update(prior,initial_ll,query_ll,history)
    for step in range(2):
        name,info=choose(weights,queries,design,policy,used=history,
                         seed=seed+step,fantasies=fantasies)
        history.append(name);diagnostics.append(info)
        weights=update(prior,initial_ll,query_ll,history)
    return history,weights,diagnostics


def pair_category(pair):
    views=sum(name.startswith('view_') for name in pair)
    return ('probes','mixed','views')[views]


def fit_fixed(rows):
    """Choose same fixed pair per camera/track/category using DEV loss only."""
    groups={}
    for row in rows:
        key=(row['initial_camera'],row['track'],pair_category(row['sequence']))
        groups.setdefault(key,{}).setdefault(tuple(row['sequence']),[]).append(row['metrics']['normalized_design_mse'])
    return {key:min(candidates,key=lambda pair:float(np.mean(candidates[pair])))
            for key,candidates in groups.items()}


def reference(world,expanded):
    indices=[i for i,a in enumerate(expanded['actions']) if a.startswith('design_')]
    fields=[];wrenches=[];evidence={}
    for i in indices:
        folder=BASE/'worlds'/world/'n6'/f'pose-{i:03d}'
        data,h=checked_npz(folder,'private-state.npz');evidence.update(h)
        with data:fields.append(data['field'].copy())
        data,h=checked_npz(folder,'reaction-wrench.npz');evidence.update(h)
        with data:wrenches.append(np.r_[data['forces_N'][1],data['torques_Nm'][1]])
    from scripts.phystwin.sequential_fixture import points
    shapes=np.asarray([nonrigid(points(),f) for f in fields])
    targets=np.concatenate([shapes.reshape(len(indices),-1),np.asarray(wrenches)],axis=-1).reshape(-1)
    return np.asarray(fields),np.asarray(wrenches),targets,indices,evidence


def metrics(expanded,design,weights,reference_fields,reference_wrenches,targets,indices):
    result=prediction_metrics(expanded,weights,reference_fields,reference_wrenches,indices)
    error=(weights@design.values-targets)/design.scales
    result['normalized_design_mse']=float(np.mean(error**2))
    result['normalized_design_utility']=-result['normalized_design_mse']
    return result


def average_metrics(rows):
    return {key:float(np.mean([row['metrics'][key] for row in rows])) for key in rows[0]['metrics']}


def compute(bank,*,gradient_count=9,scale_count=97,fantasies=128):
    expanded=expand(bank,gradient_count,scale_count);design=mechanical_design(expanded)
    evidence=dict(bank['evidence_sha256']);pairs=[];adaptive=[];initial_rows=[]
    for wi,world in enumerate(development_worlds()):
        f,w,target,indices,h=reference(world['name'],expanded);evidence.update(h)
        for ti,track in enumerate(('V','VF')):
            initial,queries=prediction_queries(expanded,world['initial_camera'],track)
            likelihoods={q.name:Likelihood(q) for q in [initial]+queries}
            for repeat in range(4):
                identity=dict(world=world['name'],initial_camera=world['initial_camera'],track=track,repeat=repeat)
                y,h=observation(world['name'],repeat,'initial',world['initial_camera'],initial);evidence.update(h)
                initial_ll=likelihoods[initial.name].log_density(y);terms={}
                for q,(name,action,camera) in zip(queries,menu(world['initial_camera'])):
                    y,h=observation(world['name'],repeat,action,camera,q);evidence.update(h)
                    terms[name]=likelihoods[name].log_density(y)
                score=lambda weights:metrics(expanded,design,weights,f,w,target,indices)
                initial_rows.append(dict(**identity,policy='initial_only',sequence=[],metrics=score(update(expanded['weights'],initial_ll,terms,[]))))
                for pair in itertools.combinations([q.name for q in queries],2):
                    weights=update(expanded['weights'],initial_ll,terms,pair)
                    pairs.append(dict(**identity,sequence=list(pair),metrics=score(weights)))
                for policy in ('visual_uncertainty','physical_risk'):
                    seed=710001+wi*10000+ti*1000+repeat*10
                    history,weights,diagnostics=select_sequence(expanded['weights'],initial_ll,terms,queries,design,policy,seed=seed,fantasies=fantasies)
                    adaptive.append(dict(**identity,policy=policy,sequence=history,metrics=score(weights),acquisition=diagnostics,seed=seed))
                print(world['name'],track,repeat,'complete',flush=True)
    fixed=fit_fixed(pairs);rows=initial_rows+adaptive
    for original in initial_rows:
        identity={key:original[key] for key in ('world','initial_camera','track','repeat')}
        matched=[r for r in pairs if all(r[key]==value for key,value in identity.items())]
        if len(matched)!=15:raise ValueError('All 15 pairs required; no failure filtering')
        rows.append(dict(**identity,policy='random_exact',sequence=None,pair_count=15,metrics=average_metrics(matched),interpretation='Expected error over uniformly sampled distinct pairs; not error of averaged predictions'))
        for category in ('mixed','views','probes'):
            pair=fixed[(identity['initial_camera'],identity['track'],category)]
            selected=next(r for r in matched if tuple(r['sequence'])==pair)
            rows.append(dict(**identity,policy='fixed_'+category,sequence=list(pair),metrics=selected['metrics'],interpretation='In-sample development-selected fixed policy'))
    summaries=[]
    for track in ('V','VF'):
        for policy in dict.fromkeys(r['policy'] for r in rows):
            selected=[r for r in rows if r['track']==track and r['policy']==policy]
            summaries.append(dict(track=track,policy=policy,cases=len(selected),metrics=average_metrics(selected)))
    return dict(scope='Four development worlds, four noise repeats, V/VF; budget two. Fixed policies trained and scored on development; no reserved evaluation or calibrated-truth claim.',
                gradient_count=gradient_count,scale_count=scale_count,fantasies=fantasies,
                worlds=4,repeats=4,tracks=['V','VF'],budget=2,
                fixed_sequences=[dict(initial_camera=k[0],track=k[1],category=k[2],sequence=list(v)) for k,v in fixed.items()],
                rows=rows,pair_rows=pairs,summary=summaries,evidence_sha256=evidence,
                limitations=['Posterior discretization and rendered sensor convergence must be established separately.',
                             'Acquisition MCSE describes fantasy integration only, not physical uncertainty calibration.',
                             'Repeat errors are correlated within four worlds; means are descriptive, not population confidence intervals.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--gradient-count',type=int,default=9);p.add_argument('--scale-count',type=int,default=97)
    p.add_argument('--fantasies',type=int,default=128);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();allowed=(ROOT/'reports/physical-refinement-v15').resolve()
    if not out.is_relative_to(allowed) or out.exists():raise ValueError('Fresh development report path required')
    sources={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),*[ROOT/'scripts/physical_refinement_v15'/f'{name}.py' for name in ('belief','risk','banks','posterior','development_data','quadrature','study','predictive_scores')]]}
    result=compute(assemble(a.gradient_count),gradient_count=a.gradient_count,scale_count=a.scale_count,fantasies=a.fantasies)
    if any(sha(ROOT/path)!=digest for path,digest in sources.items()):raise ValueError('Sources changed during diagnostic')
    result['source_sha256']=sources;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
