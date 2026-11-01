"""Additive development diagnostics; no change to the completed policy run."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.physical_refinement_v15 import development_policy as d
from scripts.physical_refinement_v15.predictive_scores import marginal_scores
from scripts.physical_refinement_v15.banks import sha


def additional_baselines(report):
    pairs=report['pair_rows'];groups={}
    for r in pairs:
        groups.setdefault((r['initial_camera'],r['track']),{}).setdefault(tuple(r['sequence']),[]).append(r['metrics']['normalized_design_mse'])
    fixed={key:min(candidates,key=lambda pair:np.mean(candidates[pair])) for key,candidates in groups.items()}
    rows=[]
    for initial in (r for r in report['rows'] if r['policy']=='initial_only'):
        identity={k:initial[k] for k in ('world','initial_camera','track','repeat')}
        matched=[r for r in pairs if all(r[k]==v for k,v in identity.items())]
        pair=fixed[(initial['initial_camera'],initial['track'])]
        selected=next(r for r in matched if tuple(r['sequence'])==pair)
        rows.append(dict(**identity,policy='fixed_all',sequence=list(pair),metrics=selected['metrics']))
        probes=[r for r in matched if d.pair_category(r['sequence'])=='probes']
        assert len(probes)==6
        rows.append(dict(**identity,policy='random_probes',sequence=None,pair_count=6,metrics=d.average_metrics(probes)))
    return rows,[dict(initial_camera=k[0],track=k[1],sequence=list(v)) for k,v in fixed.items()]


def run(report):
    bank=d.assemble(report['gradient_count']);expanded=d.expand(bank,report['gradient_count'],report['scale_count']);design=d.mechanical_design(expanded)
    extra,fixed=additional_baselines(report);allrows=report['rows']+extra
    evidence=dict(bank['evidence_sha256']);budget_one=[];proper=[];risk=[];policy_one=[]
    base_names=bank['case_names'];indices=np.array([base_names.index(r['base_name']) for r in expanded['rows']])
    first=np.array([np.flatnonzero(indices==i)[0] for i in range(len(base_names))])
    for world in d.development_worlds():
        fields,wrench,targets,actions,h=d.reference(world['name'],expanded);evidence.update(h)
        target=targets.reshape(3,249)
        for track in ('V','VF'):
            initial,queries=d.prediction_queries(expanded,world['initial_camera'],track)
            likelihoods={q.name:d.Likelihood(q) for q in [initial]+queries}
            for repeat in range(4):
                identity=dict(world=world['name'],initial_camera=world['initial_camera'],track=track,repeat=repeat)
                y,h=d.observation(world['name'],repeat,'initial',world['initial_camera'],initial);evidence.update(h)
                prefix=likelihoods[initial.name].log_density(y);terms={}
                for q,(name,action,camera) in zip(queries,d.menu(world['initial_camera'])):
                    y,h=d.observation(world['name'],repeat,action,camera,q);evidence.update(h);terms[name]=likelihoods[name].log_density(y)
                cache={}
                def evaluate(history):
                    key=tuple(sorted(history))
                    if key not in cache:
                        weights=d.update(expanded['weights'],prefix,terms,history)
                        metric=d.metrics(expanded,design,weights,fields,wrench,targets,actions)
                        geomweights=np.bincount(indices,weights=weights,minlength=len(base_names))
                        shapes=expanded['shapes'][first][:,actions].reshape(len(first),len(actions),-1)
                        forces=expanded['wrenches'][:,actions,:3];torques=expanded['wrenches'][:,actions,3:]
                        scores=dict(shape_m=marginal_scores(shapes,geomweights,target[:,:243]),
                                    force_N=marginal_scores(forces,weights,target[:,243:246]),
                                    torque_Nm=marginal_scores(torques,weights,target[:,246:]))
                        cache[key]=(metric,scores)
                    return cache[key]
                for q in queries:
                    metric,scores=evaluate([q.name]);budget_one.append(dict(**identity,query=q.name,metrics=metric,proper_scores=scores))
                matching=[r for r in allrows if all(r[k]==v for k,v in identity.items())]
                for row in matching:
                    if row['policy']!='initial_only':
                        if row['sequence'] is None:
                            names=[q.name for q in queries if row['policy']!='random_probes' or not q.name.startswith('view_')]
                            one=d.average_metrics([{'metrics':evaluate([name])[0]} for name in names])
                            seq=None
                        else:
                            seq=row['sequence'][:1];one=evaluate(seq)[0]
                        policy_one.append(dict(**identity,policy=row['policy'],sequence=seq,metrics=one))
                    if row['sequence'] is None:
                        combos=list(__import__('itertools').combinations([q.name for q in queries if row['policy']!='random_probes' or not q.name.startswith('view_')],2))
                        values=[evaluate(c)[1] for c in combos]
                        scores={mod:{key:float(np.mean([v[mod][key] for v in values])) for key in ('crps_mean','central90_coordinate_coverage','mean_interval_width')} for mod in values[0]}
                    else:scores=evaluate(row['sequence'])[1]
                    proper.append(dict(**identity,policy=row['policy'],proper_scores=scores))
                    if row['policy']=='physical_risk':
                        previous=evaluate([])[0]['normalized_design_mse']
                        for step,name in enumerate(row['sequence']):
                            after=evaluate(row['sequence'][:step+1])[0]['normalized_design_mse']
                            info=row['acquisition'][step]
                            risk.append(dict(**identity,step=step+1,query=name,estimated_gain=info['estimated_scores'][name],
                                             mcse=info['monte_carlo_diagnostics'][name]['standard_error'],realized_gain=previous-after))
                            previous=after
                print(world['name'],track,repeat,'follow-on complete',flush=True)
    # Independently trained budget-one strongest query, by camera and track.
    grouped={}
    for row in budget_one:grouped.setdefault((row['initial_camera'],row['track']),{}).setdefault(row['query'],[]).append(row['metrics']['normalized_design_mse'])
    fixed_one=[dict(initial_camera=k[0],track=k[1],query=min(v,key=lambda q:np.mean(v[q]))) for k,v in grouped.items()]
    return dict(scope='Additive in-sample DEVELOPMENT diagnostic; all finite-ensemble and rendering convergence caveats retained.',
                extra_budget_two_rows=extra,fixed_all_sequences=fixed,budget_one_rows=budget_one,fixed_budget_one=fixed_one,
                budget_one_policy_rows=policy_one,proper_score_rows=proper,estimated_vs_realized=risk,evidence_sha256=evidence,
                notes=['Proper scores and coverage are coordinate-marginal, with no independent-coordinate confidence claim.',
                       'Random scores average scores over pairs, not predictive mixtures or averaged forecasts.',
                       'Fixed baselines use development targets only and are in-sample training diagnostics.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=d.ROOT/'reports/physical-refinement-v15'
    if not a.input.resolve().is_relative_to(root) or not a.output.resolve().is_relative_to(root) or a.output.exists():raise ValueError('Fresh development report required')
    record=json.loads(a.input.read_text());sources=dict(record['source_sha256']);sources[str(Path(__file__).relative_to(d.ROOT))]=sha(__file__)
    if any(sha(d.ROOT/k)!=v for k,v in sources.items()):raise ValueError('Input policy source changed')
    inputhash=sha(a.input);result=run(record)
    if sha(a.input)!=inputhash or any(sha(d.ROOT/k)!=v for k,v in sources.items()):raise ValueError('Inputs/source changed during diagnostic')
    result.update(input_report_sha256=inputhash,source_sha256=sources);a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
