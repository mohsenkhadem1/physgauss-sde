"""Surface-v04 fixed-case postrun MC and force-prior audit; never alters final policies.

First four matched cases of each geometry, every recorded active step. Larger
predictive MC uses independent seeds. Prior sensitivity replays recorded actions.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from . import surface as bank, surface_campaign as campaign
from .inference import risks_after_values

ROOT=bank.ROOT
REPORT=ROOT/'reports/joint-field-surface-v04'
BASE=ROOT/'runs/joint-field-surface-v04'


def check_frozen(protocol):
    for name,sha in protocol['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('Frozen source changed: '+name)


def covariance_terms(p):
    mean=p.weights@p.means
    conditional=np.einsum('h,hfg->fg',p.weights,p.covariances)
    between=np.einsum('h,hf,hg->fg',p.weights,p.means-mean,p.means-mean)
    return dict(conditional_trace_N2=float(np.trace(conditional)),
                between_hypothesis_trace_N2=float(np.trace(between)),
                total_trace_N2=float(np.trace(conditional+between)),
                between_fraction=float(np.trace(between)/max(np.trace(conditional+between),1e-300)))


def audit_actions(p,m,record,seed,n_samples):
    target=p.target(m['T'],m['offset'])
    # Same draws across actions give paired comparison SEs, but each action maps
    # those random numbers through its own predictive measurement distribution.
    rng=np.random.default_rng(seed)
    hypothesis=rng.choice(len(p.weights),n_samples,p=p.weights)
    white=rng.standard_normal(n_samples)
    candidates=[];draw_risks=[]
    for saved in record['scores']:
        a=int(saved['action']);action=m['actions'][a]
        mu,var,_=p.scalar_predictive(action['h'],bank.NOISE**2,action['offset'])
        values=mu[hypothesis]+np.sqrt(var[hypothesis])*white
        risks=risks_after_values(p,m['T'],action['h'],bank.NOISE**2,values,
                                action['offset'],m['offset'],target_moments=target)
        draw_risks.append(risks)
        candidates.append(dict(action=a,kind=action['kind'],name=action['name'],
            recorded128_expected_risk_m2=saved['expected_risk'],
            recorded128_standard_error_m2=saved['standard_error'],
            audit1024_expected_risk_m2=float(risks.mean()),
            audit1024_standard_error_m2=float(risks.std(ddof=1)/np.sqrt(n_samples))))
    means=np.array([r['audit1024_expected_risk_m2'] for r in candidates])
    best=int(np.argmin(means));chosen=next(i for i,r in enumerate(candidates) if r['action']==record['chosen'])
    difference=draw_risks[chosen]-draw_risks[best]
    se=float(difference.std(ddof=1)/np.sqrt(n_samples));delta=float(difference.mean())
    saved_chosen=candidates[chosen]['recorded128_expected_risk_m2']
    return dict(step=record['step'],recorded_choice=record['chosen'],
                audit_best=candidates[best]['action'],agreement=chosen==best,
                n_samples=n_samples,seed=seed,prior_risk_m2=target.risk,
                estimated_regret_m2=delta,paired_standard_error_m2=se,
                paired_normal95_interval_m2=[delta-1.96*se,delta+1.96*se],
                estimated_regret_fraction_of_current_risk=delta/max(target.risk,1e-300),
                recorded_chosen_risk_minus_audited_m2=float(saved_chosen-means[chosen]),
                caveat='Audit-best is selected using these same1024samples; pairwise normal interval is descriptive, not a simultaneous or selection-corrected guarantee of regret.',
                candidates=candidates)


def sensitivity(data,m,truth,case,records):
    post={scale:campaign.initial(m,truth['y'][case],scale=scale) for scale in [.5,1.,2.]}
    result=[]
    for stage in [0,4]:
        if stage:
            for scale,p in post.items():
                for r in records:
                    a=r['chosen'];p=campaign.action_update(p,m,a,truth['measurements'][case,a])
                post[scale]=p
        nominal=post[1.];pred=nominal.target(m['T'],m['offset']).mean
        material=nominal.weights@data['material'];force=nominal.weights@nominal.means
        for scale,p in post.items():
            changed=p.target(m['T'],m['offset']).mean
            result.append(dict(stage=stage,force_prior_std_scale=scale,
                force_prior_std_N=scale*bank.FORCE_STD,
                predictive_mean_change_from_nominal_mm=float(np.sqrt(np.mean((changed-pred)**2))*1000),
                material_mean_relative_change_from_nominal=float(np.sqrt(np.mean((p.weights@data['material']-material)**2)/np.mean(material**2))),
                force_mean_change_from_nominal_N=float(np.sqrt(np.mean((p.weights@p.means-force)**2))),
                covariance_decomposition=covariance_terms(p),
                **campaign.metrics(p,data,m,truth,case,intervals=True)))
    return result


def wait_for(path,deadline):
    while not path.exists():
        if time.time()>deadline:raise TimeoutError('Waiting for '+str(path))
        print('Waiting',path,flush=True);time.sleep(20)
    # A case may be observed during its atomic-less final text write; wait once.
    for attempt in range(4):
        try:return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt==3:raise
            time.sleep(1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=1024)
    ap.add_argument('--wait-seconds',type=int,default=14400);args=ap.parse_args()
    if args.samples!=1024:raise ValueError('Frozen audit sample count is1024')
    begin=time.time();deadline=begin+args.wait_seconds
    protocol_path=REPORT/'protocol.json';protocol=json.loads(protocol_path.read_text())
    check_frozen(protocol);protocol_sha=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    plan=dict(shapes=['block','step'],population='matched',case_ids=[0,1,2,3],steps=[0,1,2,3],
              mc_samples=args.samples,seed_rule='recorded_case_seed+50000000+step',
              prior_std_scales=[.5,1.,2.],prior_stages=[0,4],
              actions='Replay originally selected active actions without reselection under changed priors.',
              protocol_sha256=protocol_sha)
    plan_path=REPORT/'acquisition-audit-plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise RuntimeError('Audit plan changed')
    else:plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    out=REPORT/'acquisition-audit';out.mkdir(exist_ok=True)
    allcases=[]
    for shape in plan['shapes']:
        data=campaign.load_bank(shape);m=bank.maps(data)
        truth_path=BASE/shape/'matched-truth.npz'
        while not truth_path.exists():
            if time.time()>deadline:raise TimeoutError('Waiting for truth')
            time.sleep(20)
        truth=dict(np.load(truth_path));assert str(truth['protocol_sha256'])==protocol_sha
        for case in plan['case_ids']:
            dest=out/f'{shape}-case-{case:03d}.json'
            if dest.exists():
                old=json.loads(dest.read_text());assert old['protocol_sha256']==protocol_sha
                allcases.append(old);continue
            record=wait_for(REPORT/shape/'matched'/f'case-{case:03d}.json',deadline)
            assert record['protocol_sha256']==protocol_sha
            p=campaign.initial(m,truth['y'][case]);actions=[];began=time.time()
            for r in record['acquisition']:
                actions.append(audit_actions(p,m,r,int(record['seed'])+50000000+r['step'],args.samples))
                a=r['chosen'];p=campaign.action_update(p,m,a,truth['measurements'][case,a])
                print(shape,case,'step',r['step'],'agreement',actions[-1]['agreement'],
                      'riskregret',actions[-1]['estimated_regret_m2'],flush=True)
            doc=dict(shape=shape,case=case,protocol_sha256=protocol_sha,
                     actions=actions,prior_sensitivity=sensitivity(data,m,truth,case,record['acquisition']),
                     frozen_final_policy_metrics=[r for r in record['rows'] if r['step']==4],
                     seconds=time.time()-began)
            dest.write_text(json.dumps(doc,indent=2)+'\n');allcases.append(doc)
    check_frozen(protocol)
    rows=[a for c in allcases for a in c['actions']]
    assert len(allcases)==8 and len(rows)==32
    replay_max_difference=0.
    for c in allcases:
        original=json.loads((REPORT/c['shape']/'matched'/f"case-{c['case']:03d}.json").read_text())
        for stage in [0,4]:
            recorded=next(r for r in original['rows'] if r['policy']=='active' and r['step']==stage)
            replay=next(r for r in c['prior_sensitivity'] if r['stage']==stage and r['force_prior_std_scale']==1.)
            for key in ['rmse_mm','material_relative_rmse','force_rmse_N','coverage90']:
                replay_max_difference=max(replay_max_difference,abs(recorded[key]-replay[key]))
    assert replay_max_difference<1e-12,'Nominal posterior replay differs from frozen run'
    summary=dict(plan=plan,cases=len(allcases),action_decisions=len(rows),
                 nominal_replay_max_metric_difference=replay_max_difference,
                 exact_choice_agreement=float(np.mean([r['agreement'] for r in rows])),
                 mean_estimated_regret_m2=float(np.mean([r['estimated_regret_m2'] for r in rows])),
                 maximum_estimated_regret_fraction=float(max(r['estimated_regret_fraction_of_current_risk'] for r in rows)),
                 disagreements_with_pairwise95_lower_above_zero=sum(r['paired_normal95_interval_m2'][0]>0 for r in rows),
                 mean_chosen_recorded128_minus_audited1024_risk_m2=float(np.mean([r['recorded_chosen_risk_minus_audited_m2'] for r in rows])),
                 prior_sensitivity_summary=[],seconds=time.time()-begin,
                 scope='Predeclared8matchedcases/32decisions; diagnostics only, not full-cohort policy ranking. No final action is changed. Higher-budget minima and pairwise intervals have selection bias; no exact regret guarantee.',
                 source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),**protocol['source_sha256']})
    for stage in [0,4]:
        for scale in [.5,1.,2.]:
            selected=[r for c in allcases for r in c['prior_sensitivity'] if r['stage']==stage and r['force_prior_std_scale']==scale]
            keys=['rmse_mm','hidden_rmse_mm','changed_load_rmse_mm','material_relative_rmse','force_rmse_N','coverage90','material_coverage90','force_coverage90','predictive_mean_change_from_nominal_mm','material_mean_relative_change_from_nominal','force_mean_change_from_nominal_N']
            summary['prior_sensitivity_summary'].append(dict(stage=stage,force_prior_std_scale=scale,**{k:float(np.mean([r[k] for r in selected])) for k in keys}))
    (REPORT/'acquisition-audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_sha256','prior_sensitivity_summary','plan']},indent=2),flush=True)


if __name__=='__main__':main()
