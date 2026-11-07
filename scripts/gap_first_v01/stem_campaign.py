"""Run sealed compliant-stem cohort; plan and record before evaluating truth."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, time
from pathlib import Path
import numpy as np
from scripts.gap_first_v01.stem_mechanics import response_row
from scripts.gap_first_v01.stem_inference import plan, prior, action_menu

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'scripts/gap_first_v01/stem_protocol.json'
OUT=ROOT/'reports/gap-first-v01/stem'


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_cohort(c):
    rng=np.random.default_rng(c['test_seed'])
    lo,hi=c['compliance_prior_m_per_n']; obs=[]; truth=[]
    for i in range(c['continuous_prior_test_cases_per_condition']):
        theta=rng.uniform(lo,hi,2)
        L=c['lengths_m'][i%len(c['lengths_m'])]
        y=float(theta.sum()*c['initial_force_n']+rng.normal(0,c['initial_sigma_m']))
        truth.append(dict(case=i,compliance=theta.tolist()))
        for cond in c['conditions']:
            obs.append(dict(case=i,condition=cond,length_m=L,initial_y_m=y))
    return obs,truth


def plan_one(args):
    obs,c=args
    _,_,r=plan(obs['initial_y_m'],obs['condition'],obs['length_m'],c)
    return {**obs,**r}


def evaluate(records,truth,c):
    """Only this stage receives hidden compliance / held-out response."""
    rows=[]
    for r in records:
        theta=np.asarray(truth[r['case']]['compliance'])
        L=r['length_m']; cond=r['condition']
        p=prior(c).update(response_row(L,L,L,c['initial_force_n']),r['initial_y_m'],c['initial_sigma_m'])
        frac=1 if cond=='repeat_tip' else c['heldout_load_fraction']
        row=response_row(frac*L,frac*L,L,c['heldout_force_n'])
        actual=float(theta@row)
        menu=action_menu(cond,L,c); byname={a['name']:a for a in menu}
        # Common random corrective evidence for all policies choosing same action.
        rng=np.random.default_rng(c['test_seed']+10000+r['case']*17)
        ys={a['name']:float(theta@a['row']+rng.normal(0,a['sigma'])) for a in menu}
        random_action=menu[np.random.default_rng(c['test_seed']+20000+r['case']).integers(len(menu))]['name']
        policies=dict(proceed='proceed',fixed_look='look',fixed_half_push='push_0.50',
                      random=random_action,covariance=r['covariance_action'],
                      covariance_cost=r['covariance_cost_action'],
                      covariance_gaussian=r['covariance_gaussian_action'],risk_policy=r['selected_action'])
        evaluated={}
        for policy,name in policies.items():
            post=p if name=='proceed' else p.update(byname[name]['row'],ys[name],byname[name]['sigma'])
            prediction=post.predictions(row,c['risk_tolerance_m'])
            errors={m:abs(prediction[m]-actual) for m in c['point_models']}
            evaluated[policy]=dict(action=name,action_cost=0 if name=='proceed' else byname[name]['cost'],
                corrective_y_m=None if name=='proceed' else ys[name],prediction=prediction,
                error_m=errors,failure={m:e>c['risk_tolerance_m'] for m,e in errors.items()},
                coverage90=prediction['interval90_m'][0]<=actual<=prediction['interval90_m'][1])
        rows.append({**r,'true_response_m':actual,'policies':evaluated})
    return rows


def summarize(rows,c):
    result={}
    for cond in c['conditions']:
        group=[r for r in rows if r['condition']==cond]
        conditions={}
        for policy in c['action_controls']:
            values=[r['policies'][policy] for r in group]
            models={}
            for m in c['point_models']:
                err=np.array([v['error_m'][m] for v in values])
                risk=np.array([v['prediction'][m+'_risk'] for v in values])
                failed=err>c['risk_tolerance_m']
                models[m]=dict(rmse_mm=float(np.sqrt(np.mean(err**2))*1000),
                    mean_absolute_error_mm=float(err.mean()*1000),
                    observed_failure_rate=float(failed.mean()),predicted_failure_rate=float(risk.mean()),
                    brier_score=float(np.mean((risk-failed)**2)))
            names=[v['action'] for v in values]
            conditions[policy]=dict(n=len(group),models=models,
                actions={name:names.count(name) for name in sorted(set(names))},
                mean_action_cost=float(np.mean([v['action_cost'] for v in values])),
                coverage90=float(np.mean([v['coverage90'] for v in values])))
        result[cond]=conditions
    # Paired bootstrap uses independent physical case IDs, never treats three
    # visibility/task variants as three independent objects.
    rng=np.random.default_rng(731117); paired={}
    for control in ['fixed_look','fixed_half_push','random','covariance','covariance_cost','covariance_gaussian']:
        case_delta=[]
        for case in sorted({r['case'] for r in rows}):
            rr=[r for r in rows if r['case']==case]
            losses=[]
            for policy in ['risk_policy',control]:
                losses.append(np.mean([float(r['policies'][policy]['failure']['posterior_mean'])+
                                       r['policies'][policy]['action_cost'] for r in rr]))
            case_delta.append(losses[0]-losses[1])
        a=np.array(case_delta)
        boot=a[rng.integers(len(a),size=(5000,len(a)))].mean(1)
        paired[control]=dict(mean_failure_plus_cost_difference=float(a.mean()),
                            paired_case_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
    return dict(conditions=result,paired_decision_loss=paired,
        interpretation='Task-conditional synthetic reference study with known force and calibrated noise. No raw RGB inference or universal policy advantage claimed.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--evaluate',action='store_true');args=ap.parse_args()
    c=json.loads(PROTOCOL.read_text());OUT.mkdir(parents=True,exist_ok=True)
    if not args.evaluate:
        if (OUT/'pre-outcome-predictions.json').exists():
            raise SystemExit('Frozen predictions exist; refusing to overwrite')
        obs,truth=make_cohort(c)
        dump(ROOT/'runs/gap-first-v01/stem/hidden-truth.json',truth)
        dump(OUT/'observations.json',obs)
        freeze=dict(protocol_sha256=digest(PROTOCOL),time_unix=time.time(),
            hidden_truth_sha256=digest(ROOT/'runs/gap-first-v01/stem/hidden-truth.json'),
            observation_sha256=digest(OUT/'observations.json'),
            implementation={str(p.relative_to(ROOT)):digest(p) for p in [
                Path(__file__),ROOT/'scripts/gap_first_v01/stem_mechanics.py',
                ROOT/'scripts/gap_first_v01/stem_inference.py',ROOT/'scripts/gap_first_v01/quantiles.py']})
        dump(OUT/'protocol-freeze.json',freeze)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            records=list(pool.map(plan_one,[(o,c) for o in obs],chunksize=3))
        dump(OUT/'pre-outcome-predictions.json',records)
        dump(OUT/'prediction-freeze.json',dict(sha256=digest(OUT/'pre-outcome-predictions.json'),
             protocol_sha256=digest(PROTOCOL),time_unix=time.time(),n=len(records)))
        print('Predictions frozen; held-out responses not evaluated.',flush=True)
    else:
        if (OUT/'outcomes.json').exists():
            raise SystemExit('Outcomes exist; refusing to overwrite')
        freeze=json.loads((OUT/'prediction-freeze.json').read_text())
        assert freeze['sha256']==digest(OUT/'pre-outcome-predictions.json')
        assert freeze['protocol_sha256']==digest(PROTOCOL)
        source_freeze=json.loads((OUT/'protocol-freeze.json').read_text())
        for path,sha in source_freeze['implementation'].items():
            assert digest(ROOT/path)==sha, f'Frozen implementation changed: {path}'
        assert source_freeze['hidden_truth_sha256']==digest(ROOT/'runs/gap-first-v01/stem/hidden-truth.json')
        assert source_freeze['observation_sha256']==digest(OUT/'observations.json')
        records=json.loads((OUT/'pre-outcome-predictions.json').read_text())
        truth=json.loads((ROOT/'runs/gap-first-v01/stem/hidden-truth.json').read_text())
        rows=evaluate(records,truth,c)
        dump(OUT/'outcomes.json',rows)
        dump(OUT/'summary.json',summarize(rows,c))
        print('Held-out outcomes evaluated after prediction freeze.',flush=True)


if __name__=='__main__': main()
