"""Numerical proposal repair only: exact importance correction with Beta(a+2,b+2).

Frozen neural densities/results are untouched. A shape-shifted proposal removes
endpoint singularities; its own normalized density appears in L/g. Prior and
likelihood are identical. Original inverse-CDF failure remains preserved.
"""
from __future__ import annotations
import argparse,json,os,signal,time
from pathlib import Path
import numpy as np
from experiments import campaign_c38_beta_importance as original
from experiments import run_campaign_c38_beta_control as frozen

ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/campaign_c38_beta_importance_shifted.py'
OLD_PROTOCOL='runs/operator-dev/c38-beta-importance-v01/protocol-v01.json'
OLD_SHA='47fe34bf91cc8782af408ccebf54364a7d704731a24f015f6b6ed4bf74fc46d2'

def shifted_parameters(parameters):
    p={k:np.asarray(v,dtype=float).copy() for k,v in parameters.items()}
    original.numpy_parameters(p)
    p['alpha']+=2.;p['beta']+=2.
    return p

def prepare(path,output):
    if Path(path).exists() or Path(output).exists():raise FileExistsError('Fresh proposal protocol and output only')
    old=original.validate(ROOT/OLD_PROTOCOL,OLD_SHA);directory=ROOT/old['output'];failed=json.loads((directory/'summary.json').read_text());terminal=ROOT/'runs/operator-dev/c38-beta-importance-v01/failed-terminal.json';proof=json.loads(terminal.read_text())
    if failed.get('completed') is not False or failed.get('failure',{}).get('type')!='FloatingPointError' or proof.get('terminal') is not True or proof.get('pid_absent') is not True or proof.get('source_pins_unchanged') is not True:raise ValueError('Original failed proposal must be terminal/preserved')
    p=dict(old);p.update(schema='c38-beta-shifted-importance-v02',status='FROZEN_BEFORE_SHIFTED_PROPOSAL_EVALUATION',output=frozen.rel(output),source_sha256={n:frozen.sha(ROOT/n) for n in [SOURCE]+original.source_names()},proposal='0.5 uniform prior +0.5 mixture with frozen NN weights and EACH normalized Beta shape shifted a+2,b+2; use THIS exact proposal density in likelihood/g, never the original NN density',proposal_repair='Endpoint-singularity removal ONLY, no prior/likelihood/noise/observations/fit/criteria/budget/seed changes',original_failed_protocol=frozen.record(ROOT/OLD_PROTOCOL),original_failed_summary=frozen.record(directory/'summary.json'),original_failed_terminal=frozen.record(terminal),reused_prior_records={},reused_reference_normalizers={})
    for c in failed['cases']:
        if 'prior' in c['proposals']:
            key=f"{c['role']}-{c['object_index']:03d}-context-{c['context']}";p['reused_prior_records'][key]=c['proposals']['prior']['records']
    for f in sorted(directory.glob('*-reference-normalizer.json')):p['reused_reference_normalizers'][f.name]=frozen.record(f)
    p['reuse_scope']='Only COMPLETE original-prior clouds and completed continuous evidence normalizers; ALL shifted neural proposals computed anew; original NNproposal outcomes not relabelled'
    p['proposal_shape_shift']=2;p['preserved_plus1_pure_control_failure']=frozen.record(ROOT/'runs/operator-dev/c38-beta-importance-v02/shape-plus1-control-failure.json')
    Path(path).parent.mkdir(parents=True,exist_ok=True);frozen.write(path,p);return frozen.sha(path)

def validate(path,digest):
    if frozen.sha(path)!=digest:raise ValueError('Shifted proposal protocol changed')
    p=json.loads(Path(path).read_text())
    if p.get('schema')!='c38-beta-shifted-importance-v02' or p.get('proposal_shape_shift')!=2 or p.get('budgets')!=list(original.BUDGETS) or p.get('scramble_seeds')!=list(original.SCRAMBLES) or p.get('fit_seeds')!=list(frozen.FIT_SEEDS) or p.get('comparison')!=original.CRITERIA or set(p.get('roles',{}))!={'DEV','CAL'} or set(p['source_sha256'])!=set([SOURCE]+original.source_names()):raise ValueError('Exact prospective shifted-proposal study required')
    for n,d in p['source_sha256'].items():
        if frozen.sha(ROOT/n)!=d:raise ValueError('Frozen shifted-proposal dependency changed')
    original.validate(frozen.checked(p['original_failed_protocol']),OLD_SHA)
    frozen.checked(p['original_failed_summary']);frozen.checked(p['original_failed_terminal']);frozen.checked(p['preserved_plus1_pure_control_failure'])
    for refs in p['reused_prior_records'].values():
        if len(refs)!=4:raise ValueError('All4 prior scrambles required for reuse')
        for r in refs:frozen.checked(r)
    for r in p['reused_reference_normalizers'].values():frozen.checked(r)
    frozen.guard_resources(p);return p

def group_reference(p,public,role,c):
    key=f'{role}-context-{c}-reference-normalizer.json'
    if key not in p['reused_reference_normalizers']:
        ids,y,F,sides,refs=original.reference_group(p,public,role,c);return ids,y,F,sides,refs,False
    ids=np.array([i for i in p['object_indices'] if i%8==c]);mask=public['mask'][0,c];F=public['forces'][0,c,mask];sides=public['sides'][0,c,mask].astype(int);y=public['observed_m'][ids,c][:,mask];refs=frozen.load_record(p['reused_reference_normalizers'][key])
    if len(refs)!=len(ids):raise ValueError('Reused reference case count changed')
    return ids,y,F,sides,refs,True

def reuse_prior(p,key,role,i):
    records=[frozen.load_record(r) for r in p['reused_prior_records'][key]]
    expected={s+(0 if role=='DEV' else 10000)+100*i for s in original.SCRAMBLES}
    if {r['scramble_seed'] for r in records}!=expected or not all(r.get('completed') is True and r.get('proposal')=='prior' and [b['count'] for b in r['budgets']]==list(original.BUDGETS) for r in records):raise ValueError('Prior reuse does not match exact4scrambles/allbudgets')
    summary=original.final_case_summary(records);summary.update(records=p['reused_prior_records'][key],reused_from_original_failed_attempt=True);return summary

def run(p,protocol_record):
    out=frozen.fresh(ROOT/p['output']);ledger=dict(completed=False,protocol=protocol_record,pid=os.getpid(),cases=[],TEST_opened=False,raw_local=True,original_neural_results_unchanged=True,original_failed_attempt_preserved=True);frozen.write(out/'summary.json',ledger);started=time.monotonic()
    def expired(*_):raise TimeoutError('Frozen900s CPU2 shifted-proposal cap; preserve every partial checkpoint')
    old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,p['wall_cap_seconds'])
    try:
        for role in ('DEV','CAL'):
            public=frozen.load_arrays(p['roles'][role]['public']);params={s:frozen.load_arrays(r) for s,r in p['roles'][role]['parameters'].items()}
            for c in range(8):
                ledger['current_attempt']=dict(role=role,context=c,stage='reference_normalizer');frozen.write(out/'summary.json',ledger);ids,y,F,sides,refs,reused=group_reference(p,public,role,c);frozen.write(out/f'{role}-context-{c}-reference-normalizer.json',refs)
                if any(r['new_reference_normalizer_relative_error']>p['reference']['relative_normalizer_tolerance'] for r in refs):raise RuntimeError('Evidence resolution gate failed; attempted values preserved')
                for j,i in enumerate(ids):
                    key=f'{role}-{i:03d}-context-{c}';directory=out/key;directory.mkdir();case=dict(role=role,object_index=int(i),context=c,proposals={},reference_log_evidence=refs[j]['log_evidence'],reference_normalizer_reused=reused,new_reference_group_seconds=0. if reused else refs[j]['new_reference_group_seconds']);ledger['cases'].append(case)
                    for proposal in ['prior']+list(map(str,frozen.FIT_SEEDS)):
                        if proposal=='prior' and key in p['reused_prior_records']:
                            case['proposals'][proposal]=reuse_prior(p,key,role,int(i));frozen.write(out/'summary.json',ledger);continue
                        pars=None if proposal=='prior' else shifted_parameters(frozen.slice_parameters(params[proposal],[int(i)*8+c]));records=[]
                        for scramble in original.SCRAMBLES:
                            ledger['current_attempt']=dict(role=role,context=c,object_index=int(i),proposal=proposal,scramble_seed=scramble,stage='weighted_cloud');frozen.write(out/'summary.json',ledger);frozen.guard_resources(p);seed=scramble+(0 if role=='DEV' else 10000)+100*int(i);u=original.sample_proposal(original.sobol_points(seed),pars);lw,q=original.evaluate_cloud(u,y[j],F,sides,pars,chunk=p['chunk']);value=dict(proposal=proposal,proposal_shapes_shifted=proposal!='prior',scramble_seed=seed,completed=True,budgets=[])
                            for N in original.BUDGETS:
                                s=original.summarize_cloud(u[:N],lw[:N],q[:N],p['grid']);value['budgets'].append(dict(count=N,moments=s,comparison=original.compare(s,refs[j])))
                            path=directory/f'{proposal}-scramble-{scramble}.json';frozen.write(path,value);records.append(value)
                        case['proposals'][proposal]=original.final_case_summary(records);case['proposals'][proposal]['records']=[frozen.record(directory/f'{proposal}-scramble-{s}.json') for s in original.SCRAMBLES];frozen.write(out/'summary.json',ledger)
                    print(role,int(i),c,'complete',round(time.monotonic()-started,2),flush=True)
        perproposal={name:dict(cases=len(ledger['cases']),passed_cases=sum(c['proposals'][name]['passed'] for c in ledger['cases']),all_cases_passed=all(c['proposals'][name]['passed'] for c in ledger['cases'])) for name in ['prior']+list(map(str,frozen.FIT_SEEDS))}
        ledger.update(completed=True,per_proposal_qualification=perproposal,all_case_proposals_numerically_passed=all(x['all_cases_passed'] for x in perproposal.values()),seconds=time.monotonic()-started,resources=frozen.guard_resources(p));frozen.write(out/'summary.json',ledger)
    except BaseException as e:
        ledger.update(seconds=time.monotonic()-started,failure=dict(type=type(e).__name__,message=str(e)));frozen.write(out/'summary.json',ledger);raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)

def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');args=a.parse_args()
    if args.mode=='prepare':print(prepare(args.protocol,args.output));return
    p=validate(args.protocol,args.protocol_sha256);run(p,frozen.record(args.protocol))
if __name__=='__main__':main()
