"""Independent all48 scalar quadrature/ranking audit, no model/GPU/NPZ replay."""
import argparse,numpy as np
from experiments import run_layout_grouped_allactions_v01 as run
ROOT=run.ROOT;LOCAL=run.LOCAL
SOURCE='experiments/audit_layout_grouped_allactions_v01.py';TEST='tests/test_audit_layout_grouped_allactions_v01.py'
EXPECTED_PROTOCOL='f5ad5826f253de752ef223fe9240964f4983bb007f13ec1b2ac7c5da342479eb'
def need(v,message):
    if not v:raise ValueError(message)
def check_rows(rows,gates,reported):
    need(len(rows)==48,'All48 retained');table={tuple(r['job'][k]for k in('outer','amplitude','action','order')):r['result']for r in rows}
    need(len(table)==48 and set(table)=={(o,a,i,n)for i in range(8)for o,a in run.VARIANTS for n in(8,16)},'Every unique fixed bank/action/outcome job')
    actions=[]
    for i in range(8):
        outcome=[];ok=True
        for o,a in run.VARIANTS:
            c,f=table[o,a,i,8],table[o,a,i,16]
            for value in(c,f):
                need(all(np.isfinite(value[k])and value[k]>=0 for k in('predictive_mass','omitted_mass_bound','omitted_risk_bound'))and abs(value['predictive_mass']-1)<=1e-10+value['omitted_mass_bound'],'Every predictive density mass/tail')
            error=abs(c['expected_target_variance']-f['expected_target_variance'])+f['omitted_risk_bound'];outcome.append(error);ok=ok and np.isfinite(error)and error<=gates['outcome_relative']*f['prior_target_variance']
        fine=table[512,64,i,16];risk=fine['expected_target_variance'];prior=fine['prior_target_variance'];amp=abs(table[512,32,i,16]['expected_target_variance']-risk);outer=abs(table[256,64,i,16]['expected_target_variance']-risk)
        ok=bool(ok and np.isfinite([risk,prior,amp,outer]).all()and prior>0 and 0<=risk<=prior and amp<=gates['latent_relative']*prior and outer<=gates['latent_relative']*prior)
        actions.append(dict(action=i,qualified=ok,risk=risk,uncertainty=outcome[-1]+amp+outer,outcome_errors=outcome,amplitude_error=amp,outer_error=outer))
    best=min(actions,key=lambda a:(a['risk'],a['action']));ties=[a['action']for a in actions if a['risk']-best['risk']<=a['uncertainty']+best['uncertainty']];qualified=all(a['qualified']for a in actions);winner=best['action']if qualified and len(ties)==1 else None
    need(reported['actions']==actions and reported['qualified']is qualified and reported['numerical_ties']==ties and reported['unique_best_action']==winner and reported['all8_retained']is True,'Independently recomputed every risk/refinement/tie/admission')
    return dict(qualified=qualified,actions_qualified=sum(a['qualified']for a in actions),numerical_tie_count=len(ties),unique_winner_resolved=winner is not None,maximum_outcome_error_m2=max(e for a in actions for e in a['outcome_errors']),maximum_amplitude_error_m2=max(a['amplitude_error']for a in actions),maximum_outer_error_m2=max(a['outer_error']for a in actions),indicators_are_rigorous_bounds=False)

def audit():
    path=LOCAL/'independent-terminal-review.json';need(not path.exists(),'No audit overwrite');p=run.validate(EXPECTED_PROTOCOL);e=run.read(LOCAL/'execution.json');need(e.get('completed')is True and e.get('terminal')is True and'failure'not in e,'Completed clean controller');need(e['protocol_SHA256']==EXPECTED_PROTOCOL and e['release']==run.record(LOCAL/'release.json'),'Controller protocol/release');need(0<=e['wall_s']<=p['resources']['global_wall_s'],'Global cap')
    rows=e['devices'];need(len(rows)==3 and[r['slot']for r in rows]==[0,1,2]and[r['physical_UUID']for r in rows]==p['devices']and len({r['pid']for r in rows})==3,'Fixed complete3device worklist');absent=run.parity.old.old.mechanical_screen.shared.old.absent;absent(e['controller_pid']);jobs=[];values=[]
    for row in rows:
        absent(row['pid']);need(row.get('reaped')is True and row.get('exit_code')==0 and row.get('cleanup_exit_code')==0 and row['status']=='qualified','Clean reaped worker');f=ROOT/row['result']['path'];need(run.record(f)==row['result']and f==LOCAL/f"worker-{row['slot']}/result-LOCAL.json",'Exact worker bytes');v=run.read(f);run.worker_context(v,row,p,EXPECTED_PROTOCOL);jobs.extend(v['jobs']);values.append(v)
        need(len(v['bank_replay'])==3 and[(b['outer'],b['amplitude'])for b in v['bank_replay']]==list(run.VARIANTS)and all(0<=b['logZ_error']<=1e-8 and 0<=b['layout_probability_error']<=1e-10 for b in v['bank_replay']),'Same all3 admitted bank replays')
        need(0<=v['wall_s']<=120 and 0<=v['peak_RSS_bytes']<=p['resources']['aggregate_host_RSS_bytes']and 0<=row['peak_host_RSS_bytes']<=p['resources']['aggregate_host_RSS_bytes']and 0<=row['peak_owned_VRAM_bytes']<=p['resources']['owned_VRAM_bytes'],'Owned bounded resources')
    classification=check_rows(jobs,p['gates'],e['ranking']);need(e['qualified']is classification['qualified'],'Controller numerical classification')
    summary=dict(integrity_passed=True,completed=True,all48_jobs_retained=True,**classification,wall_s=e['wall_s'],peak_host_RSS_bytes=max(r['peak_host_RSS_bytes']for r in rows),peak_owned_VRAM_bytes=max(r['peak_owned_VRAM_bytes']for r in rows),utility_wall_s=sum(j['wall_s']for j in jobs),scope='Same old LINEAR development case ranking only; no nonlinear/spatial/closed-loop advantage admission',no_model_GPU_or_array_replay=True)
    run.write(path,dict(summary=summary,source_SHA256={s:run.sha(ROOT/s)for s in(SOURCE,TEST)},protocol=run.record(LOCAL/'protocol.json'),release=run.record(LOCAL/'release.json'),execution=run.record(LOCAL/'execution.json'),worker_results=[r['result']for r in rows]));return summary

def inventory():
    r=run.read(LOCAL/'independent-terminal-review.json');need(r['summary']['integrity_passed'],'Audited integrity');directory=ROOT/'reports/layout-grouped-allactions-v01';directory.mkdir(parents=True,exist_ok=True);files=[]
    for path,value in((directory/'summary.json',r['summary']),(directory/'source-provenance.json',dict(protocol=r['protocol'],release=r['release'],source_SHA256=r['source_SHA256'],worker_SHA256=[v['sha256']for v in r['worker_results']]))):
        need(not path.exists(),'No public overwrite');run.write(path,value);files.append(run.record(path))
    files.extend(run.record(ROOT/s)for s in(SOURCE,TEST));path=LOCAL/'publication-inventory-completed.json';need(not path.exists(),'No inventory overwrite');run.write(path,dict(files=files,raw_observations_category_weights_risk_vectors_included=False));return run.record(path)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','inventory']);a=p.parse_args();print(audit()if a.action=='audit'else inventory())
