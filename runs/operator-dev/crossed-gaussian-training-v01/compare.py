"""Final-only descriptive six-fit comparison with frozen physical/static baselines."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.train_campaign_crossed_prediction import metrics,ROLES

PROTOCOL=ROOT/'runs/operator-dev/crossed-gaussian-training-v01/protocol.json'
LINEAR=ROOT/'runs/gaussian-campaign-v01/hypothesis-linear-cohort-v01'
LINEAR_PREDICTIONS_SHA256='99ee3490fd56f474ae20bcd6c26c5b6f9ec414f7cde12cabf8a9874d83f9e99b'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def group_scores(rows,metadata):
    def summarize(group):
        return dict(cases=len(group),mean_position_rmse_m=float(np.mean([r['position_rmse_m'] for r in group])),
            mean_after_1s_rmse_m=float(np.mean([r['after_1s_rmse_m'] for r in group])),
            mean_static_rmse_m=float(np.mean([r['static_rmse_m'] for r in group])),
            minimum_J=float(min(r['minimum_J'] for r in group)),
            inverted_tetrahedron_frames=int(sum(r['inverted_tetrahedron_frames'] for r in group)))
    result=[]
    for axes in [('role',),('BC',),('material',),('role','BC','material')]:
        keys=sorted({tuple(metadata[r['id']][a] for a in axes) for r in rows})
        for key in keys:
            selected=[r for r in rows if tuple(metadata[r['id']][a] for a in axes)==key]
            result.append(dict(group=dict(zip(axes,key)),**summarize(selected)))
    return result

def run():
    torch.set_num_threads(2)
    d=json.loads(PROTOCOL.read_text());stage=ROOT/d['output']
    execution=json.loads((stage/'execution.json').read_text())
    assert execution['terminal'] is True and execution['completed'] is True
    assert len(execution['jobs'])==6 and all(j['state']=='completed' for j in execution['jobs'])
    for name,digest in d['source_sha256'].items():assert sha(ROOT/name)==digest,name
    dataset=stage/'dataset/dataset.npz';assert sha(dataset)==execution['dataset_sha256']
    with np.load(dataset,allow_pickle=False) as f:
        a={k:f[k].copy() for k in ['q','rest','times','ids','evaluation_role','tets']}
    manifest=json.loads((ROOT/d['input_manifest']['path']).read_text())
    assert sha(ROOT/d['input_manifest']['path'])==d['input_manifest']['sha256']
    hypotheses={h['id']:h for h in manifest['hypotheses']}
    metadata={c['id']:dict(role=c['evaluation_role'],BC=hypotheses[c['hypothesis_id']]['spec']['boundary_kind'],
        material=hypotheses[c['hypothesis_id']]['material_tuple_id']) for c in manifest['cells']}
    assert len(metadata)==96 and set(metadata)==set(map(str,a['ids']))
    references=[];models=[];role_means={}
    for job in d['jobs']:
        folder=stage/job['id'];state=json.loads((folder/'execution.json').read_text());p=json.loads((folder/'protocol.json').read_text())
        assert state['terminal'] is True and state['completed'] is True
        assert p['steps']==4000 and p['dataset_sha256']==execution['dataset_sha256'] and p['seed']==job['seed'] and p['arm']==job['arm']
        assert sha(folder/'predictions.npz')==state['predictions_sha256'] and sha(folder/'final.pt')==state['checkpoint_sha256']
        report=json.loads((folder/'metrics.json').read_text());rows=report['cases']
        assert len(rows)==96 and len({r['id'] for r in rows})==96 and {r['id'] for r in rows}==set(metadata)
        assert all(r['evaluation_role']==metadata[r['id']]['role'] for r in rows)
        models.append(dict(name=job['id'],arm=job['arm'],seed=job['seed'],groups=group_scores(rows,metadata)))
        role_means[(job['arm'],job['seed'])]={r:report['role_summaries'][r]['mean_position_rmse_m'] for r in ROLES}
        references.append(dict(path=str((folder/'metrics.json').relative_to(ROOT)),sha256=sha(folder/'metrics.json')))
    linear_execution=json.loads((LINEAR/'execution.json').read_text());linear_summary=json.loads((LINEAR/'summary.json').read_text())
    assert linear_execution['completed'] is True and linear_summary['completed'] is True
    assert sha(LINEAR/'predictions.npz')==LINEAR_PREDICTIONS_SHA256==linear_summary['cohort_predictions_sha256']
    with np.load(LINEAR/'predictions.npz',allow_pickle=False) as f:
        ids=list(map(str,f['ids']));assert len(ids)==96 and len(set(ids))==96 and set(ids)==set(metadata)
        lookup={name:i for i,name in enumerate(ids)};order=[lookup[str(name)] for name in a['ids']]
        assert np.array_equal(f['times_s'],a['times'])
        assert np.array_equal(f['evaluation_roles'][order],a['evaluation_role'])
        linear=f['positions_m'][order].copy()
    linear_report=metrics(linear,a['q'],a['rest'],a['times'],a['evaluation_role'],a['ids'],a['tets'])
    static_report=metrics(np.broadcast_to(a['rest'][:,None],a['q'].shape),a['q'],a['rest'],a['times'],a['evaluation_role'],a['ids'],a['tets'])
    for name,report in [('physical_linear',linear_report),('static_rest',static_report)]:
        models.append(dict(name=name,arm=name,seed=None,groups=group_scores(report['cases'],metadata)))
    spread=[];paired=[]
    for role in ROLES:
        for arm in ['exchange','no_exchange']:
            values=[role_means[(arm,s)][role] for s in [20261022,20261023,20261024]]
            spread.append(dict(arm=arm,role=role,seeds=3,seed_means_position_rmse_m=values,
                mean_across_seeds_m=float(np.mean(values)),sample_std_across_seeds_m=float(np.std(values,ddof=1))))
        for seed in [20261022,20261023,20261024]:
            paired.append(dict(role=role,seed=seed,exchange_minus_no_exchange_mean_position_rmse_m=role_means[('exchange',seed)][role]-role_means[('no_exchange',seed)][role]))
    destination=stage/'final-comparison';destination.mkdir(exist_ok=False)
    result=dict(schema='fixed-six-fit-crossed-prediction-comparison-v0.1',cases=96,role_counts={r:32 for r in ROLES},
        models=models,seed_variation=spread,paired_ablation_differences=paired,
        physical_linear_role_scores=linear_report['role_summaries'],static_role_scores=static_report['role_summaries'],
        evidence=references+[dict(path=str((LINEAR/'predictions.npz').relative_to(ROOT)),sha256=sha(LINEAR/'predictions.npz'))],
        protocol_sha256=sha(PROTOCOL),comparison_source_sha256=sha(__file__),dataset_sha256=execution['dataset_sha256'],
        selection='All six fixed final checkpoints reported; no seed, budget or checkpoint chosen using held scores.',
        uncertainty_scope='Three-seed spread describes optimization variation on one dependent cohort; not predictive uncertainty or an iid confidence interval.',
        information_scope='True physical hypotheses and known controls from rest; zero motion/image observations. Same four hardware contexts, three joint material tuples.',
        raw_arrays_checkpoints_LOCAL=True,calibrated_uncertainty=False,sparse_view_posterior=False,production_dataset_admission=False)
    write(destination/'summary.json',result)
    text='''# Fixed six-fit crossed prediction comparison

All96 engineering captures and six fixed final checkpoints are included. Inputs are supplied physical hypotheses, known rest/hardware and causal commands. No motion observations or images enter. Scores below average32 cases per role, excluding the rest frame.

| Role | Exchange mean ± seed SD | No exchange mean ± seed SD | Physical linear | Static rest |
|---|---:|---:|---:|---:|
'''
    for role in ROLES:
        x=next(s for s in spread if s['arm']=='exchange' and s['role']==role);n=next(s for s in spread if s['arm']=='no_exchange' and s['role']==role)
        text+=f"| {role} | {x['mean_across_seeds_m']*1000:.4f} ± {x['sample_std_across_seeds_m']*1000:.4f} mm | {n['mean_across_seeds_m']*1000:.4f} ± {n['sample_std_across_seeds_m']*1000:.4f} mm | {linear_report['role_summaries'][role]['mean_position_rmse_m']*1000:.4f} mm | {static_report['role_summaries'][role]['mean_position_rmse_m']*1000:.4f} mm |\n"
    text+='\nPer-BC/material and joint role/BC/material groups, individual seed scores, folding failures and paired ablation differences are retained in summary.json. Seed SD is optimization variation, not calibrated predictive uncertainty. The cohort uses known layouts and three dependent joint material tuples; no sparse-view inference, native/continuum accuracy, production admission or probing-policy claim follows. Raw predictions and checkpoints remain LOCAL.\n'
    (destination/'REPORT.md').write_text(text)
    print(json.dumps(dict(output=str(destination.relative_to(ROOT)),models=len(models),fits=6,cases=96)),flush=True)

if __name__=='__main__':run()
