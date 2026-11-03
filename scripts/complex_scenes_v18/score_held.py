"""Score the complete prospective census, after every prediction is terminal.

Unsupported and failed slots remain in the census. Physical reference targets
are opened only after all supported prediction slots have terminal seals.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.complex_scenes_v18.run_development import ROOT,write
from scripts.complex_scenes_v18.held_manifest import sha
from scripts.complex_scenes_v18.world_queries import Query
from scripts.complex_scenes_v18.score_fitted_development import metrics
from scripts.complex_scenes_v18.world_aggregation import paired

METHODS=('ours','analytic-control','spring-gaus','phystwin','uniphy')


def census(campaign):
    campaign=Path(campaign);manifest=campaign/'public-manifest.json';seal=json.loads((campaign/'manifest-seal.json').read_text())
    if sha(manifest)!=seal['public_manifest_sha256']:raise ValueError('Public manifest integrity')
    public=json.loads(manifest.read_text())
    if public['split']!='held' or public['freeze_sha256']!=seal['freeze_sha256']:raise ValueError('Held protocol identity')
    rows=[]
    for world in public['worlds']:
        for index,query in enumerate(world['queries']):
            for method in METHODS:
                row=dict(world_id=world['world_id'],family=world['family'],query_index=index,public_query=query,world_material_commitment=world['world_material_commitment'],method=method,state='pending')
                rows.append(row)
                if method=='uniphy'and world['family']!='container_drop':row['state']='unsupported';continue
                path=campaign/'predictions'/method/world['world_id']/f'query{index}'/'prediction-seal.json'
                if not path.exists():continue
                prediction=json.loads(path.read_text())
                if prediction['forecast_targets_read'] or prediction['world_id']!=world['world_id'] or prediction['query_index']!=index or prediction['method']!=method or prediction['public_manifest_sha256']!=seal['public_manifest_sha256']:raise ValueError('Prediction identity mismatch')
                row['prediction_seal_sha256']=sha(path)
                if not prediction['completed']:
                    row.update(state=prediction.get('state','rollout_failed'),error=prediction.get('error'))
                    if row['state']not in ('fit_failed','rollout_failed'):raise ValueError('Unknown terminal failure')
                    continue
                folder=ROOT/prediction['physical_output'];done=folder/'complete.json'
                if sha(done)!=prediction['physical_complete_sha256']:raise ValueError('Changed physical prediction')
                status=json.loads(done.read_text())
                if not status['completed']or sha(folder/'trajectory.npz')!=status['trajectory_sha256']:raise ValueError('Incomplete or changed prediction trajectory')
                row.update(state='sealed',physical_output=str(folder))
    keys=[(r['world_id'],r['query_index'],r['method'])for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate world/query/method slot')
    if len(rows)!=seal['total_queries']*len(METHODS):raise ValueError('Incomplete declared census')
    return rows


def load(folder):
    with np.load(Path(folder)/'trajectory.npz')as a:
        x=np.asarray(a['surface_positions_m']);times=a['times_s'].copy();rigid=a['rigid_centers_m'].copy()if 'rigid_centers_m'in a else None
        if np.iscomplexobj(x):
            if np.any(x.imag!=0):raise ValueError('Nonreal trajectory')
            x=x.real
        if not np.isfinite(x).all()or not np.isfinite(times).all():raise ValueError('Nonfinite complete trajectory')
        return x.copy(),times,rigid


def score(campaign):
    campaign=Path(campaign);rows=census(campaign)
    if any(r['state']=='pending'for r in rows):raise ValueError('Pending predictions: no held reference targets were opened')
    from scripts.complex_scenes_v18 import impulse_metrics,contact_metrics,common_volume_metrics
    for row in rows:
        if row['state']!='sealed':continue
        paths=list((campaign/'references'/row['world_id']/str(row['query_index'])).glob('*/complete.json'))
        if len(paths)!=1:raise ValueError('Missing or ambiguous declared reference')
        reference=paths[0].parent;actual=json.loads(paths[0].read_text())
        if actual['private_world_sha256']!=row['world_material_commitment']or actual['public_query']!=row['public_query']:raise ValueError('Reference world/query mismatch')
        if not actual['completed']:
            row.update(state='reference_invalid',reference_error=actual.get('error'));continue
        if sha(reference/'trajectory.npz')!=actual['trajectory_sha256']:raise ValueError('Changed reference trajectory')
        predicted=Path(row['physical_output']);a,t,rigid_a=load(reference);b,s,rigid_b=load(predicted)
        if a.shape!=b.shape or not np.allclose(t,s,atol=1e-9,rtol=0):raise ValueError('Complete identical physical clocks and material points required')
        row.update(state='complete',metrics=metrics(a,b),reference_sha256=actual['trajectory_sha256'],reference_minimum_dense_J=actual.get('minimum_dense_J'),numerical_reference_qualified=actual.get('reference_qualified',False))
        row['impulse']=impulse_metrics.compare(reference,predicted)
        row['sampled_volume']=common_volume_metrics.audit(predicted)
        query=Query(**row['public_query'])
        row['contact']=contact_metrics.compare(contact_metrics.trace(query,a,t,rigid_a),contact_metrics.trace(query,b,s,rigid_b))
    aggregates=[]
    for family in [None]+sorted({r['family']for r in rows}):
        selected=rows if family is None else[r for r in rows if r['family']==family]
        for baseline in METHODS[1:]:
            result=paired(selected,'ours',baseline);result['family']=family or'all';aggregates.append(result)
    return dict(scope='Held changed-interaction queries on independent material worlds; full-query errors include initial frame; unknown material and known boundaries; common appearance evaluated separately',rows=rows,paired_world_aggregates=aggregates,public_manifest_sha256=sha(campaign/'public-manifest.json'),numerical_reference_note='Read accompanying n4/n6 sensitivity; successful reference completion alone is not spatial convergence evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    write(a.output,score(a.campaign))
