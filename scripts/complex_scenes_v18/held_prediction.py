"""Replay a held-world fit using only its sealed prefix and public query manifest.

This worker never opens private-worlds.json or reference trajectories. Each
prediction is sealed before downstream scoring is allowed to read its target.
"""
import argparse,json
from pathlib import Path
from scripts.complex_scenes_v18.held_manifest import sha
from scripts.complex_scenes_v18.world_queries import Query
from scripts.complex_scenes_v18.run_development import ROOT,write


def context(campaign,fitdir,world_id,index):
    campaign=Path(campaign);fitdir=Path(fitdir)
    seal=json.loads((campaign/'manifest-seal.json').read_text())
    if sha(campaign/'public-manifest.json')!=seal['public_manifest_sha256']:raise ValueError('Changed public manifest')
    public=json.loads((campaign/'public-manifest.json').read_text())
    if public['split']!='held' or public['freeze_sha256']!=seal['freeze_sha256']:raise ValueError('Held protocol identity mismatch')
    rows=[w for w in public['worlds']if w['world_id']==world_id]
    if len(rows)!=1 or index not in (0,1):raise ValueError('Unknown public query')
    row=rows[0];fit=json.loads((fitdir/'fit-seal.json').read_text());prefix=Path(fit['calibration_prefix'])
    if not fit['completed'] or fit['forecast_targets_read'] or fit['evaluation_split']!='held':raise ValueError('Completed held calibration seal required')
    if sha(prefix/'metadata.json')!=fit['prefix_metadata_sha256'] or sha(prefix/'observations.npz')!=fit['observations_sha256']:raise ValueError('Calibration prefix changed')
    meta=json.loads((prefix/'metadata.json').read_text())
    if meta['world_id']!=world_id or meta['world_material_commitment']!=row['world_material_commitment'] or meta['public_query']!=row['calibration'] or meta['split']!='held':raise ValueError('Fit belongs to a different world or calibration')
    if fit['public_query']!=row['calibration']:raise ValueError('Fit action identity mismatch')
    return fit,Query(**row['queries'][index])


def execute(campaign,fitdir,world_id,index,method,seconds=3600):
    campaign=Path(campaign);fitdir=Path(fitdir);fit,query=context(campaign,fitdir,world_id,index)
    relative=campaign.resolve().relative_to((ROOT/'runs/complex-scenes-v18').resolve())
    label=relative/'predictions'/method/world_id/f'query{index}';out=ROOT/'runs/complex-scenes-v18'/label
    if out.exists():raise ValueError('Preserve existing prediction; inspect its process/seal before retrying')
    if method in ('ours','analytic-control'):
        law='learned'if method=='ours'else'analytic'
        if fit['law_type']!=law:raise ValueError('Wrong fitted material method')
        from scripts.complex_scenes_v18.material_rollout import run
        result=run(query.family,query.variant,fit['best']['parameters'],label=str(label),query=query,law_type=law,split='held',seconds=seconds)
        physical=out/f'{query.family}-v{query.variant}'
    elif method in ('spring-gaus','phystwin'):
        if fit['method']!=method:raise ValueError('Wrong fitted spring method')
        from scripts.complex_scenes_v18.replay_query import replay
        result=replay(fitdir,query,out,seconds);physical=out
    elif method=='uniphy':
        if query.family!='container_drop' or 'UniPhy'not in fit['method']:raise ValueError('Unsupported UniPhy adapter')
        from scripts.complex_scenes_v18.uniphy_replay_query import replay
        result=replay(fitdir,query,out,seconds);physical=out
    else:raise ValueError('Unknown benchmark method')
    write(out/'prediction-seal.json',dict(completed=result['completed'],evaluation_split='held',method=method,world_id=world_id,query_index=index,fit_seal_sha256=sha(fitdir/'fit-seal.json'),public_manifest_sha256=sha(campaign/'public-manifest.json'),physical_output=str(physical.relative_to(ROOT)),physical_complete_sha256=sha(physical/'complete.json'),forecast_targets_read=False,error=result.get('error')))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--fitdir',required=True);p.add_argument('--world-id',required=True);p.add_argument('--index',type=int,choices=[0,1],required=True);p.add_argument('--method',choices=['ours','analytic-control','spring-gaus','phystwin','uniphy'],required=True);a=p.parse_args()
    r=execute(a.campaign,a.fitdir,a.world_id,a.index,a.method);print(json.dumps(dict(completed=r['completed'],error=r.get('error'))));raise SystemExit(0 if r['completed']else 1)
