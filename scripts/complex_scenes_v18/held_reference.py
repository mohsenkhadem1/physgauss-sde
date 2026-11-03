"""One restart-safe reference task from an immutable private held manifest."""
import argparse,json
from dataclasses import asdict
from pathlib import Path
from scripts.complex_scenes_v18.held_manifest import sha,validate_freeze
from scripts.complex_scenes_v18.world_queries import World,Material,Query
from scripts.complex_scenes_v18.run_development import ROOT,run


def execute(campaign,freeze_path,world_id,slot,refine=False,seconds=10800):
    campaign=Path(campaign);freeze_path=Path(freeze_path)
    seal=json.loads((campaign/'manifest-seal.json').read_text())
    if sha(freeze_path)!=seal['freeze_sha256']:raise ValueError('Changed frozen protocol')
    for filename,key in [('private-worlds.json','private_worlds_sha256'),('public-manifest.json','public_manifest_sha256')]:
        if sha(campaign/filename)!=seal[key]:raise ValueError('Changed world manifest')
    protocol=validate_freeze(json.loads(freeze_path.read_text()),ROOT)
    public=[w for w in json.loads((campaign/'public-manifest.json').read_text())['worlds']if w['world_id']==world_id]
    private=[w for w in json.loads((campaign/'private-worlds.json').read_text())['worlds']if w['world_id']==world_id]
    if len(public)!=1 or len(private)!=1:raise ValueError('Unknown world identity')
    row=public[0];raw=private[0];world=World(raw['world_id'],tuple(Material(**m)for m in raw['materials']))
    if world.digest!=row['world_material_commitment']:raise ValueError('Material commitment mismatch')
    if refine and (not row['refinement']or slot=='calibration'):raise ValueError('Refinement outside prospective census')
    query=Query(**(row['calibration']if slot=='calibration'else row['queries'][int(slot)]))
    settings=protocol['reference'];n=6 if refine else settings['mesh_n'];dt=settings['dt_s']
    # Campaign must be local; labels cannot escape runs/complex-scenes-v18.
    relative=campaign.resolve().relative_to((ROOT/'runs/complex-scenes-v18').resolve())
    label=str(relative/'references'/world_id/(str(slot)+('-refined'if refine else'')))
    surface=0 if query.family=='deformable_pair'else 5
    destination=ROOT/'runs/complex-scenes-v18'/label/f'{query.family}-v{query.variant}-n{n}-dt{dt:.8f}-s{surface}'
    if destination.exists():
        done=destination/'complete.json'
        if not done.exists():raise RuntimeError('Existing unfinished task: inspect its process before retrying')
        result=json.loads(done.read_text())
        if result['private_world_sha256']!=world.digest or result['public_query']!=asdict(query) or result['n']!=n or result['dt_s']!=dt:raise ValueError('Existing task identity mismatch')
        if result['completed'] and sha(destination/'trajectory.npz')!=result['trajectory_sha256']:raise ValueError('Existing reference changed')
        return result
    return run(query.family,n,dt,variant=query.variant,label=label,surface_samples=surface,seconds=seconds,minimum_j=settings['minimum_nodal_quadrature_J_guard'],dense_j_samples=settings['dense_J_samples_per_axis'],bulk_quadrature=settings['quadrature'],world=world,query=query,split='held')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--freeze',required=True);p.add_argument('--world-id',required=True);p.add_argument('--slot',choices=['calibration','0','1'],required=True);p.add_argument('--refine',action='store_true');p.add_argument('--seconds',type=int,default=10800);a=p.parse_args()
    r=execute(a.campaign,a.freeze,a.world_id,a.slot,a.refine,a.seconds);print(json.dumps({k:r.get(k)for k in ['completed','path','error','minimum_dense_J']}));raise SystemExit(0 if r['completed']else 1)
