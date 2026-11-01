"""Development-only outcome reader. Never imported by the acquisition selector."""
import hashlib
import json
import numpy as np
from scripts.physical_refinement_v15.banks import BASE, ROOT, CAMERAS, checked_npz, query


def menu(initial_camera):
    views = [(f'view_{camera}','initial',camera) for camera in CAMERAS if camera != initial_camera]
    probes = [(name,name,'left') for name in ('probe_left_twist','probe_right_twist','probe_left_bend','probe_right_bend')]
    return views+probes


def prediction_queries(expanded, initial_camera, track):
    if track not in ('V','VF'):
        raise ValueError('Declared observation track required')
    initial = query(expanded,'initial',initial_camera,name='initial_observation')
    candidates = [query(expanded,action,camera,include_wrench=track=='VF' and name.startswith('probe'),name=name)
                  for name,action,camera in menu(initial_camera)]
    return initial,candidates


def observation(world, repeat, action, camera, prediction_query, *, level=6):
    if world not in [f'development-{i}' for i in range(4)] or repeat not in range(4):
        raise ValueError('Only four declared development worlds/repeats are available here')
    folder = BASE/'world-render-bank'/world/f'n{level}'/action
    image, evidence = checked_npz(folder,camera+'.npz')
    with image:
        mean = image['features'].copy()
    if prediction_query.means.shape[1] == 70:
        from scripts.physical_refinement_v15.study import development_actions
        index = list(development_actions()).index(action)
        wrench, hashes = checked_npz(BASE/'worlds'/world/f'n{level}'/f'pose-{index:03d}','reaction-wrench.npz')
        evidence.update(hashes)
        with wrench:
            mean = np.r_[mean,wrench['forces_N'][1],wrench['torques_Nm'][1]]
    if mean.shape!=(prediction_query.means.shape[1],):
        raise ValueError('Observed and predicted modalities differ')
    # Same visual noise for V/VF, and same query outcome for all policies.
    # Independent packets are a declared synthetic-sensor assumption.
    query_seed=int.from_bytes(hashlib.sha256((action+'@'+camera).encode()).digest()[:4],'little')
    seed=np.random.SeedSequence([931109,int(world.rsplit('-',1)[1]),repeat,query_seed])
    rng=np.random.default_rng(seed)
    C=prediction_query.variance
    values=mean+np.linalg.cholesky(C)@rng.normal(size=len(mean))
    return values,evidence


def development_worlds():
    path=ROOT/'reports/physical-refinement-v15/development-truth-plan.json'
    return json.loads(path.read_text())['scenes']
