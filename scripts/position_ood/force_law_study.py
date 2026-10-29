"""Separately predeclared force-law OOD test using unchanged frozen predictors.

This extension adds fields/observations under force_law_test names only. The
original cohorts, method seal, calibration and acquisition study remain intact.
"""
from pathlib import Path
import argparse
import datetime
import json
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.position_ood import study
from scripts.position_ood.seal import verify as verify_main, digest
from scripts.position_ood.transfer import controls, TRANSFER_ACTIONS
from scripts.position_ood.pilot import ACTIONS as LOCAL_ACTIONS
from scripts.position_ood.force_laws import ForceLawFixture
from scripts.phystwin.sequential_reference import write
from scripts.phystwin.sequential_fixture import LAYOUTS, points

BASE, ROOT, OUT = study.BASE, study.ROOT, study.OUT
SPLIT = 'force_law_test'
PLAN = BASE / 'force-law-protocol.json'
SEAL = BASE / 'force-law-seal.json'
DATA_SEAL = BASE / 'force-law-data-seal.json'


def freeze():
    main = verify_main()
    if SEAL.exists():
        return verify()
    assert not list((BASE / 'fields').glob(f'{SPLIT}-*.npz'))
    assert not list((BASE / 'inverse').glob(f'{SPLIT}-*'))
    rng = np.random.default_rng(295001)
    rows = []
    for i in range(6):
        tilt = rng.uniform(-.3, .3, 2)
        gravity = np.r_[tilt, -1.]
        gravity *= 9.81/np.linalg.norm(gravity)
        case = dict(E=float(np.exp(rng.uniform(np.log(8000), np.log(40000)))),
                    Kleft=float(np.exp(rng.uniform(np.log(1500), np.log(15000)))),
                    Kright=float(np.exp(rng.uniform(np.log(1500), np.log(15000)))),
                    layout=LAYOUTS[i % 3], material_gradient=float(rng.uniform(-.5, .5)),
                    gravity=gravity.tolist())
        if i < 3:
            rotation = Rotation.from_rotvec(rng.uniform(-.5, .5, 3)).as_matrix()
            matrix = rotation @ np.diag([10., 25., 40.]) @ rotation.T
            law = dict(kind='anisotropic_harmonic', center=[.12, 0., -.06],
                       stiffness_N_per_m=matrix.tolist())
        else:
            law = dict(kind='two_gaussian', components=[
                dict(center=[.07, -.035, -.06], width_m=.045,
                     amplitude_J=float(.07*rng.uniform(.85, 1.15))),
                dict(center=[.17, .035, -.09], width_m=.055,
                     amplitude_J=float(-.09*rng.uniform(.85, 1.15)))])
        rows.append(dict(**case, force_law=law))
    plan = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                seed=295001, observation_seed=296000,
                actions=[*LOCAL_ACTIONS, *TRANSFER_ACTIONS[3:]],
                reference_levels=study.protocol()['reference_levels'],
                noise_sigma_m=.0003, **{SPLIT: rows},
                groups=dict(prefix=[0, 1, 2], new_interaction=[4, 5, 6, 7, 8],
                            larger_amplitude=[9], spatial_transfer=list(range(10, 16))),
                methods=['extrapolated', 'extrapolated_material_frame',
                         'neural_id', 'neural_all', 'phystwin', 'mpm'],
                iterations=main['settings']['iterations'],
                scope='Six separately declared objects, three per unseen force law with one per attachment layout. Material and coupling coefficients are drawn within familiar ranges but remain hidden. Same original three position observations and all frozen predictor settings. No method tuning, no new training, no force labels, no probe acquisition in this extension. Report all six objects and both force laws, including failures. This extension was declared after the primary method freeze but before generating any of its reference fields or fitting its objects.')
    write(PLAN, plan)
    paths = [PLAN, BASE/'method-seal.json', Path(__file__).resolve(),
             ROOT/'scripts/position_ood/force_laws.py', ROOT/'tests/test_position_force_laws.py']
    record = dict(created_utc=plan['created_utc'],
                  hashes={str(p.relative_to(ROOT)):digest(p) for p in paths})
    write(SEAL, record)
    write(OUT/'force-law-protocol.json', plan)
    write(OUT/'force-law-seal.json', record)
    return verify()


def verify():
    verify_main()
    record = json.loads(SEAL.read_text())
    for name, expected in record['hashes'].items():
        if digest(ROOT/name) != expected:
            raise RuntimeError(f'Force-law protocol changed: {name}')
    return json.loads(PLAN.read_text())


def observations(i):
    from scripts.phystwin.v06_camera import render, track
    dest = BASE/f'observations/{SPLIT}-{i}.npz'
    if dest.exists():
        return
    ids = np.flatnonzero(np.isclose(points()[:, 1], -.04))
    seed = verify()['observation_seed']+i
    rng = np.random.default_rng(seed)
    with np.load(BASE/f'fields/{SPLIT}-{i}.npz') as data:
        measured = []
        for field in data['field']:
            rgb, depth, camera = render(points()+field, 0, ids)
            measured.append(track(rgb, depth, camera, ids, rng)-points()[ids])
        poses = data['poses'].copy()
    temp = dest.with_suffix('.tmp.npz')
    np.savez_compressed(temp, measurements=measured, visible_ids=ids, poses=poses, seed=seed)
    temp.replace(dest)


def reference(i):
    plan = verify()
    # Process-local reference factory; the inverse backend is never modified.
    import scripts.position_ood.smooth_fixture as smooth
    smooth.SmoothMaterialFixture = ForceLawFixture
    study.protocol = lambda: plan
    study.ACTIONS = plan['actions']
    study.controls = controls
    study.reference(SPLIT, i)
    observations(i)


def seal_data():
    verify()
    paths = [BASE/f'{folder}/{SPLIT}-{i}.{extension}' for i in range(6)
             for folder, extension in [('fields', 'npz'), ('qualification', 'json'), ('observations', 'npz')]]
    for i in range(6):
        assert json.loads((BASE/f'qualification/{SPLIT}-{i}.json').read_text())['qualified']
    record = dict(hashes={str(p.relative_to(ROOT)):digest(p) for p in paths})
    if DATA_SEAL.exists():
        assert json.loads(DATA_SEAL.read_text()) == record, 'Force-law data changed'
    else:
        write(DATA_SEAL, record)
        write(OUT/'force-law-data-seal.json', record)
    return record


def run(task, i, method):
    study.within_deadline()
    if task == 'freeze':
        freeze()
        return
    plan = verify()
    if task == 'reference':
        reference(i)
        return
    seal_data()
    if task == 'seal-data':
        return
    if task == 'inverse':
        from scripts.position_ood.run_inverse import run as fit
        fit(SPLIT, i, method, 3, plan['iterations'])
    elif task == 'neural':
        from scripts.position_ood.predict_selected import run as predict
        for mode in ['id', 'all']:
            predict(SPLIT, i, mode)
    elif task in ['phystwin', 'mpm']:
        spec = verify_main()['settings'][task]
        script = 'phystwin.py' if task == 'phystwin' else 'fit_mpm.py'
        subprocess.run([str(ROOT/'runs/phystwin-env/bin/python'),
                        str(ROOT/'scripts/position_ood'/script), '--split', SPLIT,
                        '--object', str(i), *map(str, spec['args'])], check=True, cwd=ROOT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['freeze', 'reference', 'seal-data', 'inverse', 'neural', 'phystwin', 'mpm'], required=True)
    parser.add_argument('--object', type=int, default=0)
    parser.add_argument('--method', choices=['extrapolated', 'extrapolated_material_frame'], default='extrapolated')
    args = parser.parse_args()
    run(args.task, args.object, args.method)
