"""Assemble predicted-hypothesis banks, never actual reserved observations."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.physical_refinement_v15.study import development_actions
from scripts.physical_refinement_v15.quadrature import geometry_cases, hypotheses
from scripts.physical_refinement_v15.sensor import SensorSettings
from scripts.physical_refinement_v15.feature_likelihood import covariance
from scripts.phystwin.sequential_fixture import points
from scripts.physical_refinement_v15.belief import Query, DesignResponses

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'runs/physical-refinement-v15/development'
CAMERAS = ('left', 'right', 'oblique')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nonrigid(rest, displacement):
    x = rest-rest.mean(0)
    y = rest+displacement
    y = y-y.mean(0)
    u, _, vt = np.linalg.svd(y.T@x)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u@vt)
    return y@(u@correction@vt)-x


def checked_npz(folder, filename):
    record_path = folder/'complete.json'
    record = json.loads(record_path.read_text())
    path = folder/filename
    if record['artifact_sha256'][filename] != sha(path):
        raise ValueError('Hypothesis bank artifact changed: '+str(path))
    return np.load(path, allow_pickle=False), {str(path.relative_to(ROOT)):sha(path),
                                              str(record_path.relative_to(ROOT)):sha(record_path)}


def assemble(gradient_count=5):
    cases = geometry_cases(gradient_count)
    actions = list(development_actions())
    fields, shapes, wrenches, image_features = [], [], [], []
    evidence = {}
    rest = points()
    for case in cases:
        f, z, w, visual = [], [], [], []
        for index, action in enumerate(actions):
            folder = BASE/'hypotheses'/case['name']/'n4'/f'pose-{index:03d}'
            data, hashes = checked_npz(folder, 'private-state.npz')
            evidence.update(hashes)
            with data:
                field = data['field'].copy()
            f.append(field)
            z.append(nonrigid(rest, field))
            data, hashes = checked_npz(folder, 'reaction-wrench.npz')
            evidence.update(hashes)
            with data:
                w.append(np.r_[data['forces_N'][1], data['torques_Nm'][1]])
            row = []
            render_folder = BASE/'render-bank'/case['name']/action
            for camera in CAMERAS:
                data, hashes = checked_npz(render_folder, camera+'.npz')
                evidence.update(hashes)
                with data:
                    row.append(data['features'].copy())
            visual.append(row)
        fields.append(f); shapes.append(z); wrenches.append(w); image_features.append(visual)
    covariances = []
    settings = SensorSettings()
    for action in actions:
        row = []
        for camera in CAMERAS:
            data, hashes = checked_npz(BASE/'render-bank/parallel_lower-g2-r0'/action, camera+'.npz')
            evidence.update(hashes)
            with data:
                row.append(covariance(data['rgb'], data['depth'], settings.depth_sd_m, settings.rgb_sd))
        covariances.append(row)
    result = dict(fields=np.asarray(fields), shapes=np.asarray(shapes), wrenches=np.asarray(wrenches),
                  features=np.asarray(image_features), covariances=np.asarray(covariances),
                  case_names=[case['name'] for case in cases], actions=actions, cameras=CAMERAS,
                  evidence_sha256=evidence, role='hypothesis_predictions_for_development_design')
    for key in ('fields','shapes','wrenches','features','covariances'):
        if not np.isfinite(result[key]).all():
            raise ValueError('Nonfinite hypothesis predictions: '+key)
    return result


def expand(bank, gradient_count=5, scale_count=25):
    rows, weights = hypotheses(gradient_count, scale_count)
    indices = np.array([bank['case_names'].index(row['base_name']) for row in rows])
    scales = np.array([row['scale'] for row in rows])
    return dict(rows=rows, weights=weights, fields=bank['fields'][indices],
                shapes=bank['shapes'][indices], wrenches=bank['wrenches'][indices]*scales[:,None,None],
                features=bank['features'][indices], covariances=bank['covariances'],
                actions=bank['actions'], cameras=bank['cameras'])


def query(expanded, action, camera, *, include_wrench=False, name=None):
    i = expanded['actions'].index(action)
    j = expanded['cameras'].index(camera)
    means = expanded['features'][:,i,j]
    visual_count = means.shape[1]
    C = expanded['covariances'][i,j].copy()
    if include_wrench:
        means = np.c_[means, expanded['wrenches'][:,i]]
        joint = np.zeros((means.shape[1],means.shape[1]))
        joint[:visual_count,:visual_count] = C
        settings = SensorSettings()
        joint[visual_count:,visual_count:] = np.diag([settings.force_sd_N**2]*3+[settings.torque_sd_Nm**2]*3)
        C = joint
    return Query(name or action+'@'+camera, 'rgbd_force_probe' if include_wrench else 'rgbd',
                 means, C, visual_indices=tuple(range(visual_count)))


def mechanical_design(expanded):
    indices = [i for i,name in enumerate(expanded['actions']) if name.startswith('design_')]
    if len(indices)!=3:
        raise ValueError('Exactly three declared development design actions required')
    shape = expanded['shapes'][:,indices].reshape(len(expanded['rows']),3,-1)
    wrench = expanded['wrenches'][:,indices]
    values = np.concatenate([shape,wrench],axis=-1)
    n = values.shape[-1]
    # Prespecified engineering task normalization. Report raw-unit errors
    # separately; this weighted design utility is not a universal fidelity score.
    block_scales = np.r_[np.full(shape.shape[-1], .001*np.sqrt(shape.shape[-1]/(.5*n))),
                         np.full(3, .1*np.sqrt(3/(.25*n))),
                         np.full(3, .01*np.sqrt(3/(.25*n)))]
    return DesignResponses(values.reshape(len(values),-1),np.tile(block_scales,3))
