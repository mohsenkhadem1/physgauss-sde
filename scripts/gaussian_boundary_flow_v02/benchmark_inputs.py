"""Strict input/evaluation separation for exposed benchmark development cases.

PhysTwin's pickle must be materialized in an isolated exporter; only prefix
motion leaves that scope. Its offline upstream tracking is not certified causal.
SoftVT HDF5 object arrays are sliced BEFORE reading. Fresh episodes are refused
until a later, explicit frozen-protocol adapter is implemented. Inventory reads
keys/attributes only. No force, material, deformation-summary or suffix labels
are returned by predictor loaders.
"""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
from scipy.spatial import cKDTree
from scripts.external_benchmark.data import fps, interpolate_matrix, phystwin_position_mask
from scripts.prefix_ridge_switch_v13.inputs import softvt_payload

ROOT = Path(__file__).resolve().parents[2]
PHYSTWIN = ROOT/'runs/editable-scene-v23/phystwin-full-rgb/data/different_types'
SOFTVT = ROOT/'runs/external-benchmark-v11/raw/SoftVTBench'
MASK_VERSION = 'incoming-v22'


def _case_path(root, case):
    if not case or Path(case).name != case:
        raise ValueError('A single case name, not a relative path, is required')
    return Path(root)/case


def _cut(requested, official):
    cut = int(official if requested is None else requested)
    if cut < 3 or cut > official:
        raise ValueError('Cutoff must remain within the official observed prefix')
    return cut


def validate_predictor(data):
    forbidden = {'evaluation','manual','points','object_points','forces','material',
                 'finger_force','gripper_net_force','fem_deformation_rms'}
    if forbidden.intersection(data):
        raise ValueError('Evaluator/privileged target fields in predictor package')
    end = int(data['train_end'])
    if len(data['observed']) != end or data['observed_mask'].shape != data['observed'].shape[:2]:
        raise ValueError('Prefix observation shape mismatch')
    if not np.isfinite(data['observed'][data['observed_mask']]).all():
        raise ValueError('Valid observations must be finite')
    if len(data['controller_targets']) != len(data['times']):
        raise ValueError('Complete prescribed controller schedule required')
    return data


def load_phystwin_predictor(case, max_nodes=120, prefix_end=None,
                             include_rgb_paths=False, root=PHYSTWIN):
    if not isinstance(max_nodes,int) or max_nodes < 4:
        raise ValueError('At least four mechanical nodes required')
    folder = _case_path(root,case)
    split = json.loads((folder/'split.json').read_text())
    end = _cut(prefix_end,split['train'][1])
    with (folder/'final_data.pkl').open('rb') as f:
        raw = pickle.load(f)
    # Drop all future object/visibility/motion validity arrays at export boundary.
    allowed = {k:np.asarray(raw[k][:end]).copy() for k in
               ('object_points','object_visibilities','object_motions_valid')}
    allowed.update({k:np.asarray(raw[k]).copy() for k in
                    ('controller_points','surface_points','interior_points')})
    del raw
    transform = np.diag([1.,-1.,-1.])
    observed = allowed['object_points'] @ transform
    mask = phystwin_position_mask(allowed,end)
    initial = np.concatenate([observed[0],allowed['surface_points']@transform,
                              allowed['interior_points']@transform])
    if not np.isfinite(initial).all():
        raise ValueError('Initial geometry must be finite')
    nodes = initial[fps(initial,max_nodes)]
    controls = allowed['controller_points']@transform
    groups = np.zeros(controls.shape[1],int)
    if case.startswith('double') or '_double_' in case:
        from scipy.cluster.vq import kmeans2
        _,groups = kmeans2(controls[0],2,minit='++',seed=42)
    weights,targets = [],[]
    for label in np.unique(groups):
        c = controls[:,groups==label]
        distances,indices = cKDTree(c[0]).query(nodes,k=min(3,len(c[0])))
        if distances.ndim == 1:
            distances,indices = distances[:,None],indices[:,None]
        proximity = 1/np.maximum(distances,1e-6)**2
        proximity /= proximity.sum(1,keepdims=True)
        weights.append(np.exp(-.5*(distances[:,0]/.015)**2))
        targets.append(nodes[None]+np.einsum('nk,tnkj->tnj',proximity,c[:,indices]-c[0,indices]))
    data = dict(name=case,dataset='PhysTwin',nodes=nodes,support_geometry=initial,
                observed=observed,observed_initial=observed[0].copy(),observed_mask=mask,
                observation_matrix=interpolate_matrix(nodes,observed[0]),
                controller_targets=np.asarray(targets).transpose(1,0,2,3),
                attachment_weights=np.asarray(weights),
                attachment_active=np.ones((len(controls),len(weights))),
                floor=float(initial[:,2].min()),train_end=end,times=np.arange(len(controls))/30,
                controls=controls,source=str(folder/'final_data.pkl'),mask_version=MASK_VERSION,
                observation_contract='Supplied initial geometry and prefix 3D tracks; upstream offline causality unaudited',
                boundary_contract='attachment_weights are a proximity initialization/prior, NOT known true contact; learn or ablate them',
                units='SI metres, seconds; no measured forces/materials provided')
    if include_rgb_paths:
        rgb = [[str(folder/'color'/str(c)/f'{t}.png') for c in range(3)] for t in range(end)]
        # Keep JSON-serializable paths; caller explicitly chooses input cameras.
        data['prefix_rgb_paths'] = rgb
        data['rgb_camera_ids'] = [0,1,2]
        data['camera_metadata_path'] = str(folder/'metadata.json')
        data['camera_calibration_path'] = str(folder/'calibrate.pkl')
        data['rgb_contract'] = 'Prefix paths only; choose held camera before decoding; no future RGB paths'
    return validate_predictor(data)


def _softvt_path(subset,task,root):
    if subset not in ('object-soft','spatial-soft') or task not in (0,1,2):
        raise ValueError('Unknown local SoftVT suite/task')
    family = 'object' if subset == 'object-soft' else 'spatial'
    candidates = list((Path(root)/subset/f'libero_{family}'/f'libero_{family}_task{task}'/'replayed_demos').glob('*.hdf5'))
    if len(candidates) != 1:raise ValueError('Expected exactly one local HDF5 file')
    return candidates[0]


def _require_exposed(task,episode):
    if not 0 <= episode <= (9 if task == 0 else 49):
        raise PermissionError('Candidate fresh episode: metadata only until a new protocol is frozen')


def load_softvt_predictor(subset,task,episode,prefix_end=None,root=SOFTVT):
    _require_exposed(task,episode)
    import h5py
    path = _softvt_path(subset,task,root)
    with h5py.File(path) as f:
        g = f[f'data/demo_{episode}'];asset = g.attrs['asset_name']
        T = len(g['obs/eef_pose']);end = _cut(prefix_end,int(.6*T))
        raw = dict(object_prefix=g[f'states/deformable_object/{asset}/nodal_pos_w'][:end],
                   eef_pose=g['obs/eef_pose'][:],robot_root=g['initial_state/articulation/robot/root_pose'][0],
                   width=g['obs/gripper_width'][:],command=g['obs/gripper_binary'][:])
    data = softvt_payload(raw,subset,task,episode,str(path),end)
    data.update(observation_contract='Privileged initial volume mesh and 48 sparse prefix node positions; local HDF5 has no RGB',
                boundary_contract='Hand-proximity attachments are initialized priors, not true contacts',
                units='SI metres, seconds; force/deformation diagnostic channels withheld')
    return validate_predictor(data)


def load_phystwin_evaluation(case,root=PHYSTWIN):
    """EVALUATOR ONLY: never import/call from a predictor/training forward pass."""
    folder = _case_path(root,case);transform=np.diag([1.,-1.,-1.])
    with (folder/'final_data.pkl').open('rb') as f:raw=pickle.load(f)
    with (folder/'gt_track_3d.pkl').open('rb') as f:manual=pickle.load(f)
    return dict(points=np.asarray(raw['object_points'])@transform,
                mask=np.asarray(raw['object_visibilities'],bool),
                manual=np.asarray(manual)@transform,transform=transform)


def load_softvt_evaluation(subset,task,episode,root=SOFTVT):
    """EVALUATOR ONLY, currently restricted to historical development cases."""
    _require_exposed(task,episode)
    import h5py
    with h5py.File(_softvt_path(subset,task,root)) as f:
        g=f[f'data/demo_{episode}'];asset=g.attrs['asset_name']
        points=g[f'states/deformable_object/{asset}/nodal_pos_w'][:]
    return dict(points=points,mask=np.ones(points.shape[:2],bool),transform=np.eye(3))


def metadata_inventory(softvt_root=SOFTVT,phystwin_root=PHYSTWIN):
    """No motion arrays: file paths, keys, attributes, splits and RGB file counts."""
    import h5py
    soft=[]
    for subset in ('object-soft','spatial-soft'):
        for task in range(3):
            path=_softvt_path(subset,task,softvt_root)
            with h5py.File(path) as f:
                ids=sorted(int(k.split('_')[-1])for k in f['data'])
                assets=sorted({str(f['data'][k].attrs['asset_name'])for k in f['data']})
            soft.append(dict(subset=subset,task=task,path=str(path),episode_ids=ids,assets=assets,
                             candidate_fresh_ids=[i for i in ids if i>(9 if task==0 else 49)]))
    phys=[]
    for folder in sorted(Path(phystwin_root).iterdir()):
        if not(folder/'final_data.pkl').exists():continue
        phys.append(dict(case=folder.name,path=str(folder),split=json.loads((folder/'split.json').read_text()),
                         has_manual=(folder/'gt_track_3d.pkl').exists(),RGB_files=sum(1 for _ in (folder/'color').rglob('*.png'))))
    return dict(SoftVTBench=soft,PhysTwin=phys,access='Metadata only, no trajectory/RGB arrays read')


def predictor_digest(data):
    """Content hash independent of evaluator files and forbidden suffix arrays."""
    validate_predictor(data);h=hashlib.sha256()
    for key,value in sorted(data.items()):
        h.update(key.encode())
        if isinstance(value,np.ndarray):
            h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(np.ascontiguousarray(value).tobytes())
        else:h.update(json.dumps(value,sort_keys=True).encode())
    return h.hexdigest()
