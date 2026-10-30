"""External data adapters. Predictor payloads contain no future object motion.

Synthetic sparse geometry observations for SoftVTBench are explicitly privileged;
PhysTwin uses the authors' processed visible tracks. Evaluation lives separately.
"""
from pathlib import Path
import sys, pickle, json
import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.external_benchmark.common import ROOT, BASE


def fps(points, count):
    points = np.asarray(points)
    ids = [int(np.argmin(np.sum((points-points.mean(0))**2, axis=1)))]; d = np.full(len(points), np.inf)
    for _ in range(min(count, len(points))-1):
        d = np.minimum(d, np.sum((points-points[ids[-1]])**2, axis=1)); d[ids] = -1
        ids.append(int(np.argmax(d)))
    return np.asarray(ids)


def quat_rotation(q):
    # Isaac uses wxyz, scipy uses xyzw.
    return Rotation.from_quat(np.asarray(q)[..., [1, 2, 3, 0]]).as_matrix()


def rigid_map(source, target):
    a, b = source.mean(0), target.mean(0)
    u, _, v = np.linalg.svd((source-a).T @ (target-b))
    d = np.eye(3); d[-1,-1] = np.linalg.det(u@v)
    R = u@d@v
    return R, b-a@R


def interpolate_matrix(nodes, points):
    tri = Delaunay(nodes); s = tri.find_simplex(points)
    mat = np.zeros((len(points),len(nodes)))
    inside = s >= 0; si = s[inside]
    b = np.einsum('nij,nj->ni',tri.transform[si,:3],points[inside]-tri.transform[si,3])
    b = np.c_[b,1-b.sum(1)]
    mat[np.where(inside)[0][:,None],tri.simplices[si]] = b
    outside = np.where(~inside)[0]
    d, idx = cKDTree(nodes).query(points[outside],k=min(4,len(nodes)))
    w = 1/np.maximum(d,1e-9)**2; w /= w.sum(1,keepdims=True)
    mat[outside[:,None],idx]=w
    return mat


def phystwin(case, max_nodes=180):
    directory = ROOT/'runs/third_party/PhysTwin/data/different_types'/case
    with open(directory/'final_data.pkl','rb') as f: raw=pickle.load(f)
    split=json.loads((directory/'split.json').read_text()); train_end=split['train'][1]
    transform=np.diag([1.,-1.,-1.])  # PhysTwin z is down; proper 180-degree rotation.
    full=raw['object_points'] @ transform
    ctrl=raw['controller_points'] @ transform
    initial=np.concatenate([full[0],raw['surface_points']@transform,raw['interior_points']@transform])
    nodes=initial[fps(initial,max_nodes)]
    # Retain the full observed point set used by the official method. Only the
    # internal mechanics mesh is reduced, not the observations available to fit.
    select=np.arange(len(full[0]))
    mask=(raw['object_visibilities'][:train_end,select].astype(bool) & raw['object_motions_valid'][:train_end,select].astype(bool))
    # Controller point identities are known input trajectories, not estimated from targets.
    groups=np.zeros(ctrl.shape[1],int)
    if case.startswith('double'):
        from scipy.cluster.vq import kmeans2
        _,groups=kmeans2(ctrl[0],2,minit='++',seed=42)
    group_labels=np.unique(groups)
    targets=[]; weights=[]
    for label in group_labels:
        c=ctrl[:,groups==label]
        distances,indices=cKDTree(c[0]).query(nodes,k=min(3,len(c[0])))
        d=distances[:,0]
        w=np.exp(-.5*(d/.015)**2)
        # Local tracked control displacement permits nonrigid hand motion;
        # fitting a single hand rotation can incorrectly torque the whole patch.
        proximity=1/np.maximum(distances,1e-6)**2;proximity/=proximity.sum(1,keepdims=True)
        target=nodes[None]+np.einsum('nk,tnkj->tnj',proximity,c[:,indices]-c[0,indices])
        weights.append(w);targets.append(target)
    return dict(name=case,dataset='PhysTwin',nodes=nodes,support_geometry=initial,observed=full[:train_end,select],
        observed_initial=full[0,select],observed_mask=mask,observation_matrix=interpolate_matrix(nodes,full[0,select]),
        controller_targets=np.asarray(targets).transpose(1,0,2,3),attachment_weights=np.asarray(weights),
        attachment_active=np.ones((len(ctrl),len(weights))),floor=float(initial[:,2].min()),
        train_end=train_end,times=np.arange(len(ctrl))/30,controls=ctrl,
        evaluation=dict(points=full,mask=raw['object_visibilities'].astype(bool),transform=transform,
                        surface_count=len(full[0])+len(raw['surface_points']),
                        manual=pickle.load(open(directory/'gt_track_3d.pkl','rb'))@transform),
        source=str(directory/'final_data.pkl'))


def softvt(subset,task,episode,observation_count=48):
    import h5py
    family='object' if subset=='object-soft' else 'spatial'
    directory=BASE/'raw/SoftVTBench'/subset/f'libero_{family}'/f'libero_{family}_task{task}'/'replayed_demos'
    path=next(directory.glob('*.hdf5'))
    with h5py.File(path) as f:
        group=f[f'data/demo_{episode}'];asset=group.attrs['asset_name']
        x=group[f'states/deformable_object/{asset}/nodal_pos_w'][:].astype(float)
        local=group['obs/eef_pose'][:].astype(float)
        base=group['initial_state/articulation/robot/root_pose'][0].astype(float)
        width=group['obs/gripper_width'][:].astype(float)
        command=group['obs/gripper_binary'][:].astype(float)
    R0=quat_rotation(base[3:]); rotations=np.einsum('ij,tjk->tik',R0,quat_rotation(local[:,3:]))
    control=local[:,:3]@R0.T+base[:3]
    nodes=x[0]; T=len(x); train_end=int(.6*T)
    # Input-only reference contact: first closing gripper sufficiently close to rest object.
    distance=np.linalg.norm(control-nodes.mean(0),axis=1)
    contact=np.flatnonzero((width<.065)&(distance<.10)&(np.arange(T)<train_end))
    if not len(contact):raise ValueError('No input-defined contact in observed prefix')
    # The first closing command can occur while the hand is still descending.
    # Define the attachment offset at closest approach to the INITIAL object,
    # using the observed-prefix tool poses only (no later object states).
    grasp=int(contact[np.argmin(distance[contact])])
    upper=np.flatnonzero(nodes[:,2]>=np.quantile(nodes[:,2],.4))
    observed_ids=upper[fps(nodes[upper],observation_count)]
    # Calibrate the gripper/object transform from the SAME sparse observed prefix.
    # The first grasp shape can already have translated relative to initial rest.
    grasp_R,grasp_b=rigid_map(nodes[observed_ids],x[grasp,observed_ids])
    grasp_geometry=nodes@grasp_R+grasp_b
    rest_local=(grasp_geometry-control[grasp])@rotations[grasp]
    transverse=rest_local[:,1]; mid=(transverse.max()+transverse.min())/2
    span=transverse.max()-transverse.min()
    weights=[]; targets=[]
    for sign in [-1,1]:
        face=mid+sign*span/2
        weights.append(np.exp(-.5*((transverse-face)/max(.004,span*.12))**2))
        local_target=np.broadcast_to(rest_local,(T,len(nodes),3)).copy()
        squeeze=np.minimum(width/span,1.)
        local_target[:,:,1]=mid+(transverse-mid)[None]*squeeze[:,None]
        targets.append(np.einsum('tnj,tij->tni',local_target,rotations)+control[:,None])
    active=((np.arange(T)>=grasp)&(width<span*1.15)&(command<0)).astype(float)
    observation_matrix=np.eye(len(nodes))[observed_ids]
    # No material/force/state-derived pose after t=0 is exposed outside evaluation.
    return dict(name=f'{subset}-task{task}-demo{episode}',dataset='SoftVTBench',nodes=nodes,
        observed=x[:train_end,observed_ids],observed_initial=nodes[observed_ids],
        observed_mask=np.ones((train_end,len(observed_ids)),bool),observation_matrix=observation_matrix,
        controller_targets=np.asarray(targets).transpose(1,0,2,3),attachment_weights=np.asarray(weights),
        attachment_active=np.repeat(active[:,None],2,axis=1),floor=float(nodes[:,2].min()),
        train_end=train_end,times=np.arange(T)/20,controls=control[:,None],
        evaluation=dict(points=x,mask=np.ones(x.shape[:2],bool),transform=np.eye(3)),
        source=str(path),grasp=grasp,gripper_width=width,gripper_command=command)


def predictor_payload(data):
    return {k:v for k,v in data.items() if k!='evaluation'}
