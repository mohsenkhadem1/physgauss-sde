"""Input-only exporters. No evaluator dictionary/manual tracks enter constructors.
PhysTwin pickle must be materialized in an isolated exporter; only allowed arrays
are persisted. SoftVT object states are sliced at the observed cutoff in HDF.
"""
from pathlib import Path
import json,pickle
import numpy as np
from scipy.spatial import cKDTree
from scripts.external_benchmark.data import fps,quat_rotation,rigid_map,interpolate_matrix
from scripts.external_benchmark.common import ROOT,BASE
def phystwin_payload(raw, case, train_end, source="", max_nodes=300):
    transform=np.diag([1.,-1.,-1.])  # PhysTwin z is down; proper 180-degree rotation.
    full=np.asarray(raw['object_points'][:train_end]) @ transform
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
        source=source)

def softvt_payload(raw, subset, task, episode, source="", end=None, observation_count=48):
    local=np.asarray(raw['eef_pose'],float);base=np.asarray(raw['robot_root'],float)
    width=np.asarray(raw['width'],float);command=np.asarray(raw['command'],float)
    T=len(local);train_end=int(.6*T) if end is None else int(end)
    x=np.asarray(raw['object_prefix'][:train_end],float)
    if len(x)!=train_end:raise ValueError('Insufficient permitted prefix')
    R0=quat_rotation(base[3:]); rotations=np.einsum('ij,tjk->tik',R0,quat_rotation(local[:,3:]))
    control=local[:,:3]@R0.T+base[:3]
    nodes=x[0]
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
    # Attachment pose uses permitted observed-prefix grasp only; no suffix object state or material/force labels.
    return dict(name=f'{subset}-task{task}-demo{episode}',dataset='SoftVTBench',nodes=nodes,
        observed=x[:train_end,observed_ids],observed_initial=nodes[observed_ids],
        observed_mask=np.ones((train_end,len(observed_ids)),bool),observation_matrix=observation_matrix,
        controller_targets=np.asarray(targets).transpose(1,0,2,3),attachment_weights=np.asarray(weights),
        attachment_active=np.repeat(active[:,None],2,axis=1),floor=float(nodes[:,2].min()),
        train_end=train_end,times=np.arange(T)/20,controls=control[:,None],
        source=source,grasp=grasp,gripper_width=width,gripper_command=command)


def load_pair(spec):
    if 'case' in spec:
        case=spec['case'];directory=ROOT/'runs/third_party/PhysTwin/data/different_types'/case
        end=json.loads((directory/'split.json').read_text())['train'][1]
        with open(directory/'final_data.pkl','rb') as f:raw=pickle.load(f)
        # Quarantine export: forbidden future object/visibility fields never
        # leave this scope, and manual target file is never opened.
        allowed={k:np.asarray(raw[k][:end]).copy() for k in ('object_points','object_visibilities','object_motions_valid')}
        allowed.update({k:np.asarray(raw[k]).copy() for k in ('controller_points','surface_points','interior_points')});del raw
        full=phystwin_payload(allowed,case,end,str(directory/'final_data.pkl'))
        short=dict(full,observed=full['observed'][:max(3,int(end*.7))].copy(),observed_mask=full['observed_mask'][:max(3,int(end*.7))].copy(),train_end=max(3,int(end*.7)))
        return full,short
    import h5py
    subset,task,episode=spec['subset'],spec['task'],spec['episode'];family='object' if subset=='object-soft' else 'spatial'
    path=next((BASE/'raw/SoftVTBench'/subset/f'libero_{family}'/f'libero_{family}_task{task}'/'replayed_demos').glob('*.hdf5'))
    with h5py.File(path) as f:
        g=f[f'data/demo_{episode}'];asset=g.attrs['asset_name'];T=len(g['obs/eef_pose']);end=int(.6*T)
        if len(g[f'states/deformable_object/{asset}/nodal_pos_w'])!=T:raise ValueError('Object/command length mismatch')
        raw=dict(object_prefix=g[f'states/deformable_object/{asset}/nodal_pos_w'][:end],eef_pose=g['obs/eef_pose'][:],robot_root=g['initial_state/articulation/robot/root_pose'][0],width=g['obs/gripper_width'][:],command=g['obs/gripper_binary'][:])
    return softvt_payload(raw,subset,task,episode,str(path),end),softvt_payload(raw,subset,task,episode,str(path),max(3,int(end*.7)))

def save_payload(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if 'evaluation' in data:raise ValueError('Evaluator forbidden')
    np.savez_compressed(path,**{k:v for k,v in data.items() if isinstance(v,np.ndarray)})
    path.with_suffix('.json').write_text(json.dumps({k:v for k,v in data.items() if not isinstance(v,np.ndarray)},indent=2)+'\n')

def load_payload(path):
    path=Path(path);data=json.loads(path.with_suffix('.json').read_text())
    with np.load(path,allow_pickle=False) as f:data.update({k:f[k].copy() for k in f.files})
    if 'evaluation' in data:raise ValueError('Evaluator forbidden')
    return data


def save_scene(path,data):
    """Exporter-only raw initial-pose access; persist world geometry for forecast."""
    from scripts.contact_benchmark_v12.scene import load_scene
    path=Path(path);scene=load_scene(data);arrays={};meshes=[]
    for i,m in enumerate(scene.meshes):
        arrays[f'vertices_{i}']=m['vertices'];arrays[f'triangles_{i}']=m['triangles']
        meshes.append({k:v for k,v in m.items() if k not in ('vertices','triangles')})
    np.savez_compressed(path,**arrays)
    path.with_suffix('.json').write_text(json.dumps({'floor':scene.floor,'spacing':scene.spacing,'meshes':meshes},indent=2)+'\n')
    return scene

def load_scene_payload(path):
    from scripts.contact_benchmark_v12.scene import StaticScene
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text());meshes=[]
    with np.load(path,allow_pickle=False) as a:
        for i,m in enumerate(meta['meshes']):meshes.append(dict(m,vertices=a[f'vertices_{i}'].copy(),triangles=a[f'triangles_{i}'].copy()))
    return StaticScene(meshes,floor=meta['floor'],spacing=meta['spacing'])
