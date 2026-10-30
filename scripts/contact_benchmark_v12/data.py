"""v12 input preparation; all calibration respects the actual observation cut."""
import numpy as np
import h5py
from scripts.external_benchmark.data import softvt, phystwin, rigid_map, quat_rotation, predictor_payload

def trim_prefix(data,end):
    out=dict(data);end=int(end)
    if not 2<=end<=data['train_end']:raise ValueError('Invalid prefix end')
    out['observed']=data['observed'][:end].copy();out['observed_mask']=data['observed_mask'][:end].copy();out['train_end']=end
    if data['dataset']!='SoftVTBench':return out
    # Recompute the grasp from permitted controls/observations; NEVER reuse a
    # grasp calibrated on later validation frames.
    T=len(data['times']);nodes=data['nodes'];control=data['controls'][:,0];width=data['gripper_width'];command=data['gripper_command']
    distance=np.linalg.norm(control-nodes.mean(0),axis=1)
    contact=np.flatnonzero((width<.065)&(distance<.10)&(np.arange(T)<end))
    if not len(contact):raise ValueError('No input-defined grasp in shorter prefix')
    grasp=int(contact[np.argmin(distance[contact])])
    episode=int(data['name'].rsplit('demo',1)[1])
    with h5py.File(data['source']) as f:
        group=f[f'data/demo_{episode}'];local=group['obs/eef_pose'][:].astype(float);base=group['initial_state/articulation/robot/root_pose'][0].astype(float)
    rotations=np.einsum('ij,tjk->tik',quat_rotation(base[3:]),quat_rotation(local[:,3:]))
    R,b=rigid_map(data['observed_initial'],out['observed'][grasp]);geometry=nodes@R+b
    local=(geometry-control[grasp])@rotations[grasp];transverse=local[:,1]
    mid=(transverse.max()+transverse.min())/2;span=float(np.ptp(transverse));targets=[];weights=[]
    for sign in [-1,1]:
        weights.append(np.exp(-.5*((transverse-(mid+sign*span/2))/max(.004,span*.12))**2))
        target=np.broadcast_to(local,(T,len(nodes),3)).copy();target[:,:,1]=mid+(transverse-mid)[None]*np.minimum(width/span,1.)[:,None]
        targets.append(np.einsum('tnj,tij->tni',target,rotations)+control[:,None])
    active=((np.arange(T)>=grasp)&(width<span*1.15)&(command<0)).astype(float)
    out.update(grasp=grasp,attachment_weights=np.asarray(weights),controller_targets=np.asarray(targets).transpose(1,0,2,3),attachment_active=np.repeat(active[:,None],2,axis=1))
    return out

def load(spec,validation=False):
    data=phystwin(spec['case'],300) if 'case' in spec else softvt(spec['subset'],spec['task'],spec['episode'])
    return trim_prefix(data,max(3,int(data['train_end']*.7))) if validation else data
