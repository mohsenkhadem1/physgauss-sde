"""Localized gripper hypothesis calibrated at grasp, using prefix geometry only.

The original adapter squeezes the entire object span to the measured jaw width.
This alternative attaches only neighborhoods of the finger pads and prescribes
changes in width relative to the observed grasp. It does not assume the object
fills the entire jaw gap, and does not use force or future object motion.
"""
import numpy as np,h5py
from scripts.external_benchmark.data import softvt as original_softvt,quat_rotation,rigid_map

def softvt(subset,task,episode,observation_count=48):
    data=original_softvt(subset,task,episode,observation_count)
    with h5py.File(data['source']) as file:
        group=file[f'data/demo_{episode}'];local=group['obs/eef_pose'][:].astype(float);base=group['initial_state/articulation/robot/root_pose'][0].astype(float)
    R0=quat_rotation(base[3:]);rotation=np.einsum('ij,tjk->tik',R0,quat_rotation(local[:,3:]))
    nodes=data['nodes'];grasp=data['grasp'];control=data['controls'][:,0];width=data['gripper_width']
    R,b=rigid_map(data['observed_initial'],data['observed'][grasp]);grasp_geometry=nodes@R+b
    reference=(grasp_geometry-control[grasp])@rotation[grasp]
    weights=[];targets=[]
    for sign in [-1,1]:
        pad=np.array([0.,sign*width[grasp]/2,0.])
        distances=(reference-pad)/np.array([.010,.008,.012])
        weights.append(np.exp(-.5*(distances**2).sum(1)))
        target=np.broadcast_to(reference,(len(width),len(nodes),3)).copy()
        target[:,:,1]+=sign*.5*(width-width[grasp])[:,None]
        targets.append(np.einsum('tnj,tij->tni',target,rotation)+control[:,None])
    data['attachment_weights']=np.asarray(weights);data['controller_targets']=np.asarray(targets).transpose(1,0,2,3)
    data['grip_hypothesis']='Localized 10/8/12 mm finger-pad kernels; width changes relative to prefix-calibrated grasp'
    return data
