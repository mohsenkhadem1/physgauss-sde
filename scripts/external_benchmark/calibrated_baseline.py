"""Strong rigid baseline conditioned on the same final observed prefix state."""
import numpy as np
from scripts.external_benchmark.data import rigid_map

def calibrated_rigid(data):
    initial=data.get('support_geometry',data['evaluation']['points'][0]);T=len(data['times']);end=data['train_end'];frame=end-1
    active=data['attachment_active'][frame];weights=active if active.sum()>0 else np.ones_like(active)
    desired=np.average(data['controller_targets'][frame],axis=0,weights=weights)
    control_R,control_b=rigid_map(data['nodes'],desired)
    valid=data['observed_mask'][frame];local=(data['observed'][frame,valid]-control_b)@control_R.T
    offset_R,offset_b=rigid_map(data['observed_initial'][valid],local)
    local_shape=initial@offset_R+offset_b;pred=[]
    for t in range(T):
        active=data['attachment_active'][t]
        if active.sum()>0:
            desired=np.average(data['controller_targets'][t],axis=0,weights=active)
            R,b=rigid_map(data['nodes'],desired);points=local_shape@R+b
        elif data['dataset']=='SoftVTBench' and t>data['grasp'] and len(pred)>1:
            dt=float(data['times'][t]-data['times'][t-1]);velocity=(pred[-1].mean(0)-pred[-2].mean(0))/dt
            points=pred[-1]+velocity*dt+[0.,0.,-9.81*dt*dt];points[:,2]+=max(0.,data['floor']-points[:,2].min())
        else:points=initial.copy()
        pred.append(points)
    return np.asarray(pred)
