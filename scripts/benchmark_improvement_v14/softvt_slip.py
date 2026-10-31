"""Prefix-relative slip extrapolation, a reduced attachment model, not contact inference."""
import numpy as np
from scripts.contact_benchmark_v12.object_frame import control_frames
from scripts.contact_benchmark_v12.contact_rollout import rollout


def predict(data, base_prediction=None, config=None):
    config={} if config is None else config
    if 'evaluation' in data:raise ValueError('Evaluator dictionary forbidden')
    if data['dataset']!='SoftVTBench':raise ValueError('SoftVT only')
    base=np.asarray(base_prediction,float);end=int(data['train_end']);last=end-1
    if len(data['observed'])!=end or base.shape!=(len(data['times']),len(data['nodes']),3):raise ValueError('Invalid prefix/shape')
    scene=config['scene'];window=int(config.get('window',8))
    if window!=8:raise ValueError('Development law fixes8frame window')
    frames=control_frames(data);active=np.asarray(data['attachment_active']).sum(1)>0
    ids=np.arange(max(int(data['grasp']),end-window),end)
    ids=ids[active[ids]]
    if len(ids)<4 or not active[last]:return base.copy(),dict(fallback='Insufficient active prefix',trigger=None)
    mask=np.asarray(data['observed_mask'][ids],bool).all(0)&np.isfinite(data['observed'][ids]).all(axis=(0,2))
    if mask.sum()<3:return base.copy(),dict(fallback='Insufficient shared markers',trigger=None)
    local=np.array([(data['observed'][t,mask].mean(0)-frames[t][1])@frames[t][0].T for t in ids])
    times=np.asarray(data['times']);X=np.c_[np.ones(len(ids)),times[ids]-times[last]]
    coef=np.linalg.lstsq(X,local,rcond=None)[0];velocity=coef[1]
    # Only motion along gravity can cause this reduced slip-release transition.
    gravity=np.array([0.,0.,-1.])@frames[last][0].T
    speed=max(0.,float(velocity@gravity));slip=gravity*speed
    extent=float(np.ptp(np.asarray(data['nodes'])@gravity));threshold=.5*extent
    out=base.copy();trigger=None
    # Preserve marker-to-node centroid offset in the fitted terminal geometry.
    predicted_marker=data['observed_initial']+data['observation_matrix']@(base[last]-data['nodes'])
    centroid_offset=(base[last].mean(0)-predicted_marker[mask].mean(0))@frames[last][0].T
    terminal=local[-1]+centroid_offset
    for t in range(end,len(base)):
        if not active[t]:break
        elapsed=times[t]-times[last]
        if speed*elapsed>=threshold:
            trigger=t;break
        R,b=frames[t];center=(terminal+slip*elapsed)@R+b
        out[t]+=center-out[t].mean(0)
    modified=dict(data)
    if trigger is not None:
        modified['gripper_command']=np.array(data['gripper_command'],copy=True)
        modified['gripper_command'][trigger:]=1.
    out,contact=rollout(modified,out,scene)
    return out,dict(method='prefix_gravity_slip_overlap',fit_frames=ids.tolist(),shared_markers=int(mask.sum()),local_velocity_m_s=velocity.tolist(),slip_velocity_m_s=slip.tolist(),speed_m_s=speed,half_extent_m=threshold,trigger=trigger,fit_end=end,contact=contact,scope='Fixed8frame position fit; half initial object extent overlap budget from final prefix. Unknown finger geometry approximated; no measured forces or true attachment labels.')
