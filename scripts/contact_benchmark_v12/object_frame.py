"""Prefix-only object-frame adapters, not a new coupled-contact solver.

API: adapt(data, base_prediction, method='pose', fit_end=None, strength=1.).
`data` must be a predictor payload (evaluation is ignored); `base_prediction`
contains simulated mesh-node trajectories. Outputs retain its dimensions.
`select_adapter` uses a chronological subprefix to select one fixed candidate,
then recalibrates on the complete permitted prefix. Independent validation
requires separately fitted validation_prediction and shortened validation_data.
The default without them is an explicitly conditional diagnostic only.

Row-vector convention: world = material @ R + b. Pose calibration changes no
shape. Residual calibration is a distinct, regularized empirical correction;
it is not interpreted as material identification or a contact force.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scripts.external_benchmark.data import rigid_map


def decompose(reference, points):
    """Proper rigid Kabsch pose and residual with no best-fit rigid component."""
    reference=np.asarray(reference,float);points=np.asarray(points,float)
    R,b=rigid_map(reference,points)
    local=(points-b)@R.T
    return R,b,local-reference


def project_deformation(reference, displacement):
    """Remove finite rigid motion, rather than a small-angle approximation."""
    return decompose(reference,reference+displacement)[2]


def control_frames(data):
    """Rigid part of commanded attachment geometry; no observed object state."""
    nodes=np.asarray(data['nodes']);targets=data['controller_targets']
    frames=[]
    for t in range(len(targets)):
        active=np.asarray(data['attachment_active'][t],float)
        # This adapter applies corrections only while attachments are active.
        # Inactive frames retain the base simulator's coupled release dynamics.
        if active.sum()<=0:
            frames.append(frames[-1] if frames else (np.eye(3),np.zeros(3)));continue
        desired=np.average(targets[t],axis=0,weights=active)
        frames.append(rigid_map(nodes,desired))
    return frames


def _rotation_blend(R,amount):
    return Rotation.from_rotvec(Rotation.from_matrix(R).as_rotvec()*amount).as_matrix()


def _observed_prediction(data,pred):
    return data['observed_initial'][None]+np.einsum('pn,tnc->tpc',data['observation_matrix'],pred-data['nodes'][None])


def _calibration(data,base,frames,end):
    # Last valid active prefix frame supplies a current object-to-controller
    # correction. No late targets and no best-case ground truth selection.
    candidates=np.flatnonzero(np.asarray(data['attachment_active'][:end]).sum(1)>0)
    for k in candidates[::-1]:
        valid=np.asarray(data['observed_mask'][k],bool).copy()
        valid &= np.isfinite(data['observed'][k]).all(1)
        if valid.sum()<3:continue
        R,b=frames[k]
        obs_pred=_observed_prediction(data,base[k:k+1])[0]
        source=(obs_pred[valid]-b)@R.T
        target=(data['observed'][k,valid]-b)@R.T
        correction_R,correction_b=rigid_map(source,target)
        return int(k),correction_R,correction_b
    return None,np.eye(3),np.zeros(3)


def adapt(data,base_prediction,method='pose',fit_end=None,strength=1.,ridge=1.):
    """Return corrected nodes plus metadata; never reads `evaluation`.

    pose: transport a last-prefix rigid FEM-to-observation discrepancy in the
      moving controller frame, preserving each frame's FEM shape exactly.
    residual: pose correction plus bounded ridge prediction of nonrigid local
      discrepancy from controller positions/orientation. Its final correction
      is projected to contain no rigid component.
    none: unchanged base, included as an explicit regression safeguard.
    """
    base=np.asarray(base_prediction,float);out=base.copy();nodes=data['nodes']
    end=int(data['train_end'] if fit_end is None else fit_end)
    if not 1<=end<=min(len(data['observed']),len(base)):raise ValueError('fit_end exceeds observed prefix')
    if method not in ('none','pose','residual'):raise ValueError(method)
    active=np.asarray(data['attachment_active'][:len(base)]).sum(1)>0
    finite=np.isfinite(base).all(axis=(1,2))
    if np.any(active & ~finite):raise ValueError('Active base frames must be finite; failures cannot be calibrated away')
    metadata=dict(method=method,strength=float(strength),fit_end=end,ridge=float(ridge),input='prefix positions and known controls only',
                  inherited_failed_frames=np.flatnonzero(~finite).tolist(),
                  inherited_failure_policy='inactive failures retained explicitly; contact rollout may replace them')
    if method=='none' or strength==0:return out,metadata
    frames=control_frames(data);k,Q,c=_calibration(data,base,frames,end)
    metadata['calibration_frame']=k
    if k is None:return out,metadata
    Q=_rotation_blend(Q,strength);c=c*strength
    active=np.asarray(data['attachment_active']).sum(1)>0
    for t,(R,b) in enumerate(frames[:len(base)]):
        if active[t]:out[t]=((base[t]-b)@R.T@Q+c)@R+b
    # Initial geometry determines official nearest-marker correspondence.
    out[0]=base[0]
    if method=='pose':return out,metadata
    # Use a compact input basis: controller-centroid translation in the moving
    # frame and rotated gravity direction. No time polynomial/extrapolated force.
    features=[]
    for t,(R,b) in enumerate(frames[:len(base)]):
        local=(data['controller_targets'][t]-b)@R.T-nodes[None]
        features.append(np.r_[local.mean(1).ravel(),np.array([0.,0.,-1.])@R.T])
    F=np.asarray(features);mu=F[:end].mean(0);sd=F[:end].std(0);sd[sd<1e-5]=1
    Z=np.c_[np.ones(len(F)),(F-mu)/sd]
    # Fit local observed discrepancy after the rigid pose correction. Sparse
    # tracks are interpolated onto nodes; no evaluation geometry is consulted.
    observed_pred=_observed_prediction(data,out[:end]);res=np.empty_like(observed_pred)
    for t,(R,b) in enumerate(frames[:end]):res[t]=(data['observed'][t]-observed_pred[t])@R.T
    learned=np.zeros((len(base),res.shape[1],3));cap=np.zeros(res.shape[1])
    for j in range(res.shape[1]):
        valid=np.asarray(data['observed_mask'][:end,j],bool)&active[:end]&np.isfinite(res[:,j]).all(1)
        if valid.sum()<4:continue
        X=Z[:end][valid];y=res[valid,j]
        coef=np.linalg.solve(X.T@X+np.eye(Z.shape[1])*ridge,X.T@y)
        learned[:,j]=Z@coef
        cap[j]=max(np.linalg.norm(y,axis=1).max()*2,1e-6)
    length=np.linalg.norm(learned,axis=-1)
    learned*=np.minimum(1.,cap[None]/np.maximum(length,1e-12))[...,None]
    distance,index=cKDTree(data['observed_initial']).query(nodes,k=min(6,len(data['observed_initial'])))
    if distance.ndim==1:distance=distance[:,None];index=index[:,None]
    weights=1/np.maximum(distance,1e-8)**2;weights/=weights.sum(1,keepdims=True)
    nodal=np.einsum('nk,tnkc->tnc',weights,learned[:,index])
    for t,(R,b) in enumerate(frames[:len(base)]):
        if active[t] and t>0:
            local=(out[t]-b)@R.T
            correction=project_deformation(local,nodal[t]*strength)
            out[t]+=correction@R
    metadata['residual_cap_metres_max']=float(cap.max())
    return out,metadata


def select_adapter(data,base_prediction,fit_end=None,validation_fraction=.3,
                   validation_prediction=None,validation_data=None):
    """Chronological selector, requiring 2% gain over the matched base.

    Final experiments supply BOTH validation inputs: a separately subprefix-fit
    FEM prediction and a recalibrated payload whose observed arrays end exactly
    at its train_end. Candidates see only that payload; full-payload observations
    in [cut,end) are used solely to score predictions. The selected adapter is
    then refit on the full permitted prefix/base. `pose` is explicitly the
    matched pose-reset baseline (same FEM and observations, no shape residual).

    Omitting both validation inputs retains conditional diagnostic mode, where
    the supplied FEM may already have seen validation observations. That mode
    must not be reported as independent validation of the FEM.
    """
    end=int(data['train_end'] if fit_end is None else fit_end)
    if (validation_prediction is None)!=(validation_data is None):
        raise ValueError('Both independent validation inputs are required together')
    independent=validation_data is not None
    if independent:
        cut=int(validation_data['train_end'])
        if not 1<=cut<end:raise ValueError('Validation prefix must be shorter than full prefix')
        if len(validation_data['observed'])!=cut or len(validation_data['observed_mask'])!=cut:
            raise ValueError('Validation candidate payload must contain only subprefix observations')
        if len(validation_prediction)<end:raise ValueError('Validation prediction does not cover validation interval')
        if validation_data['observed'].shape[1:]!=data['observed'].shape[1:]:
            raise ValueError('Validation and full observation identities must match')
        candidate_data=validation_data;candidate_base=validation_prediction
    else:
        cut=max(3,int(end*(1-validation_fraction)))
        candidate_data=data;candidate_base=base_prediction
    if cut>=end:return adapt(data,base_prediction,'none',end)
    candidates=[('none',0.)]+[(m,a) for m in ('pose','residual') for a in (.25,.5,1.)]
    rows=[]
    for method,strength in candidates:
        prediction,meta=adapt(candidate_data,candidate_base,method,cut,strength)
        observed=_observed_prediction(candidate_data,prediction[cut:end])
        valid=np.asarray(data['observed_mask'][cut:end],bool)&np.isfinite(data['observed'][cut:end]).all(2)
        err=observed-data['observed'][cut:end]
        failed=np.flatnonzero(~np.isfinite(prediction[cut:end]).all(axis=(1,2)))+cut
        value=float(np.sqrt(np.mean(err[valid]**2))) if valid.any() and not len(failed) else float('inf')
        rows.append(dict(method=method,strength=strength,validation_rmse_m=value,failed_validation_frames=failed.tolist()))
    best=min(rows,key=lambda r:r['validation_rmse_m'])
    if not best['validation_rmse_m']<rows[0]['validation_rmse_m']*.98:best=rows[0]
    out,meta=adapt(data,base_prediction,best['method'],end,best['strength'])
    mode='independent subprefix FEM and adapter' if independent else 'conditional diagnostic, full-prefix FEM may see validation'
    meta.update(selection=mode+'; >=2% improvement',independent_validation=independent,
                validation_cut=cut,validation=rows,
                pose_baseline='same FEM with prefix-calibrated rigid pose reset; no deformation residual')
    return out,meta


def estimate_attachment_state(data,fit_end=None,window=8):
    """Conservative position-only attachment diagnostic for rollout initialization.

    A closed command is rejected only when an informative moving tool leaves
    the observed object nearly stationary. Ambiguous motion remains unknown;
    no-contact force, true simulator attachment, or future target is consulted.
    Thresholds are geometric: >=3 mm commanded displacement and <25% object
    response, with >=75% relative discrepancy. This is evidence of detachment,
    not an estimate of friction or a general probabilistic contact classifier.
    """
    end=int(data['train_end'] if fit_end is None else fit_end)
    if not 1<=end<=len(data['observed']):raise ValueError('fit_end exceeds observed prefix')
    last=end-1;first=max(0,end-window)
    commanded=bool(np.asarray(data['attachment_active'][last]).sum()>0)
    answer=dict(state='unknown' if commanded else 'released',attachment_confidence=None if commanded else 0.,
                commanded_active=commanded,window=[first,last],source='observed prefix/tool kinematics only')
    if not commanded:return answer
    valid=np.asarray(data['observed_mask'][first],bool)&np.asarray(data['observed_mask'][last],bool)
    valid &= np.isfinite(data['observed'][first]).all(1)&np.isfinite(data['observed'][last]).all(1)
    if valid.sum()<3:return answer
    frames=control_frames(data);R0,b0=frames[first];R1,b1=frames[last]
    start=data['observed'][first,valid];finish=data['observed'][last,valid]
    held=(start-b0)@R0.T@R1+b1
    tool_motion=float(np.sqrt(np.mean((held-start)**2)))
    object_motion=float(np.sqrt(np.mean((finish-start)**2)))
    discrepancy=float(np.sqrt(np.mean((finish-held)**2)))
    answer.update(tool_motion_rms_m=tool_motion,object_motion_rms_m=object_motion,relative_discrepancy_rms_m=discrepancy)
    if tool_motion<.003:return answer
    if object_motion<.25*tool_motion and discrepancy>.75*tool_motion:
        answer.update(state='detached_stationary',attachment_confidence=0.)
    elif discrepancy<max(.001,.2*tool_motion):
        answer.update(state='kinematically_attached',attachment_confidence=1.)
    return answer
