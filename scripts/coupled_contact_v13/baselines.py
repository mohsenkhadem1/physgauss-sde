"""Position-only baselines for transfer between interactions on the same object.

Fit sees only an observed calibration trajectory. Predict accepts the new initial
geometry, commanded tool positions/times and optional tool-active masks; no new
object trajectory, force, material or obstacle labels. Node identities and tool
identities are shared between calibration and forecast. These empirical baselines
do not solve collision/contact; a separate rigid-contact baseline is required.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


def _rigid_map(a, b):
    ac=a-a.mean(0);bc=b-b.mean(0)
    u,_,vt=np.linalg.svd(ac.T@bc)
    correction=np.eye(3);correction[-1,-1]=np.sign(np.linalg.det(u@vt))
    r=u@correction@vt
    return r,b.mean(0)-a.mean(0)@r


def _validate(initial, tools, times, active):
    initial=np.asarray(initial,float);tools=np.asarray(tools,float);times=np.asarray(times,float)
    if initial.ndim!=2 or initial.shape[1]!=3 or len(initial)<3:raise ValueError('initial must be [N>=3,3]')
    if tools.ndim!=3 or tools.shape[2]!=3 or len(tools)!=len(times) or len(times)<1:raise ValueError('tools must be [T,K,3], times [T]')
    if not np.isfinite(initial).all() or not np.isfinite(tools).all() or not np.isfinite(times).all():raise ValueError('nonfinite public inputs')
    if np.any(np.diff(times)<=0):raise ValueError('times must strictly increase')
    active=np.ones(tools.shape[:2]) if active is None else np.asarray(active,float)
    if active.shape!=tools.shape[:2] or not np.isfinite(active).all() or np.any((active<0)|(active>1)):raise ValueError('active must be [T,K] in [0,1]')
    return initial,tools,times,active


def _features(initial, tools, times, active, scale):
    # Initial-relative displacements express action, absolute object-relative
    # positions express contact location. Velocities are commanded kinematics.
    delta=(tools-tools[:1])/scale
    relative=(tools-initial.mean(0))/scale
    velocity=np.zeros_like(tools)
    if len(times)>1:velocity[1:]=np.diff(tools,axis=0)/np.diff(times)[:,None,None]/scale
    return np.concatenate([delta.reshape(len(tools),-1),relative.reshape(len(tools),-1),
                           velocity.reshape(len(tools),-1),active],axis=1)


@dataclass
class PositionBaselines:
    scale: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    node_coeff: np.ndarray
    pose_coeff: np.ndarray
    node_count: int
    tool_count: int
    ridge: float
    observed_counts: np.ndarray

    def predict(self, initial, tool_positions, times, active=None):
        initial,tools,times,active=_validate(initial,tool_positions,times,active)
        if len(initial)!=self.node_count or tools.shape[1]!=self.tool_count:raise ValueError('node/tool identities must match calibration')
        features=_features(initial,tools,times,active,self.scale)
        design=np.c_[np.ones(len(times)),(features-self.feature_mean)/self.feature_std]
        displacement=np.einsum('tf,fnc->tnc',design,self.node_coeff)
        # New interaction starts from the supplied geometry, never the last
        # calibration shape or an estimated future target shape.
        displacement-=displacement[:1]
        pose=design@self.pose_coeff;pose-=pose[:1]
        center=initial.mean(0);local=initial-center
        rigid=np.stack([local@Rotation.from_rotvec(p[3:]).as_matrix()+center+p[:3] for p in pose])
        return dict(persistence=np.broadcast_to(initial,(len(times),*initial.shape)).copy(),
                    position_ridge=initial[None]+displacement,position_rigid=rigid)

    def metadata(self):
        return dict(method='position-only ridge and rigid-pose regression',ridge=self.ridge,
                    features='tool displacement from episode start; tool positions relative to initial object center; backward-difference tool velocities; active masks',
                    feature_normalization='calibration mean/std; initial object extent spatial scale',
                    position_features_only=True,force_or_material_labels=False,
                    future_observations=False,contact_solver=False,
                    initial_geometry_policy='subtract predicted frame-zero displacement/pose',
                    node_observation_counts=self.observed_counts.tolist(),
                    limitation='same point/tool identities; rigid pose rotvec regression is local and has branch ambiguity near pi; no collision enforcement')


def fit_baselines(initial, observed, tool_positions, times, observed_mask=None, active=None, ridge=.1):
    initial,tools,times,active=_validate(initial,tool_positions,times,active)
    observed=np.asarray(observed,float)
    if observed.shape!=(len(times),len(initial),3):raise ValueError('observed must be [T,N,3]')
    mask=np.ones(observed.shape[:2],bool) if observed_mask is None else np.asarray(observed_mask,bool)
    if mask.shape!=observed.shape[:2]:raise ValueError('observed_mask must be [T,N]')
    mask=mask & np.isfinite(observed).all(2)
    if ridge<=0 or not np.isfinite(ridge):raise ValueError('ridge must be positive')
    scale=max(float(np.max(np.ptp(initial,axis=0))),1e-6)
    features=_features(initial,tools,times,active,scale)
    mean=features.mean(0);std=features.std(0);std[std<1e-5]=1.
    design=np.c_[np.ones(len(times)),(features-mean)/std]
    penalty=np.eye(design.shape[1])*ridge
    coeff=np.zeros((design.shape[1],len(initial),3))
    for node in range(len(initial)):
        use=mask[:,node]
        if use.any():
            x=design[use];y=observed[use,node]-initial[node]
            coeff[:,node]=np.linalg.solve(x.T@x+penalty,x.T@y)
    poses=[];pose_frames=[]
    center=initial.mean(0)
    for t in range(len(times)):
        use=mask[t]
        if use.sum()<3 or np.linalg.matrix_rank(initial[use]-initial[use].mean(0))<2:continue
        r,b=_rigid_map(initial[use],observed[t,use])
        poses.append(np.r_[center@r+b-center,Rotation.from_matrix(r).as_rotvec()]);pose_frames.append(t)
    pose_coeff=np.zeros((design.shape[1],6))
    if poses:
        x=design[pose_frames];pose_coeff=np.linalg.solve(x.T@x+penalty,x.T@np.asarray(poses))
    return PositionBaselines(scale,mean,std,coeff,pose_coeff,len(initial),tools.shape[1],float(ridge),mask.sum(0))
