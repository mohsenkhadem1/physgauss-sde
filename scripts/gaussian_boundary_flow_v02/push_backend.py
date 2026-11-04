"""Isolated prefix-derived tool-plane hypothesis for the exposed push pilot.

Command direction is NOT a measured surface normal. A fixed plane normal is
estimated from selected prefix controller centroids under a declared straight
approach assumption. Standoff is an explicit fixed hypothesis (not inferred).
No object suffix, force, contact label or neural training is used.
"""
import numpy as np
from scripts.gaussian_boundary_flow_v02.object_backend_geometry import GeometryAwareGaussianBoundaryBackend
from scripts.gaussian_boundary_flow_v02.unilateral_attachment import UnilateralAttachmentFEM

NORMAL_POLICY=dict(prefix_frames=12,observation_points=48,min_command_span_m=.005,
                   min_first_component_fraction=.8,min_net_alignment=.8,
                   min_initial_inward_cosine=.1,
                   normal='principal controller-centroid motion axis, signed by prefix net motion',
                   plane_origin='current prescribed controller centroid + declared standoff * fixed normal',
                   standoff_hypotheses_m=[-.01,0.,.01],
                   limitation='Straight approach direction is a proxy, not an observed tool surface normal')


def estimate_plane(data,prefix_frames,*,standoff_m=0.):
    if any(k in data for k in ('evaluation','manual','forces','material','object_points')):
        raise ValueError('Evaluator/privileged fields forbidden')
    end=int(data['train_end']);frames=np.asarray(prefix_frames)
    if len(data['observed'])!=end or len(data['observed_mask'])!=end:
        raise ValueError('Only prefix object observations permitted')
    if frames.ndim!=1 or frames.dtype.kind not in 'iu' or len(frames)<3 or frames[0]!=0 or frames[-1]!=end-1 or np.any(np.diff(frames)<=0):
        raise ValueError('Ordered prefix-only frame budget required')
    if not np.isfinite(standoff_m):raise ValueError('Finite declared standoff required')
    controls=np.asarray(data['controls'])
    if controls.ndim!=3 or controls.shape[-1]!=3 or not np.isfinite(controls).all():
        raise ValueError('Finite prescribed controller point schedule [T,P,3] required')
    centers=controls.mean(1);selected=centers[frames]
    _,singular,V=np.linalg.svd(selected-selected.mean(0),full_matrices=False)
    net=selected[-1]-selected[0];span=float(np.linalg.norm(net))
    if span<NORMAL_POLICY['min_command_span_m'] or np.sum(singular**2)==0:
        raise ValueError('Command direction unidentifiable: insufficient net movement')
    axis=V[0];axis*=1 if axis@net>=0 else -1
    fraction=float(singular[0]**2/np.sum(singular**2));alignment=float(axis@net/span)
    inward=np.asarray(data['nodes']).mean(0)-selected[0]
    inward_norm=float(np.linalg.norm(inward))
    cosine=float(axis@inward/inward_norm) if inward_norm>0 else -1.
    if fraction<NORMAL_POLICY['min_first_component_fraction'] or alignment<NORMAL_POLICY['min_net_alignment']:
        raise ValueError(f'Command direction unidentifiable: curved/reversing approach, fraction={fraction:g}, alignment={alignment:g}')
    if cosine<NORMAL_POLICY['min_initial_inward_cosine']:
        raise ValueError(f'Approach-plane assumption unsupported: initial inward cosine={cosine:g}')
    origins=centers[:,None]+float(standoff_m)*axis[None,None]
    normals=np.broadcast_to(axis,(len(centers),1,3)).copy()
    return origins,normals,dict(policy=NORMAL_POLICY.copy(),frames=frames.tolist(),normal=axis.tolist(),
                               net_span_m=span,first_component_fraction=fraction,net_alignment=alignment,
                               initial_inward_cosine=cosine,standoff_m=float(standoff_m),
                               interpretation='Identified command axis only; surface-normal and standoff assumptions remain unverified')


class PrefixPlanePushBackend(GeometryAwareGaussianBoundaryBackend):
    def __init__(self,data,device='cpu',*,prefix_frames,standoff_m=0.):
        origins,normals,info=estimate_plane(data,prefix_frames,standoff_m=standoff_m)
        super().__init__(data,device)
        if self.groups!=1:raise ValueError('Isolated single-controller pushing hypothesis only')
        self.model=UnilateralAttachmentFEM(data,device=device,field=False,contact=True,
                         nonconvex=True,materials=1,plane_origins=origins,plane_normals=normals,
                         unilateral_groups=[True])
        self.base_weights=self.model.weights.clone();self.plane_hypothesis=info
        # Parent retains identical prior/parameter mapping, table, uniform bulk,
        # geometry eligibility checks and full-command continuation schedule.
