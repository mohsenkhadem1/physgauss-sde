"""Packed supplied-support/supplied-material queries of frozen modal mechanics.

Each category is an explicit global point-pin node ID with its independently
constructed enriched basis. Categories with equal mode count share one batched
recurrence; different mode counts remain separate, without padding or dropping.
No observations, truth labels, probabilities, fitting, or finite steps enter.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import torch

try:
    from campaign_modal_material_forward import BatchedModalMaterialForward
    from campaign_hypothesis_dynamics import validate_clock
    from campaign_fixed_actuator_clock import FixedActuatorClock
except ImportError:
    from experiments.campaign_modal_material_forward import BatchedModalMaterialForward
    from experiments.campaign_hypothesis_dynamics import validate_clock
    from experiments.campaign_fixed_actuator_clock import FixedActuatorClock


class PackedBoundaryMaterialForward:
    """A fixed catalogue of known point-support geometries, not a prior/posterior.

    ``geometry_models`` maps global fixture-node integers to enriched modal
    objects. It must contain the caller's complete intended catalogue. This
    adapter creates no weights and never chooses or removes a category.
    Its approximation and small-strain pose semantics are exactly those of
    BatchedModalMaterialForward; it is not finite-SO(3) dynamics.
    """
    def __init__(self, geometry_models):
        if not isinstance(geometry_models,dict) or not geometry_models:
            raise ValueError('Nonempty explicit geometry ID -> enriched modal mapping required')
        for key,modal in geometry_models.items():
            if isinstance(key,(bool,np.bool_)) or not isinstance(key,(int,np.integer)) or key<0:
                raise ValueError('Categorical geometry IDs must be global nonnegative node integers')
            p=modal.full.model
            if p.boundary_kind!='point_pin' or p.fixture_nodes.tolist()!=[int(key)]:
                raise ValueError('Each geometry ID must exactly equal its sole physical point-pin node')
            if len(p.handle_dofs)!=2:
                raise ValueError('This adapter requires the same two complete known handles')
            if not modal.basis_diagnostics.get('parameter_independent_completion'):
                raise ValueError('Explicit enriched static-lift completion required')
        self.models={int(k):geometry_models[k] for k in sorted(geometry_models)}
        self.geometry_ids=tuple(self.models)
        self._tensor_records=None
        first=next(iter(self.models.values()))
        self.node_count=len(first.full.points);self.handle_count=2
        for modal in self.models.values():self._same_public_hardware(first,modal)
        groups=defaultdict(list)
        for key,modal in self.models.items():groups[modal.mode_count].append(key)
        self.mode_groups={k:tuple(v) for k,v in sorted(groups.items())}
        self._single={k:BatchedModalMaterialForward(modal) for k,modal in self.models.items()}
        self._cache={}

    @classmethod
    def from_tensor_records(cls,records,public_hardware):
        """Load only geometry matrices, not duplicated full systems/live clocks.

        File/source hashes are checked by the cache loader. Every record must
        additionally identify the exact shared complete public hardware.
        """
        try:
            from campaign_hypothesis_inputs import known_hardware_from_public,physical_hardware_digest
        except ImportError:
            from experiments.campaign_hypothesis_inputs import known_hardware_from_public,physical_hardware_digest
        known=known_hardware_from_public(public_hardware);fingerprint=physical_hardware_digest(known)
        if len(known.handles)!=2 or not records:raise ValueError('Nonempty two-handle geometry catalogue required')
        result=cls.__new__(cls);result.models={};result._single={};result._cache={}
        result.geometry_ids=tuple(sorted(records));result.node_count=len(known.points);result.handle_count=2
        groups=defaultdict(list);result._tensor_records={}
        for id,raw in records.items():
            k=raw['mode_count'];n=result.node_count
            if (type(id) is not int or id<0 or id>=n or raw['node_id']!=id or type(k) is not int or k<=0
                    or raw['physical_hardware_sha256']!=fingerprint
                    or raw['fixture_nodes']!=[id] or raw['support_law']!='static_point_pin'
                    or not raw['basis_diagnostics'].get('parameter_independent_completion')):
                raise ValueError('Cached geometry identity/hardware/completion mismatch')
            shapes={**{name:(k,k) for name in ['Mb','Mh','Ks','Kv']},'AV':(3*n,k),'S':(3*n,3),'G':(k,6),
                'M_lift':(k,3),'Ks_lift':(k,3),'Kv_lift':(k,3),
                'center_map':(6,k),'rotation_map':(6,k),'origins':(2,3)}
            for name,shape in shapes.items():
                a=raw[name]
                if not isinstance(a,torch.Tensor) or a.dtype!=torch.float64 or a.device.type!='cpu' or tuple(a.shape)!=shape or not bool(torch.isfinite(a).all()):
                    raise ValueError('Invalid cached float64 geometry tensor '+name)
            result._tensor_records[id]=raw;groups[k].append(id)
        result.mode_groups={k:tuple(sorted(v)) for k,v in sorted(groups.items())}
        result._rest=torch.tensor(known.points.copy(),dtype=torch.float64)
        return result

    @staticmethod
    def _same_public_hardware(first,other):
        a,b=first.full.model,other.full.model
        for name in ['points','com_offsets','auxiliary_offsets','auxiliary_mass']:
            if not torch.equal(getattr(a,name),getattr(b,name)):
                raise ValueError('All categories must share exact public rest geometry and full hardware')
        if (not np.array_equal(a.tets,b.tets) or a.origins!=b.origins
                or len(a.patches)!=len(b.patches)
                or any(not torch.equal(x,y) for x,y in zip(a.patches,b.patches))):
            raise ValueError('All categories must share exact tets/origins/ordered rigid patches')

    def catalogue(self):
        """No empirical probabilities or geometry selection are manufactured."""
        return dict(geometry_ids=list(self.geometry_ids),
            groups=[dict(mode_count=k,geometry_ids=list(ids)) for k,ids in self.mode_groups.items()],
            categories_dropped=[],weights_supplied_or_estimated=False,
            role='explicit_known_geometry_catalogue_not_prior_or_posterior')

    def _prepare(self,clock,device):
        if type(clock) is not FixedActuatorClock:
            raise TypeError('Explicit fixed known-actuator clock required')
        key=(clock.public_manifest()['manifest_sha256'],str(device))
        if key in self._cache:return self._cache[key]
        refined=validate_clock(clock,120,self.node_count,self.handle_count)
        result={}
        # Geometry preparation is once on CPU. Device queries thereafter gather
        # matrices for all particles; no serial per-geometry GPU solve occurs.
        for mode_count,ids in self.mode_groups.items():
            if self._tensor_records is None:
                prepared=[self._single[id]._prepare(clock,'cpu') for id in ids]
                G=[self.models[id].G.detach() for id in ids]
            else:
                # Static point pins have EXACT zero prescribed displacement
                # and velocity. Retain all lift coupling matrices in the raw
                # cache; their time terms are analytically zero here.
                prepared=[{**self._tensor_records[id],'rest':self._rest,
                    'lifts':torch.zeros((361,3),dtype=torch.float64),
                    **{name:torch.zeros((360,mode_count),dtype=torch.float64) for name in
                       ['lift_inertia','lift_drag','lift_shear','lift_volume']},
                    'output_indices':set(refined.output_step_indices.tolist())} for id in ids]
                G=[b['G'] for b in prepared]
            buffers={name:torch.stack([b[name] for b in prepared]).detach().to(device)
                for name in ['Mb','Mh','Ks','Kv','AV','S','lifts','center_map','rotation_map','origins']}
            buffers['G']=torch.stack(G).to(device)
            # Time-dependent forcing from the prescribed support law is kept
            # geometry-specific. Only one small tensor is gathered per step.
            buffers['lift_terms']=torch.stack([torch.stack([b[name] for name in
                ['lift_inertia','lift_drag','lift_shear','lift_volume']],dim=1) for b in prepared]).to(device)
            buffers['rest']=prepared[0]['rest'].detach().to(device)
            buffers['forces']=torch.tensor(refined.handle_forces_world_N.copy(),device=device).flatten(1)
            nodal=refined.nodal_forces_world_N
            buffers['nodal']=torch.tensor(nodal.copy(),device=device).flatten(1)
            buffers['output_indices']=prepared[0]['output_indices']
            buffers['geometry_ids']=ids
            result[mode_count]=buffers
        self._cache[key]=result
        return result

    def _solve_group(self,b,local_ids,theta,mode_count):
        """One batched solve per distinct physical mode count, no padded DOFs."""
        E,rho,nu,drag=theta.unbind(1);mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu));h=1/120
        selected={name:b[name].index_select(0,local_ids) for name in
            ['Mb','Mh','Ks','Kv','AV','S','G','center_map','rotation_map','origins']}
        M=rho[:,None,None]*selected['Mb']+selected['Mh']
        C=(rho*drag)[:,None,None]*selected['Mb']
        K=mu[:,None,None]*selected['Ks']+lam[:,None,None]*selected['Kv']
        effective=M/h**2+C/h+K
        factor=torch.linalg.cholesky((effective+effective.transpose(-1,-2))/2)
        z=theta.new_zeros((len(theta),mode_count));v=torch.zeros_like(z)
        positions=[];centers=[];rotations=[]
        mv=lambda matrix,vector:torch.bmm(matrix,vector.unsqueeze(-1)).squeeze(-1)
        def capture(index):
            lift=b['lifts'][:,index].index_select(0,local_ids)
            positions.append((mv(selected['AV'],z)+mv(selected['S'],lift)).reshape(len(theta),self.node_count,3)+b['rest'])
            centers.append(mv(selected['center_map'],z).reshape(len(theta),2,3)+selected['origins'])
            rotations.append(mv(selected['rotation_map'],z).reshape(len(theta),2,3))
        capture(0)
        for i in range(360):
            load=torch.matmul(selected['G'],b['forces'][i])
            if b['nodal'].shape[1]:load=load+torch.matmul(selected['AV'].transpose(-1,-2),b['nodal'][i])
            rhs=load+mv(M,z/h**2+v/h)+mv(C,z/h)
            terms=b['lift_terms'][:,i].index_select(0,local_ids)
            rhs=rhs+rho[:,None]*(terms[:,0]+drag[:,None]*terms[:,1])-mu[:,None]*terms[:,2]-lam[:,None]*terms[:,3]
            new_z=torch.cholesky_solve(rhs.unsqueeze(-1),factor).squeeze(-1)
            v=(new_z-z)/h;z=new_z
            if i+1 in b['output_indices']:capture(i+1)
        return dict(positions_m=torch.stack(positions,1),handle_centers_m=torch.stack(centers,1),
            handle_rotation_coordinates_rad=torch.stack(rotations,1))

    def predict(self,clock,geometry_ids,physical_coordinates,device='cpu'):
        """Inputs: explicit point-pin IDs[B], material E/rho/nu/drag[B,4].

        Returned order equals input order. Material gradients cross grouped
        gather/scatter; categorical IDs intentionally are not differentiable.
        Every supplied category remains available in the unchanged catalogue.
        """
        if (not isinstance(geometry_ids,torch.Tensor) or geometry_ids.dtype!=torch.int64
                or geometry_ids.ndim!=1):
            raise TypeError('Explicit int64 categorical geometry IDs[B] required')
        if (not isinstance(physical_coordinates,torch.Tensor) or physical_coordinates.dtype!=torch.float64
                or physical_coordinates.ndim!=2 or physical_coordinates.shape!=(len(geometry_ids),4)
                or not len(geometry_ids)):
            raise ValueError('Nonempty float64 physical coordinates[B,4] required')
        supplied=geometry_ids.detach().cpu().tolist()
        if any(id not in self.geometry_ids for id in supplied):
            raise ValueError('Unknown geometry category; no nearest-support fallback')
        theta=physical_coordinates.to(device);E,rho,nu,drag=theta.unbind(1)
        if not bool(torch.isfinite(theta).all() and (E>0).all() and (rho>0).all()
                and (nu>-1).all() and (nu<.5).all() and (drag>=0).all()):
            raise ValueError('Valid finite physical material hypotheses required')
        packed=self._prepare(clock,device);outputs={};execution_groups=[]
        for modes,ids in self.mode_groups.items():
            local_lookup={id:j for j,id in enumerate(ids)}
            rows=[i for i,id in enumerate(supplied) if id in local_lookup]
            if not rows:continue
            row_ids=torch.tensor(rows,dtype=torch.int64,device=device)
            local_ids=torch.tensor([local_lookup[supplied[i]] for i in rows],dtype=torch.int64,device=device)
            result=self._solve_group(packed[modes],local_ids,theta.index_select(0,row_ids),modes)
            for name,value in result.items():
                if name not in outputs:outputs[name]=theta.new_zeros((len(theta),*value.shape[1:]))
                outputs[name]=outputs[name].index_copy(0,row_ids,value)
            execution_groups.append(dict(mode_count=modes,particles=len(rows)))
        outputs.update(times_s=torch.tensor(clock.output_times_s.copy(),device=device),
            geometry_ids=geometry_ids.to(device),execution_groups=execution_groups,
            role='supplied_point_support_and_material_linear_modal_not_posterior')
        return outputs


def modal_tensor_record(modal,node_id,physical_hardware_sha256):
    """Minimal static geometry cache; no dense full systems/arbitrary clocks."""
    p=modal.full.model
    if p.boundary_kind!='point_pin' or p.fixture_nodes.tolist()!=[node_id]:raise ValueError('Exact point-pin category required')
    translation=[i for dofs in modal.full.handle_dofs for i in dofs[:3]]
    rotation=[i for dofs in modal.full.handle_dofs for i in dofs[3:]]
    result={name:getattr(modal,name).detach().cpu().clone() for name in
        ['Mb','Mh','Ks','Kv','AV','S','G','M_lift','Ks_lift','Kv_lift','basis']}
    result.update(node_id=node_id,fixture_nodes=[node_id],mode_count=modal.mode_count,
        physical_hardware_sha256=physical_hardware_sha256,support_law='static_point_pin',
        center_map=modal.basis[translation].detach().cpu().clone(),
        rotation_map=modal.basis[rotation].detach().cpu().clone(),
        origins=modal.full.points[p.origins].detach().cpu().clone(),
        basis_diagnostics=modal.basis_diagnostics)
    return result
