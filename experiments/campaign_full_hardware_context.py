"""Explicit complete hardware/query contract for a future forward model.

No model or dataset loader. Public hardware and commanded actions are separate
from material and boundary hypotheses. Nothing here extracts labels from an
episode, observation, recipe, image or response trajectory.

Geometry and vector commands use world axes, with one caller-supplied origin
subtracted from all point coordinates. Inertias/COM offsets use each handle's
body frame; initial rotations map that frame to world axes. Support commands
are exact supplied discrete endpoint transforms, not observed object poses.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

try:
    from campaign_rigid_inertia import auxiliary_mass_points
except ImportError:
    from experiments.campaign_rigid_inertia import auxiliary_mass_points


VERSION = 'complete-hardware-query-context-v0.1'


def array(value,shape,name,dtype=np.float64):
    raw=np.asarray(value)
    if np.issubdtype(np.dtype(dtype),np.integer) and not np.issubdtype(raw.dtype,np.integer):
        raise ValueError(name+' requires integer indices')
    if np.dtype(dtype)==np.dtype(bool) and raw.dtype!=np.dtype(bool):
        raise ValueError(name+' requires an explicit boolean mask')
    result=np.array(value,dtype=dtype,copy=True,order='C')
    if len(result.shape)!=len(shape) or any(n is not None and n!=got for n,got in zip(shape,result.shape)):
        raise ValueError(name+' has an invalid shape')
    if not np.isfinite(result).all():raise ValueError(name+' must be finite')
    result.setflags(write=False)
    return result


def positive(value,name,*,zero=False):
    result=float(value)
    if not np.isfinite(result) or (result<0 if zero else result<=0):
        raise ValueError(name+' must be finite and '+('nonnegative' if zero else 'positive'))
    return result


def rotations(value,shape,name):
    R=array(value,shape,name)
    if (not np.allclose(np.swapaxes(R,-1,-2)@R,np.eye(3),atol=1e-12,rtol=0)
            or not np.allclose(np.linalg.det(R),1.,atol=1e-12,rtol=0)):
        raise ValueError(name+' must contain proper SO(3) rotations')
    return R


def descriptor(value):
    a=np.ascontiguousarray(value)
    return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def records(source_sha256):
    """Copy caller declarations without opening the named source files."""
    result=dict(source_sha256)
    if any(not isinstance(key,str) or not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None for key,value in result.items()):
        raise ValueError('Caller source records require string names and lowercase SHA256 values')
    return result


def provenance(role,features,source_sha256,*,absolute_inputs=None):
    manifest={'schema_version':VERSION,'role':role,
        'arrays':{name:descriptor(value) for name,value in features.items()},
        'caller_declared_source_sha256_not_opened_or_verified':records(source_sha256),
        'interface_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dependency_source_sha256':{'campaign_rigid_inertia.py':hashlib.sha256(Path(auxiliary_mass_points.__code__.co_filename).read_bytes()).hexdigest()}}
    if absolute_inputs is not None:
        manifest['absolute_frame_input_arrays']={name:descriptor(value) for name,value in absolute_inputs.items()}
    manifest['manifest_sha256']=digest(manifest)
    return manifest


@dataclass(frozen=True,kw_only=True)
class KnownHandle:
    origin_world_m:np.ndarray
    ordered_patch_nodes:np.ndarray
    mass_kg:float
    com_inertia_body_kg_m2:np.ndarray
    com_offset_body_m:np.ndarray
    initial_rotation_body_to_world:np.ndarray
    origin_node_id:int|None=None

    def __post_init__(self):
        object.__setattr__(self,'origin_world_m',array(self.origin_world_m,(3,),'handle origin'))
        patch=array(self.ordered_patch_nodes,(None,),'ordered patch',np.int64)
        if not len(patch) or len(set(patch))!=len(patch) or np.min(patch)<0:
            raise ValueError('Ordered patch must contain distinct nonnegative node IDs')
        object.__setattr__(self,'ordered_patch_nodes',patch)
        object.__setattr__(self,'mass_kg',positive(self.mass_kg,'handle mass'))
        I=array(self.com_inertia_body_kg_m2,(3,3),'full COM inertia')
        auxiliary_mass_points(self.mass_kg,I)  # Reuse physical triangle checks; no new realization or repair.
        object.__setattr__(self,'com_inertia_body_kg_m2',I)
        object.__setattr__(self,'com_offset_body_m',array(self.com_offset_body_m,(3,),'body COM offset'))
        object.__setattr__(self,'initial_rotation_body_to_world',rotations(self.initial_rotation_body_to_world,(3,3),'initial handle rotation'))
        if self.origin_node_id is not None:
            if not isinstance(self.origin_node_id,(int,np.integer)) or isinstance(self.origin_node_id,(bool,np.bool_)) or self.origin_node_id<0:
                raise ValueError('Optional origin node must be a nonnegative integer')
            object.__setattr__(self,'origin_node_id',int(self.origin_node_id))


class KnownHardware:
    """Public rest geometry and hardware only; no material/BC argument exists."""
    def __init__(self,*,rest_points_world_m,tetrahedra,surface_faces,handles,
                 coordinate_origin_world_m,source_sha256):
        self.points=array(rest_points_world_m,(None,3),'rest points')
        if len(self.points)<4:raise ValueError('A tetrahedral rest mesh requires at least four points')
        self.tets=array(tetrahedra,(None,4),'tetrahedra',np.int64)
        self.faces=array(surface_faces,(None,3),'complete surface faces',np.int64)
        for indices,name in [(self.tets,'tetrahedra'),(self.faces,'surface faces')]:
            if not len(indices) or indices.min()<0 or indices.max()>=len(self.points):raise ValueError('Invalid '+name)
            if any(len(set(row))!=len(row) for row in indices):raise ValueError('Repeated node in '+name)
            if len(np.unique(np.sort(indices,axis=1),axis=0))!=len(indices):raise ValueError('Duplicate '+name)
        if len(np.unique(self.tets))!=len(self.points):raise ValueError('All rest nodes must belong to the tetrahedral body')
        all_faces=np.concatenate([self.tets[:,ids] for ids in [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]])
        unique,counts=np.unique(np.sort(all_faces,axis=1),axis=0,return_counts=True)
        if np.any(counts>2) or {tuple(x) for x in unique[counts==1]}!={tuple(x) for x in np.sort(self.faces,axis=1)}:
            raise ValueError('Complete unambiguous tetrahedral boundary faces required')
        D=np.swapaxes(self.points[self.tets[:,1:]]-self.points[self.tets[:,:1]],1,2)
        self.signed_volumes=np.linalg.det(D)/6.
        if not np.isfinite(self.signed_volumes).all() or np.any(self.signed_volumes==0):raise ValueError('Nondegenerate finite rest tetrahedra required')
        self.origin=array(coordinate_origin_world_m,(3,),'coordinate frame origin')
        self.handles=tuple(handles)
        if any(not isinstance(h,KnownHandle) for h in self.handles):raise TypeError('Explicit KnownHandle objects required')
        occupied=set()
        for h in self.handles:
            if h.ordered_patch_nodes.max()>=len(self.points):raise ValueError('Patch index outside rest mesh')
            if occupied.intersection(h.ordered_patch_nodes):raise ValueError('Overlapping rigid handles require a separately specified coupling model')
            occupied.update(h.ordered_patch_nodes)
            if h.origin_node_id is not None:
                if h.origin_node_id>=len(self.points) or not np.array_equal(h.origin_world_m,self.points[h.origin_node_id]):
                    raise ValueError('Declared origin node and supplied metric origin disagree')
        self.source_sha256=records(source_sha256)

    def features(self):
        N=len(self.points);H=len(self.handles);P=max((len(h.ordered_patch_nodes) for h in self.handles),default=0)
        ids=np.full((H,P),-1,dtype=np.int64);valid=np.zeros((H,P),dtype=bool)
        offsets=np.zeros((H,P,3));masks=np.zeros((H,N),dtype=bool)
        for j,h in enumerate(self.handles):
            n=len(h.ordered_patch_nodes);ids[j,:n]=h.ordered_patch_nodes;valid[j,:n]=True;masks[j,h.ordered_patch_nodes]=True
            offsets[j,:n]=(self.points[h.ordered_patch_nodes]-h.origin_world_m)@h.initial_rotation_body_to_world
        # Explicit reshape keeps H=0 free objects well-defined.
        stack=lambda name,shape:np.array([getattr(h,name) for h in self.handles],dtype=float).reshape((H,*shape))
        R=stack('initial_rotation_body_to_world',(3,3));c=stack('com_offset_body_m',(3,));I=stack('com_inertia_body_kg_m2',(3,3))
        origins=stack('origin_world_m',(3,))-self.origin
        masses=np.array([h.mass_kg for h in self.handles])
        return {'rest_points_relative_m':self.points-self.origin,'tetrahedra':self.tets.copy(),'surface_faces':self.faces.copy(),
            'tet_signed_rest_volumes_m3':self.signed_volumes.copy(),'handle_origins_relative_m':origins,
            'handle_origin_node_ids':np.array([-1 if h.origin_node_id is None else h.origin_node_id for h in self.handles],dtype=np.int64),
            'ordered_patch_node_ids':ids,'ordered_patch_valid_mask':valid,'patch_node_masks':masks,
            'ordered_patch_local_offsets_body_m':offsets,'handle_masses_kg':masses,
            'handle_COM_inertias_body_kg_m2':I,'handle_COM_offsets_body_m':c,'initial_handle_rotations_body_to_world':R,
            'handle_COM_relative_m':origins+np.einsum('hij,hj->hi',R,c),
            'handle_COM_inertias_world_kg_m2':R@I@np.swapaxes(R,-1,-2),
            'handle_inertias_about_origin_body_kg_m2':I+masses[:,None,None]*(np.sum(c*c,axis=1)[:,None,None]*np.eye(3)-c[:,:,None]*c[:,None,:])}

    def provenance(self):
        return provenance('public_known_hardware',self.features(),self.source_sha256,absolute_inputs={
            'supplied_rest_points_world_m':self.points,'supplied_coordinate_origin_world_m':self.origin,
            'supplied_handle_origins_world_m':np.array([h.origin_world_m for h in self.handles]).reshape(-1,3)})


@dataclass(frozen=True,kw_only=True)
class MaterialHypothesis:
    youngs_modulus_Pa:float
    poissons_ratio:float
    density_kg_m3:float
    body_linear_drag_s_inverse:float
    constitutive_model:str

    def __post_init__(self):
        for name in ['youngs_modulus_Pa','density_kg_m3']:
            object.__setattr__(self,name,positive(getattr(self,name),name))
        object.__setattr__(self,'body_linear_drag_s_inverse',positive(self.body_linear_drag_s_inverse,'body drag',zero=True))
        if not np.isfinite(self.poissons_ratio) or not -1<self.poissons_ratio<.5:raise ValueError('Stable isotropic Poisson ratio required')
        if self.constitutive_model not in ['log_J_neo_hookean','small_strain_isotropic_linear']:
            raise ValueError('Explicit supported constitutive model required; no native-equivalence default')

    def features(self):
        return {'homogeneous_material_parameters':np.array([self.youngs_modulus_Pa,self.poissons_ratio,self.density_kg_m3,self.body_linear_drag_s_inverse]),
            'constitutive_model_one_hot':np.array([self.constitutive_model==name for name in ['log_J_neo_hookean','small_strain_isotropic_linear']],dtype=bool)}


class BoundaryHypothesis:
    """Latent support membership/origins; they are not public hardware labels."""
    def __init__(self,*,constraint_axis_masks,support_origins_relative_m):
        self.masks=array(constraint_axis_masks,(None,None,3),'support constraint masks',bool)
        S,N,_=self.masks.shape
        self.origins=array(support_origins_relative_m,(S,3),'hypothesized support origins')
        if np.any(self.masks.sum(0)>1):raise ValueError('A node axis cannot be prescribed by two support groups')
        self.node_count=N;self.support_count=S

    def features(self):
        return {'support_constraint_axis_masks':self.masks.copy(),'support_origins_relative_m':self.origins.copy()}


class ForceProgram:
    """One additive known command OR latent passive-force hypothesis.

    The tag is mandatory. An unknown passive history must never be tagged as
    a known command merely because a training recipe contains its true value.
    Separate programs allow known pokes plus unknown background forcing.
    """
    def __init__(self,*,held_handle_forces_world_N,held_handle_torques_world_Nm,
                 held_nodal_forces_world_N,held_body_acceleration_world_m_s2,
                 information_role,source_sha256):
        if information_role not in ['known_actuator_command','candidate_force_hypothesis']:
            raise ValueError('Explicit known-command or candidate-force information role required')
        self.information_role=information_role
        self.forces=array(held_handle_forces_world_N,(None,None,3),'held handle forces')
        T,H,_=self.forces.shape
        self.torques=array(held_handle_torques_world_Nm,(T,H,3),'held handle torques')
        self.nodal=array(held_nodal_forces_world_N,(T,None,3),'held nodal forces')
        self.body_acceleration=array(held_body_acceleration_world_m_s2,(T,3),'held body acceleration')
        self.source_sha256=records(source_sha256)

    def features(self):
        return {'held_handle_forces_world_N':self.forces.copy(),'held_handle_torques_world_Nm':self.torques.copy(),
            'held_nodal_forces_world_N':self.nodal.copy(),'held_body_acceleration_world_m_s2':self.body_acceleration.copy()}

    def provenance(self):
        return provenance(self.information_role,self.features(),self.source_sha256)


class HeldCommandQuery:
    """Explicit time grid, role-tagged forcing and support endpoint program.

    Handle forces act at moving handle ORIGINS, with additional world pure
    torques. Support rotations act about latent support origins. No continuous
    interpolation is inferred. Future commands are allowed; RGB/responses are
    not. A candidate support law is excluded from public image conditioning.
    """
    def __init__(self,*,boundary_times_s,force_programs,support_translations_world_m,
                 support_rotations_world,support_law_role,support_law_source_sha256,
                 requested_output_times_s,source_sha256):
        self.times=array(boundary_times_s,(None,),'command boundary times')
        if len(self.times)<2 or self.times[0]!=0. or np.any(np.diff(self.times)<=0):raise ValueError('Force intervals must increase strictly from the declared rest start at zero')
        T=len(self.times)-1
        self.force_programs=tuple(force_programs)
        if not self.force_programs or any(not isinstance(program,ForceProgram) for program in self.force_programs):
            raise TypeError('At least one explicit role-tagged ForceProgram is required, including for all-zero loading')
        self.handle_count=self.force_programs[0].forces.shape[1];self.node_count=self.force_programs[0].nodal.shape[1]
        if any(program.forces.shape!=(T,self.handle_count,3) or program.nodal.shape!=(T,self.node_count,3) for program in self.force_programs):
            raise ValueError('All additive force programs must share command/hardware/node dimensions')
        if support_law_role not in ['known_actuator_command','candidate_boundary_hypothesis']:
            raise ValueError('Explicit known-command or candidate-boundary support-law information role required')
        self.support_law_role=support_law_role;self.support_law_source_sha256=records(support_law_source_sha256)
        self.support_translation=array(support_translations_world_m,(T+1,None,3),'prescribed support endpoint translations')
        self.support_count=self.support_translation.shape[1]
        self.support_rotation=rotations(support_rotations_world,(T+1,self.support_count,3,3),'prescribed support endpoint rotations')
        if np.any(self.support_translation[0]!=0) or not np.array_equal(self.support_rotation[0],np.broadcast_to(np.eye(3),self.support_rotation[0].shape)):
            raise ValueError('Initial support commands must agree with the declared undeformed rest state')
        self.outputs=array(requested_output_times_s,(None,),'requested output times')
        if not len(self.outputs) or np.any(np.diff(self.outputs)<0) or self.outputs.min()<self.times[0] or self.outputs.max()>self.times[-1]:
            raise ValueError('Ordered output times must lie inside the supplied command window')
        if not np.isin(self.outputs,self.times).all():raise ValueError('Output times require supplied exact support endpoint commands; add them to the command grid')
        self.source_sha256=records(source_sha256)

    def features(self):
        dt=np.diff(self.times)
        return {'command_boundary_times_s':self.times.copy(),'force_intervals_s':np.stack([self.times[:-1],self.times[1:]],axis=1),
            'held_interval_durations_s':dt,
            'requested_output_times_s':self.outputs.copy()}

    def support_features(self):
        return {'prescribed_support_endpoint_translations_world_m':self.support_translation.copy(),
            'prescribed_support_endpoint_rotations_world':self.support_rotation.copy()}

    def forcing_features(self,role):
        selected=[program for program in self.force_programs if program.information_role==role]
        T=len(self.times)-1;shapes={'held_handle_forces_world_N':(T,self.handle_count,3),
            'held_handle_torques_world_Nm':(T,self.handle_count,3),'held_nodal_forces_world_N':(T,self.node_count,3),
            'held_body_acceleration_world_m_s2':(T,3)}
        attributes=dict(zip(shapes,['forces','torques','nodal','body_acceleration']))
        return {key:np.stack([getattr(program,attributes[key]) for program in selected]) if selected else np.empty((0,*shape)) for key,shape in shapes.items()}

    def provenance(self):
        return provenance('public_known_time_query',self.features(),self.source_sha256)


def schema():
    """Named structured features; no lossy single 25D compression is defined."""
    groups={
        'public_known_hardware':{
            'rest_points_relative_m':('N,3','m'),'tetrahedra':('E,4','node_id'),'surface_faces':('F,3','node_id'),
            'tet_signed_rest_volumes_m3':('E','m^3'),'handle_origins_relative_m':('H,3','m'),
            'handle_origin_node_ids':('H','node_id; -1 means origin not identified as a mesh node'),
            'ordered_patch_node_ids':('H,P','node_id; -1 padding'),'ordered_patch_valid_mask':('H,P','bool'),
            'patch_node_masks':('H,N','bool'),'ordered_patch_local_offsets_body_m':('H,P,3','m'),
            'handle_masses_kg':('H','kg'),'handle_COM_inertias_body_kg_m2':('H,3,3','kg*m^2'),
            'handle_COM_offsets_body_m':('H,3','m'),'initial_handle_rotations_body_to_world':('H,3,3','1'),
            'handle_COM_relative_m':('H,3','m'),'handle_COM_inertias_world_kg_m2':('H,3,3','kg*m^2'),
            'handle_inertias_about_origin_body_kg_m2':('H,3,3','kg*m^2')},
        'public_known_commands':{
            'command_boundary_times_s':('T+1','s'),'force_intervals_s':('T,2','s'),'held_interval_durations_s':('T','s'),
            'requested_output_times_s':('Q','s')},
        'latent_material_hypothesis':{
            'homogeneous_material_parameters':('4','[E:Pa,nu:1,rho:kg/m^3,body_drag:1/s]'),
            'constitutive_model_one_hot':('2','[log_J_neo_hookean,small_strain_isotropic_linear]')},
        'latent_boundary_hypothesis':{
            'support_constraint_axis_masks':('S,N,3','bool'),'support_origins_relative_m':('S,3','m')}}
    for role,count in [('public_known_force_programs','K'),('latent_force_hypotheses','L')]:
        groups[role]={'held_handle_forces_world_N':(count+',T,H,3','N'),
            'held_handle_torques_world_Nm':(count+',T,H,3','N*m'),'held_nodal_forces_world_N':(count+',T,N,3','N'),
            'held_body_acceleration_world_m_s2':(count+',T,3','m/s^2')}
    support_fields={'prescribed_support_endpoint_translations_world_m':('T+1,S,3','m'),
        'prescribed_support_endpoint_rotations_world':('T+1,S,3,3','1')}
    groups['public_known_support_program']=support_fields;groups['latent_support_law_hypothesis']=support_fields
    return {'version':VERSION,'dimensions':{'N':'rest nodes','E':'tetrahedra','F':'complete boundary faces','H':'rigid handles',
        'P':'maximum ordered patch length, with explicit padding mask','T':'held command intervals','S':'declared support-program capacity, not observed active support count','Q':'requested times',
        'K':'explicit known additive force programs','L':'explicit latent additive force hypotheses'},
        'groups':{role:{key:{'shape':shape,'units':units} for key,(shape,units) in fields.items()} for role,fields in groups.items()},
        'spatial_frame':'World axes; point coordinates are relative to the explicitly supplied known coordinate origin. Absolute origin is provenance/output-restoration metadata, not a model feature.',
        'initial_state_contract':'Undeformed rest geometry, supplied initial rigid orientations and zero nodal/rigid velocities at t=0. Preload, nonzero initial velocity and continuous support interpolation require an explicitly extended contract.',
        'attachment_contract':'Each complete ordered handle patch follows one exact finite rigid transform. Handle mass is additional to the deformable body mass; inertia is about the handle COM in its supplied body frame. No attachment compliance or native-engine equivalence is implied.',
        'external_load_contract':'All external loads are explicit: handle-origin forces/world pure torques, nodal forces and uniform body acceleration. Density-dependent gravity need not be converted into oracle nodal forces. No implicit contact/environment or hidden background load is supplied.',
        'information_roles':'Only explicitly tagged known force/support programs enter public image conditioning. Passive force histories and unknown support motion belong to hypothesis groups. Program tags are caller declarations, not automatic evidence that recipe values were observable.',
        'support_semantics':'Exact discrete endpoint transforms about latent support origins, applied on latent node-axis masks. No hidden interpolation, observed response or support truth is loaded.'}


def inference_public_context(*,known_hardware,command_query):
    """Only public input groups; this signature cannot accept hypotheses."""
    if not isinstance(known_hardware,KnownHardware) or not isinstance(command_query,HeldCommandQuery):raise TypeError('Explicit public typed inputs required')
    if len(known_hardware.handles)!=command_query.handle_count or len(known_hardware.points)!=command_query.node_count:
        raise ValueError('Command hardware/node dimensions disagree')
    features={'public_known_hardware':known_hardware.features(),'public_known_commands':command_query.features(),
        'public_known_force_programs':command_query.forcing_features('known_actuator_command')}
    proof={'known_hardware':known_hardware.provenance(),'command_query':command_query.provenance(),
        'known_force_programs':[program.provenance() for program in command_query.force_programs if program.information_role=='known_actuator_command']}
    if command_query.support_law_role=='known_actuator_command':
        features['public_known_support_program']=command_query.support_features()
        proof['known_support_program']=provenance('known_actuator_command',command_query.support_features(),command_query.support_law_source_sha256)
    return {'feature_groups':features,'schema':schema(),'provenance':proof,
        'role':'public_input_only_no_material_or_boundary_labels'}


def conditioned_forward_context(*,known_hardware,command_query,material_hypothesis,boundary_hypothesis,
                                hypothesis_role,hypothesis_source_sha256):
    """Explicit hypotheses for matching-prior calls or known-parameter training.

    A diagnostic/training truth label can enter only through this conspicuously
    labelled hypothesis path, never by constructing public image conditioning.
    """
    if not isinstance(material_hypothesis,MaterialHypothesis) or not isinstance(boundary_hypothesis,BoundaryHypothesis):
        raise TypeError('Explicit material and boundary hypothesis objects required')
    if hypothesis_role not in ['candidate_from_prior_or_posterior','known_parameter_training_or_diagnostic']:
        raise ValueError('Explicit hypothesis provenance role required')
    result=inference_public_context(known_hardware=known_hardware,command_query=command_query)
    if boundary_hypothesis.node_count!=len(known_hardware.points) or boundary_hypothesis.support_count!=command_query.support_count:
        raise ValueError('Latent boundary masks and public support programs disagree')
    if np.any(boundary_hypothesis.masks[:,np.any(known_hardware.features()['patch_node_masks'],axis=0)]):
        raise ValueError('Support/rigid-patch overlap requires an explicitly extended coupling contract')
    latent={'latent_material_hypothesis':material_hypothesis.features(),'latent_boundary_hypothesis':boundary_hypothesis.features(),
        'latent_force_hypotheses':command_query.forcing_features('candidate_force_hypothesis')}
    if command_query.support_law_role=='candidate_boundary_hypothesis':latent['latent_support_law_hypothesis']=command_query.support_features()
    result['feature_groups'].update(latent)
    flat={role+'/'+key:value for role,fields in latent.items() for key,value in fields.items()}
    result['provenance']['hypotheses']=provenance(hypothesis_role,flat,hypothesis_source_sha256)
    result['provenance']['hypotheses']['force_programs']=[program.provenance() for program in command_query.force_programs if program.information_role=='candidate_force_hypothesis']
    if command_query.support_law_role=='candidate_boundary_hypothesis':
        result['provenance']['hypotheses']['support_law']=provenance('candidate_boundary_hypothesis',command_query.support_features(),command_query.support_law_source_sha256)
    proof=result['provenance']['hypotheses']
    proof['manifest_sha256']=digest({key:value for key,value in proof.items() if key!='manifest_sha256'})
    result['role']='forward_conditioned_on_explicit_hypotheses'
    result['hypothesis_role']=hypothesis_role
    return result


def combined_forward_forcing(command_query):
    """Sum explicit known+latent programs for physics/forward calls only."""
    if not isinstance(command_query,HeldCommandQuery):raise TypeError('Explicit HeldCommandQuery required')
    known=command_query.forcing_features('known_actuator_command');latent=command_query.forcing_features('candidate_force_hypothesis')
    return {'features':{key:known[key].sum(0)+latent[key].sum(0) for key in known},
        'role':'combined_known_and_hypothesized_forcing_not_public_image_conditioning'}


def support_targets_relative(*,known_hardware,boundary_hypothesis,command_query):
    """Full endpoint targets plus masks, not a public/observed state feature."""
    if boundary_hypothesis.node_count!=len(known_hardware.points) or boundary_hypothesis.support_count!=command_query.support_count:
        raise ValueError('Support dimensions disagree')
    rest=known_hardware.points-known_hardware.origin
    local=rest[None]-boundary_hypothesis.origins[:,None,:]
    targets=np.einsum('tsij,snj->tsni',command_query.support_rotation,local)
    targets+=boundary_hypothesis.origins[None,:,None,:]+command_query.support_translation[:,:,None,:]
    return {'targets_relative_m':targets,'constraint_axis_masks':boundary_hypothesis.masks.copy(),
        'role':'derived_from_explicit_latent_boundary_and_public_command_not_observation'}


def rigid_kinetic_energy(handle,*,origin_velocity_world_m_s,angular_velocity_world_rad_s):
    """Mechanics check demonstrating why COM and full inertia cannot be dropped."""
    v=array(origin_velocity_world_m_s,(3,),'origin velocity');w=array(angular_velocity_world_rad_s,(3,),'angular velocity')
    R=handle.initial_rotation_body_to_world;v_com=v+np.cross(w,R@handle.com_offset_body_m)
    return float(.5*handle.mass_kg*(v_com@v_com)+.5*w@(R@handle.com_inertia_body_kg_m2@R.T)@w)
