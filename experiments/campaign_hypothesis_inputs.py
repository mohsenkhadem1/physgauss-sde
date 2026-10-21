"""Finite TRAIN counterfactual hypotheses under an explicit hardware-first prior.

No response, observation, checkpoint or physics worker is loaded. Full public
hardware is fixed before new BC/material draws. Original25D decode is NOT used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

try:
    from campaign_physical_prior import ConditionalPhysicalPrior,PriorState,BOUNDARY_TYPES,FACE_NORMALS,FACE_BASES
    from campaign_full_hardware_context import KnownHardware,KnownHandle
    from boundary_block import block_mesh
except ImportError:
    from experiments.campaign_physical_prior import ConditionalPhysicalPrior,PriorState,BOUNDARY_TYPES,FACE_NORMALS,FACE_BASES
    from experiments.campaign_full_hardware_context import KnownHardware,KnownHandle
    from experiments.boundary_block import block_mesh

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'configs/gaussian_independent_diagnostic_inputs_v01.json'
FIXTURE_SHA='073380f72559c50ccc674d5f69fdff1be7e4d0b30484ad66f0233695a813847f'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:plain(v) for key,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def public_hardware_contexts(fixture_path=None):
    """Read ONLY the four allowed public TRAIN static H; strip source physics.

    Rest mesh uses the declared cells; handle patches/masses/inertias reproduce
    authored known hardware. Source BC/material labels never become new truth.
    """
    path=FIXTURE if fixture_path is None else Path(fixture_path)
    if sha(path)!=FIXTURE_SHA:raise ValueError('Exact public TRAIN static fixture required')
    fixture=json.loads(path.read_text());contexts=[]
    if [c['episode_id'] for c in fixture['cases']]!=['t0000','t0001','t0002','t0003']:raise ValueError('Four allowed TRAIN static contexts required')
    for index,item in enumerate(fixture['cases']):
        if item['recipe']['split']!='train':raise ValueError('No reserved context access')
        points,tets,faces=block_mesh(cells=item['recipe']['cells'])
        arrays={'rest_points_m':points,'tetrahedra':tets,'surface_faces':faces}
        for row in item['reference_static_arrays']:
            a=arrays[row['key']]
            if str(a.dtype)!=row['dtype'] or list(a.shape)!=row['shape'] or hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()!=row['sha256']:raise ValueError('Public static mesh changed')
        spec=item['boundaries'];handles=[]
        for origin,patch in zip(spec['load_nodes'],spec['load_patches']):
            local=points[patch]-points[origin];mass=spec['handle_mass_kg_each']
            diagonal=np.maximum(mass*np.mean(np.sum(local**2,axis=1)[:,None]-local**2,axis=0),.4*mass*.003**2)
            handles.append({'origin_world_m':points[origin].tolist(),'ordered_patch_nodes':list(patch),'mass_kg':mass,
                'com_inertia_body_kg_m2':np.diag(diagonal).tolist(),'com_offset_body_m':[0.,0.,0.],
                'initial_rotation_body_to_world':np.eye(3).tolist(),'origin_node_id':origin})
        public={'id':'H'+str(index),**plain(arrays),'handles':handles,'coordinate_origin_world_m':points.mean(0).tolist(),
            'public_hardware_provenance':{'source_fixture_sha256':FIXTURE_SHA,'source_static_H_id':item['episode_id'],
                'source_material_BC_not_new_hypothesis':True,'authored_handle_inertia_not_native_measurement':True}}
        public['physical_hardware_sha256']=physical_hardware_digest(known_hardware_from_public(public))
        public['public_hardware_sha256']=digest(public);contexts.append(public)
    return contexts


def known_hardware_from_public(public):
    return KnownHardware(rest_points_world_m=np.asarray(public['rest_points_m']),tetrahedra=np.asarray(public['tetrahedra']),
        surface_faces=np.asarray(public['surface_faces']),handles=[KnownHandle(**h) for h in public['handles']],
        coordinate_origin_world_m=public['coordinate_origin_world_m'],source_sha256={})


def make_prior(public):
    known=known_hardware_from_public(public)
    if len(known.handles)!=2 or any(len(h.ordered_patch_nodes)!=9 for h in known.handles):raise ValueError('Exact H2x9 public patches required')
    return ConditionalPhysicalPrior(known.points,np.array([h.origin_world_m for h in known.handles]),
        handle_patch_node_ids=np.stack([h.ordered_patch_nodes for h in known.handles]))


def physical_hardware_digest(known):
    return digest(plain({'points':known.points,'tetrahedra':known.tets,'surface_faces':known.faces,
        'origin':known.origin,'handles':[vars(h) for h in known.handles]}))


def conditioned_BC_sample(prior,boundary_index,rng,*,maximum_draws=10000):
    """Reject categories BEFORE dynamics, retaining the declared p(state|BC,H).

    A finite-cap failure aborts; never changes the requested BC or H. Materials
    are independently overwritten only by the separately drawn common tuple.
    """
    if boundary_index not in range(4) or isinstance(boundary_index,bool):raise ValueError('Exact BC index0..3 required')
    if type(maximum_draws) is not int or maximum_draws<1:raise ValueError('Positive bounded category-draw cap required')
    for count in range(1,maximum_draws+1):
        state=prior.sample(1,rng)
        if int(state.boundary_type[0])==boundary_index:return state,{'category_draws':count,'maximum_draws':maximum_draws,'log_category_failure_probability_bound':maximum_draws*math.log(.75)}
    raise RuntimeError('BC-conditioned prior category budget exhausted; abort entire finite design')


def realize_hypothesis(public,state,prior=None):
    """Explicit complete core input from normalized intrinsic state; no25D map."""
    prior=make_prior(public) if prior is None else prior
    if len(state)!=1 or not np.isfinite(prior.log_prob(state)).all():raise ValueError('One finite normalized-prior state required')
    known=known_hardware_from_public(public)
    if not np.array_equal(prior.reference_points,known.points) or not np.array_equal(prior.handle_patch_node_ids,np.stack([h.ordered_patch_nodes for h in known.handles])):raise ValueError('Prior must condition on this complete public H')
    x=state.coordinates[0];bc=int(state.boundary_type[0]);face=int(state.support_face[0]);cells=state.support_cells[0]
    cell_xy=np.stack([cells//prior.grid,cells%prior.grid],axis=-1)
    uv=-prior.support_uv_limit+(cell_xy+x[4:8].reshape(2,2))*(2*prior.support_uv_limit/prior.grid)
    faces=[face,(face+3)%6];origins=[]
    for tangent,f in zip(uv,faces):origins.append(prior.center+FACE_NORMALS[f]*prior.size/2+np.einsum('i,ij->j',tangent,FACE_BASES[f])*prior.size)
    counts=[1,16,9,16];active=[True,bc==2];groups=[];mask=np.zeros((2,len(known.points),3),dtype=bool)
    attachment=np.unique(np.concatenate([h.ordered_patch_nodes for h in known.handles]));used=set()
    for group,on in enumerate(active):
        if not on:groups.append([]);continue
        f=faces[group];axis=f%3;bound=prior.minimum[axis] if f<3 else prior.maximum[axis]
        candidates=np.flatnonzero(np.isclose(known.points[:,axis],bound,atol=prior.surface_tolerance_m,rtol=0))
        distance=np.linalg.norm(known.points[candidates]-origins[group],axis=1)
        selected=candidates[np.argsort(distance,kind='stable')[:counts[bc]]]
        if len(selected)!=counts[bc] or np.max(np.linalg.norm(known.points[selected]-origins[group],axis=1))>.09:raise ValueError('Actual realized compact support invalid')
        if np.min(np.linalg.norm(known.points[selected,None]-known.points[attachment][None],axis=-1))<=prior.support_handle_clearance_m:raise ValueError('Actual full-patch support clearance failed')
        if used.intersection(selected.tolist()):raise ValueError('Actual support groups overlap')
        used.update(selected.tolist());groups.append(selected.tolist());mask[group,selected]=True
    motion_direction=np.cos(x[10])*FACE_BASES[face,0]+np.sin(x[10])*FACE_BASES[face,1]
    material={'youngs_modulus_Pa':float(np.exp(x[0])),'density_kg_m3':float(np.exp(x[1])),'poissons_ratio':float(x[2]),'body_linear_drag_s_inverse':float(np.exp(x[3]))}
    return {'boundary_kind':BOUNDARY_TYPES[bc],'material':material,'fixture_nodes':sorted(used),
        'support_node_groups':groups,'support_node_axis_masks':mask.tolist(),'support_active':active,
        'support_origins_world_m':plain(origins),'motion_amplitude_m':float(x[11]) if bc==3 else 0.,
        'motion_direction_world':motion_direction.tolist(),'seconds':3.,
        'support_law':{'kind':'analytic_sine_squared_translation','information_role':'candidate_boundary_hypothesis','seconds':3.,'initial_rest_compatible':True,'initial_velocity_zero':True},
        'capacities':{'S':2,'H':2},'exact_fixture_and_rigid_hardware_not_Gaussian_springs':True,
        'retained_prior_state':plain(vars(state)),'prior_sha256':prior.metadata()['prior_sha256'],
        'intrinsic_joint_log_density':float(prior.log_prob(state)[0]),
        'known_programs_override_passive_force_nuisances':True,'public_hardware_id':public['id'],
        'public_hardware_sha256':public['public_hardware_sha256'],
        'physical_hardware_sha256':physical_hardware_digest(known)}


def validate_realized_hypothesis(hardware,spec):
    """Reconstruct from retained intrinsic state, checking complete public H.

    H identity/provenance is audited by the manifest guard; physical hashes here
    include all ordered patches/origins/mass/fullI/COM/initialR and rest mesh.
    """
    if physical_hardware_digest(hardware)!=spec['physical_hardware_sha256']:raise ValueError('Realized hypothesis/public complete hardware mismatch')
    public={'id':spec['public_hardware_id'],'public_hardware_sha256':spec['public_hardware_sha256'],
        'rest_points_m':hardware.points,'tetrahedra':hardware.tets,'surface_faces':hardware.faces,
        'handles':[plain(vars(h)) for h in hardware.handles],'coordinate_origin_world_m':hardware.origin}
    raw=spec['retained_prior_state']
    state=PriorState(**{name:np.asarray(raw[name]) for name in PriorState.__dataclass_fields__})
    expected=realize_hypothesis(public,state)
    if expected!=spec:raise ValueError('Explicit hypothesis differs from normalized-prior realization')
    return spec


def support_translation(spec,time_s):
    t=float(time_s)
    if not math.isfinite(t):raise ValueError('Finite support clock required')
    return spec['motion_amplitude_m']*np.sin(np.pi*np.clip(t/3.,0.,1.))**2*np.asarray(spec['motion_direction_world'])


def support_velocity(spec,time_s):
    t=float(time_s)
    if not math.isfinite(t):raise ValueError('Finite support clock required')
    return spec['motion_amplitude_m']*np.pi/3.*np.sin(2*np.pi*t/3.)*np.asarray(spec['motion_direction_world']) if 0<t<3 else np.zeros(3)


def make_manifest(*,program_ids,seed=20261022,material_draw_count=3):
    if len(program_ids)!=2 or len(set(program_ids))!=2 or any(not isinstance(p,str) or not p for p in program_ids):raise ValueError('Exactly two externally specified known program IDs required')
    if type(material_draw_count) is not int or not 1<=material_draw_count<=32:raise ValueError('Bounded material draw count1..32 required')
    contexts=public_hardware_contexts();priors=[make_prior(h) for h in contexts]
    rng_material=np.random.default_rng(np.random.SeedSequence([seed,1]));draw=priors[0].sample(material_draw_count,rng_material)
    materials=draw.coordinates[:,:4].copy();strata=np.linspace(np.log(1500.),np.log(20000.),material_draw_count+1)
    for mi in range(material_draw_count):materials[mi,0]=rng_material.uniform(strata[mi],strata[mi+1])
    hypotheses=[];cells=[]
    for hi,(public,prior) in enumerate(zip(contexts,priors)):
        # Geometry/motion/force nuisances are shared across the three materials.
        support_draws={}
        for bc in range(4):
            rng=np.random.default_rng(np.random.SeedSequence([seed,2,hi,bc]))
            support_draws[bc]=conditioned_BC_sample(prior,bc,rng)
        for mi,material_coordinates in enumerate(materials):
            for bc in range(4):
                state,diagnostic=support_draws[bc];coordinates=state.coordinates.copy();coordinates[0,:4]=material_coordinates
                state=state.with_coordinates(coordinates);spec=realize_hypothesis(public,state,prior)
                id=public['id']+'__m'+str(mi)+'__bc'+str(bc)
                hypotheses.append({'id':id,'material_tuple_id':'m'+str(mi),'material_stratum':mi,
                    'log_E_stratum_bounds':strata[mi:mi+2].tolist(),'E_stratum_conditional_density_factor':float(material_draw_count),
                    'individual_base_prior_log_density':spec['intrinsic_joint_log_density'],
                    'individual_conditional_BC_and_E_stratum_log_density':spec['intrinsic_joint_log_density']+math.log(4)+math.log(material_draw_count),
                    'support_draw_id':public['id']+'__bc'+str(bc),'public_hardware_id':public['id'],'spec':spec,'category_sampling':diagnostic})
                for pi,program in enumerate(program_ids):
                    role='fit' if mi in [0,2] and pi==0 else ('development_force_transfer' if mi in [0,2] else 'unseen_middle_material')
                    if material_draw_count!=3:role='future_design_unassigned'
                    cells.append({'id':id+'__'+program,'hypothesis_id':id,'public_hardware_id':public['id'],'program_id':program,'split':'train','evaluation_role':role,'status':'DECLARED_NOT_EXECUTED'})
    return {'schema':'finite-hardware-first-hypotheses-TRAIN-v0.1','status':'INPUT_DESIGN_ONLY_NOT_CAPTURE_AUTHORIZATION','seed':seed,
        'public_hardware':contexts,'shared_material_intrinsic_coordinates':materials.tolist(),'hypotheses':hypotheses,'cells':cells,
        'program_ids':list(program_ids),'material_draw_count':material_draw_count,'hypothesis_count':len(hypotheses),'capture_count':len(cells),
        'material_stratification':{'log_E_bounds':strata.tolist(),'one_draw_per_equal_logwidth_stratum':True,'conditional_density_factor':float(material_draw_count),'equal_random_stratum_mixture_matches_base_E_prior':True,'rho_nu_drag_original_marginals':True},
        'prior_contract':'Normalized atlas prior; conditionalBC support and E-stratification with recorded densityfactor. Equalstratum mixture recoversbaseE. No25D decode; no physics filtering.',
        'grouping':str(material_draw_count)+' independent conditional-stratum material tuples shared across4H/4BC;16H/BC support/motion/nuisance draws reused acrossmaterials. Dependent counterfactual cohort, not iid baseprior truth.',
        'fit_and_development_counts':{role:sum(c['evaluation_role']==role for c in cells) for role in sorted({c['evaluation_role'] for c in cells})},
        'limitations':['Three shared joint tuples cannot identify four separate material effects.','No full material/support generalization or SBC from this finite dependent cohort.','Larger draw counts do not authorize capture; a four-dimensional LHS design would require its own declared sampling law.'],
        'finite_design_BC':{'counts_per_publicH_material':[1,1,1,1],'conditioned_category_log_failure_bound_per_call':10000*math.log(.75)},
        'SBC_claim':False,'production_dataset_admitted':False,'source_sha256':{Path(__file__).relative_to(ROOT).as_posix():sha(__file__),
            'experiments/campaign_physical_prior.py':sha(ROOT/'experiments/campaign_physical_prior.py'),
            'experiments/campaign_full_hardware_context.py':sha(ROOT/'experiments/campaign_full_hardware_context.py'),
            'experiments/boundary_block.py':sha(ROOT/'experiments/boundary_block.py'),
            'configs/gaussian_independent_diagnostic_inputs_v01.json':FIXTURE_SHA}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--program-ids',nargs=2,required=True);parser.add_argument('--seed',type=int,default=20261022);parser.add_argument('--material-draw-count',type=int,default=3);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh input manifest required')
    result=make_manifest(program_ids=args.program_ids,seed=args.seed,material_draw_count=args.material_draw_count);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'hypotheses':result['hypothesis_count'],'capture_slots':result['capture_count'],'physics_runs':0}))
