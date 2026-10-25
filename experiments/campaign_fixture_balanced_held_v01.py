"""SOURCE-only balanced-held DEVELOPMENT task, never edits the old task.

First9 known loads and first12 scalar channels are unchanged. Only two held
world-force directions/IDs change; normalized Gaussian scalar coefficients,
geometry, support bank, material priors and all16 query positions stay fixed.
No mechanics, observations, noise or RNG executes on import.
"""
import copy
from types import SimpleNamespace
import numpy as np
from experiments import campaign_fixture_gaussian_linear_screen_v01 as original
from experiments.campaign_surface_p2_inference_v01 import LabelledModal

geometry=original.geometry
bank=original.bank
LAYOUTS=bank.LAYOUTS;ORIGIN_M=bank.ORIGIN_M;SIGMA_M=bank.SIGMA_M
make_forward=geometry.make_forward
HELD_RENAME={'held-offaxis-down':'held-oblique-positiveY-down','held-offaxis-axial':'held-oblique-negativeY-down'}
HELD_DIRECTIONS=(np.array([0.,1.,-1.])/np.sqrt(2),np.array([0.,-1.,-1.])/np.sqrt(2))


def load_specs():
    specs=copy.deepcopy(bank.load_specs())
    for spec,direction in zip(specs[9:],HELD_DIRECTIONS):
        spec['id']=HELD_RENAME[spec['id']];spec['direction']=direction.tolist()
    return specs


def load_catalog(n,*,order=16):
    old,old_info=geometry.load_catalog(n,order=order);old_specs=bank.load_specs();specs=load_specs()
    answer={s['id']:old[s['id']].copy() for s in specs[:9]}
    diagnostics={s['id']:copy.deepcopy(old_info[s['id']]) for s in specs[:9]}
    X=geometry.geometry(n)[0]
    for j,spec in enumerate(specs[9:]):
        old_id=old_specs[9+j]['id'];source=old[old_id]
        # c=F*normalized_signed_T6_weight is already compiled independently of
        # direction. No discrete renormalization or projection of responses.
        coefficient=-source[:,2] if j==0 else source[:,0]
        d=np.asarray(spec['direction']);force=coefficient[:,None]*d[None,:]
        centroid=np.asarray(old_info[old_id]['canonical_truncated_centroid_m'])
        moment=np.cross(X-ORIGIN_M,force).sum(0);expected=np.cross(centroid-ORIGIN_M,spec['force_N']*d)
        info=copy.deepcopy(old_info[old_id]);info.update(id=spec['id'],direction=d.tolist(),net_force_N=force.sum(0).tolist(),net_force_error_N=float(np.max(abs(force.sum(0)-spec['force_N']*d))),moment_Nm=moment.tolist(),analytic_moment_Nm=expected.tolist(),moment_error_Nm=float(np.max(abs(moment-expected))),task_version='balanced-held-v01',scalar_Gaussian_coefficients_from_unchanged_patch=True,original_scalar_patch_id=old_id,world_dead_load=True,renormalized=False)
        answer[spec['id']]=force;diagnostics[spec['id']]=info
    return answer,diagnostics


def measurement_specs():
    specs=copy.deepcopy(bank.measurement_specs())
    for spec in specs:
        if spec['load'] in HELD_RENAME:
            old=spec['load'];new=HELD_RENAME[old];spec['load']=new;spec['id']=new+spec['id'][len(old):]
    return specs


def measurement_points():
    return np.array(list(dict.fromkeys(tuple(s['point_m']) for s in measurement_specs())))


def labels():
    return {role:tuple(s['id'] for s in measurement_specs() if s['role']==role) for role in ('prefix','candidate','target')}


def measurement_response(n,fields):
    columns=[]
    for spec in measurement_specs():
        u=np.asarray(fields[spec['load']],float)
        if u.ndim!=3:raise ValueError('Explicit [batch,Nnodes,3]')
        value=geometry.field_queries(n,u,np.array([spec['point_m']]))[:,0]@np.asarray(spec['projection']);columns.append(value)
    return np.stack(columns,axis=1)[:,:,None]


def make_measurement_modal(forward,loads):
    points=measurement_points();base=LabelledModal.from_forward(forward,loads=loads,points=points,condensed=True,query_matrix=geometry.point_query_matrix)
    specs=measurement_specs();rows=[]
    for spec in specs:
        i=next(i for i,p in enumerate(points) if np.array_equal(p,spec['point_m']));j=base.labels.index(spec['load'])
        rows.append(np.asarray(spec['projection'])@base.coefficients[3*i:3*i+3,j,:])
    modal=SimpleNamespace(eigenvalues=base.modal.eigenvalues.copy(),coefficients=np.asarray(rows),output_shape=(1,len(specs)),diagnostics=dict(base.modal.diagnostics,measurement_selection='Balanced-held-v01; samefirst12channels,48honestlyrenamedtargets; allmodes+exactSchur1/Eparticular'))
    return LabelledModal(modal,tuple(s['id'] for s in specs))


def design():
    d=copy.deepcopy(bank.design());d.update(schema='fixture-balanced-held-development-v01',loads=load_specs(),measurements=measurement_specs(),geometry_supports_material_priors_unchanged=True,first9loads_first12channels_unchanged=True,query_points_unchanged=True,task_change='Only heldload directions/IDs change; newdevelopmenttask, not repair/relabel of originalNEW36negative.',symmetry='180degreeZrotation about(.18,.04) maps theheldpair/supportcategories/target8queries/candidate8set. Prefix4markers and3DFreudenthaltriangulation areNOTexactlyinvariant; no conditional/universal-policy guarantee.',control_semantics='Separate knownresetstatic heldloads, NOT simultaneous zeroresultant/moment pair.',NH_reuse='Existing6NH3/6/12prefix+far-candidate fields have identicalloads/querypositions and remainhash-reusable. No newNHfields authorized here.',scope='SOURCE-only newheldobjective/compiler; fullmode linear/reference/utility andnewheldequilibrium validations are separate. No automaticphysics/policy/calibration admission.')
    return d
