"""New Gaussian-load DEVELOPMENT version; old sin-squared loads stay frozen.

Supports, material law, geometry, marker rules and directions are unchanged.
A known full-top Gaussian dead traction uses its analytic truncated centroid,
not its authored center, for resultant/moment validation. No physics on import.
"""
import copy
import numpy as np
from scipy.special import erf
from experiments import campaign_active_fixture_bank_v01 as original

core=original.core
LAYOUTS=original.LAYOUTS;ORIGIN_M=original.ORIGIN_M;SIGMA_M=original.SIGMA_M
LOAD_STD_M=(.035,.025);FORCE_N=.06;PREFIX_FORCE_N=.04
make_forward=original.make_forward
consistent_fixture=original.consistent_fixture
measurement_specs=original.measurement_specs
measurement_points=original.measurement_points
measurement_response=original.measurement_response
make_measurement_modal=original.make_measurement_modal
nonrigid_rms=original.nonrigid_rms
layout_spec=original.layout_spec


def load_specs():
    specs=copy.deepcopy(original.load_specs())
    for spec in specs:spec['force_N']=PREFIX_FORCE_N if spec['id']=='prefix-center-down' else FORCE_N
    return specs


def gaussian_integrals(center,std=LOAD_STD_M):
    c=np.asarray(center,float);s=np.asarray(std,float);upper=np.array([.36,.08])
    if c.shape!=(2,) or s.shape!=(2,) or not np.isfinite([*c,*s]).all() or np.any(c<0) or np.any(c>upper) or np.any(s<=0):raise ValueError('Public finite2D topcenter and positiveGaussianstd')
    a=-c/s;b=(upper-c)/s
    integrals=s*np.sqrt(np.pi/2)*(erf(b/np.sqrt(2))-erf(a/np.sqrt(2)))
    G=float(np.prod(integrals));centroid=c+s*s*(np.exp(-.5*a*a)-np.exp(-.5*b*b))/integrals
    if not np.isfinite(G) or G<=0:raise ValueError('Positive finite analytic rectangle normalizer')
    return G,np.r_[centroid,.08]


def surface_patch(n,spec,*,order=16):
    X,_,faces,_,_=core.geometry(n);c=np.asarray(spec['center_m'],float);d=np.asarray(spec['direction'],float);F=float(spec['force_N'])
    if c.shape!=(3,) or d.shape!=(3,) or c[2]!=.08 or not np.isfinite(c).all() or not np.isfinite(d).all() or not np.isclose(np.linalg.norm(d),1.,atol=1e-13,rtol=0) or not np.isfinite(F) or F<0:raise ValueError('Known finite topGaussian deadload and unitworlddirection')
    G,centroid=gaussian_integrals(c[:2]);b,w=core.triangle_rule(order);N=core.triangle_shapes(b)
    selected=faces[np.all(abs(X[faces[:,:3],2]-.08)<1e-14,axis=1)];vertices=X[selected[:,:3]]
    area=np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1)/2
    q=np.einsum('qa,fai->fqi',b,vertices);profile=np.exp(-.5*np.sum(((q[:,:,:2]-c[:2])/np.asarray(LOAD_STD_M))**2,axis=2))
    weights=np.zeros(len(X));local=np.einsum('f,q,fq,qa->fa',area,w,profile,N)/G;np.add.at(weights,selected.ravel(),local.ravel())
    force=F*weights[:,None]*d;moment=np.cross(X-ORIGIN_M,force).sum(0);expected=np.cross(centroid-ORIGIN_M,F*d);first=X.T@weights
    return force,dict(id=spec['id'],order=order,load_version='whole-top-Gaussian-v02',top_area_m2=.36*.08,authored_center_m=c.tolist(),std_m=list(LOAD_STD_M),analytic_normalizer_m2=G,canonical_truncated_centroid_m=centroid.tolist(),integrated_first_moment_m=first.tolist(),first_moment_error_m=float(np.max(abs(first-centroid))),integrated_weight_mass=float(weights.sum()),normalization_error=abs(float(weights.sum())-1),net_force_N=force.sum(0).tolist(),net_force_error_N=float(np.max(abs(force.sum(0)-F*d))),moment_Nm=moment.tolist(),analytic_moment_Nm=expected.tolist(),moment_error_Nm=float(np.max(abs(moment-expected))),equal_candidate_continuum_traction_norm_N=F,signed_T6_coefficients=True,world_dead_load=True,renormalized=False)


def load_catalog(n,*,order=16):
    loads={};info={}
    for spec in load_specs():loads[spec['id']],info[spec['id']]=surface_patch(n,spec,order=order)
    return loads,info


def design():
    d=copy.deepcopy(original.design());d.pop('patch_width_m')
    d.update(schema='active-fixture-Gaussian-load-development-v02',loads=load_specs(),load_std_m=list(LOAD_STD_M),traction='Analytically normalized Gaussian on ENTIRE top rectangle; no hardpatch edge or discrete renormalization. Actual truncatedcentroid controls reference moment.',support_bank_changed=False,material_geometry_directions_markers_changed=False,predecessor='Old narrow sin²/.08N candidate/.02N prefix stage retained as physical+spatial negative.',screen_only='NEW development load experiment. No automatic NH/space/inference/policy admission; all four layouts and eight candidates retained.')
    return d
