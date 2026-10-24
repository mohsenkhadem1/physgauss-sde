"""Prospective matched external loads and ideal static observation layout.

No field solutions, hypotheses, inferred parameters or outcomes on import. All
catalog loads require separate linear-spatial and nonlinear physical admission
before any prediction/probing study. No RGB, camera, or dynamic interpretation.
"""
import numpy as np
from experiments import campaign_surface_traction_static_v01 as shared

CENTERS=(.09,.18,.27)
PATCH_BOUNDS={.09:(.045,.135),.18:(.135,.225),.27:(.225,.315)}
SIGMA_M=20e-6
SENSOR_POINTS=np.array([[.09,.02,.08],[.09,.06,.08],
                        [.18,.02,.08],[.18,.06,.08],
                        [.27,.02,.08],[.27,.06,.08],
                        [.24,.08,.02],[.24,.08,.06]])
SENSOR_POINTS.setflags(write=False)


def surface_patch_force(n,center_x,direction,total_force_N,*,face='top',order=shared.PRODUCTION_QUADRATURE):
    """Consistent finite-area sin² traction, common physical control at all levels.

    Force amplitude is the resultant on ONE external face, in newtons.
    Moments are about the block midplane center, and generally are not zero.
    """
    if not any(abs(float(center_x)-c)<1e-14 for c in CENTERS):raise ValueError('Only predeclared centers .09,.18,.27')
    center=next(c for c in CENTERS if abs(float(center_x)-c)<1e-14)
    direction=np.asarray(direction,float)
    if direction.shape!=(3,)or not np.isfinite(direction).all()or not np.isclose(np.linalg.norm(direction),1,atol=1e-13,rtol=0)or not np.isfinite(total_force_N)or total_force_N<0 or face not in('top','bottom'):raise ValueError('Unit direction, nonnegative face resultantN and top/bottom')
    X,_,faces,_,_=shared.geometry(n);left,right=PATCH_BOUNDS[center];z=.08 if face=='top'else 0.;tol=64*np.finfo(float).eps
    vertices=X[faces];active=np.all(abs(vertices[:,:,2]-z)<tol,axis=1)&np.all(vertices[:,:,0]>=left-tol,axis=1)&np.all(vertices[:,:,0]<=right+tol,axis=1)
    ids=faces[active];vertices=X[ids];area=np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1)/2
    bary,qw=shared.triangle_rule(order);points=np.einsum('qa,fai->fqi',bary,vertices);profile=np.sin(np.pi*(points[:,:,0]-left)/.09)**2
    local=np.einsum('f,q,fq,qa->fa',area,qw,profile,bary)/.0036
    weights=np.zeros(len(X));np.add.at(weights,ids.reshape(-1),local.reshape(-1));force=total_force_N*weights[:,None]*direction
    origin=np.array([.18,.04,.04]);centroid=np.array([center,.04,z]);expected=total_force_N*direction
    return force,dict(center_x_m=center,face=face,width_m=.09,patch_area_m2=float(area.sum()),exact_profile_integral_m2=.0036,quadrature_order=order,
                      integrated_weight_mass=float(weights.sum()),net_force_N=force.sum(0),expected_net_force_N=expected,
                      moment_about_block_center_Nm=np.cross(X-origin,force).sum(0),expected_moment_about_block_center_Nm=np.cross(centroid-origin,expected),
                      weights=weights)


def load_catalog(n,*,order=shared.PRODUCTION_QUADRATURE):
    """Six prospective experiments; paired axial faces cancel eccentric moment."""
    result={}
    for center in CENTERS:
        label=f'x{round(center*1000):03d}'
        normal,_=surface_patch_force(n,center,[0,0,-1],.10,order=order)
        top,_=surface_patch_force(n,center,[1,0,0],.24,order=order)
        bottom,_=surface_patch_force(n,center,[1,0,0],.24,face='bottom',order=order)
        result['normal_'+label]=normal;result['paired_axial_'+label]=top+bottom
    return result


def observation_points():return SENSOR_POINTS.copy()


def observe(n,displacement):
    field=shared.field_queries(n,displacement,SENSOR_POINTS)
    return field.reshape(*field.shape[:-2],24)


def design():
    normal=[f'normal_x{round(c*1000):03d}'for c in CENTERS]
    axial=[f'paired_axial_x{round(c*1000):03d}'for c in CENTERS]
    return dict(schema='surface-force-study-v01',status='PROSPECTIVE_SOURCE_ONLY_NO_CASE_RESPONSES',
        physical_patch_centers_x_m=list(CENTERS),width_x_m=.09,y_interval_m=[0,.08],faces_z_m=[0,.08],
        spatial_profile='sin²(pi*(x-left)/.09), supported on center±.045; uniformy; C1zero atpatchends',
        exact_profile_integral_per_face_m2=.0036,normal_force_N=.10,paired_axial_per_face_force_N=.24,paired_axial_net_force_N=.48,
        force_convention='Known dead forces in fixed world axes on fixed reference material surface; no follower-pressure or contact model',
        normal_direction=[0.,0.,-1.],axial_direction=[1.,0.,0.],
        moment_convention='About(.18,.04,.04): normalMy=.10*(center_x-.18)Nm; pairedaxial moment0. Preserve noncentral normal moment.',
        prefix_acquired_load='normal_x180',
        held_out_loads=['normal_x090','normal_x270','paired_axial_x090','paired_axial_x180','paired_axial_x270'],
        same_observation_layout_for_every_load=True,sensor_points_m=SENSOR_POINTS.tolist(),sensor_components=['x','y','z'],sensor_channels=24,
        noise='Independent known Gaussian displacement noise20um per coordinate per reset experiment; no new noise draws on import',
        sigma_m=SIGMA_M,sensor_rationale='Six top-surface material points sample longitudinal bending/stretching shape; two y=.08 points flank the known Gaussian support in z. Intended to separate bulk E response from local supportK response; identifiability must still be checked using actual sensitivities, not assumed.',
        probing_strata=dict(normal_only=normal,paired_axial_only=axial),
        cost='Compare policies within each force family only: same patch area, resultant magnitude, reference load shape. Work depends on material and response; equal-force is not equal-work. No combined-family superiority without prospective cost normalization.',
        baselines=['fixed_center_repeat','seeded_random_within_family','local_Fisher_within_family'],
        acquisition='Posterior predictive uncertainty over predeclared held-out loads; no realized target displacement or truth in action scoring',
        physics_admission='All six catalog loads must first pass frozen spatial, quadrature, residual, nonlinear-validity and physical deformation/strain/J criteria on every predeclared physical case. Preserve failures; no load-magnitude changes or object filtering after outcomes.',
        posterior='Exact2D E/supportK reference with unchanged prior and known Gaussian surface foundation; numerical mixing/integration and predictive precision gates remain required',
        scope='Ideal tracked material-point displacements, static equilibria and known deadloads only; not RGB-D, held-out camera views, dynamics, contact, physical calibration or a demonstrated probing advantage',no_automatic_execution=True)
