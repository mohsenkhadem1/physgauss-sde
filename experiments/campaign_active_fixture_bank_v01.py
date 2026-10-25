"""Development-only 3D fixture-layout bank; no automatic mechanics or draws.

Public both-end hard clamps are separate from one unknown compliant bottom
patch. Consistent signed T6 coefficients are not masses. Every category has
unit integrated Gaussian weight, so K has the same N/m meaning. Controls never
accept E,K, layout or observations. Future nonlinear/spatial admission is new.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from scipy.special import erf
from experiments import campaign_surface_p2_static_v02 as core
from experiments.campaign_surface_p2_inference_v01 import LabelledModal

LAYOUTS=tuple(dict(id=f'bottom-x{round(x*1000):03d}-y{round(y*1000):03d}',center_m=[x,y,0.])for x in(.09,.27)for y in(.02,.06))
STD_M=(.025,.015)
PATCH_WIDTH_M=(.09,.04)
FORCE_N=.08
PREFIX_FORCE_N=.02
SIGMA_M=20e-6
PREFIX_POINTS=np.array([[.1575,.033,.08],[.18,.047,.08],[.2025,.038,.08],[.18,.04,.08]])
TARGET_POINTS=np.array([[x,y,.08]for x in(.09,.18,.27)for y in(.015,.065)]+[[.135,0.,.04],[.225,.08,.04]])
ORIGIN_M=np.array([.18,.04,.04])


def layout_spec(identity):
    for spec in LAYOUTS:
        if spec['id']==identity:return dict(spec,center_m=list(spec['center_m']),std_m=list(STD_M),surface='bottom',prior_probability=.25)
    raise ValueError('One of four declared bottom fixture categories')


def gaussian_integrals(spec):
    c=np.asarray(spec['center_m'][:2]);s=np.asarray(spec['std_m']);upper=np.array([.36,.08])
    I=s*np.sqrt(np.pi/2)*(erf((upper-c)/(np.sqrt(2)*s))-erf(-c/(np.sqrt(2)*s)))
    centroid=c+s*s*(np.exp(-.5*(c/s)**2)-np.exp(-.5*((upper-c)/s)**2))/I
    return float(np.prod(I)),np.r_[centroid,0.]


def consistent_fixture(X,faces,identity,*,order=16):
    spec=layout_spec(identity);normalizer,centroid=gaussian_integrals(spec)
    b,w=core.triangle_rule(order);N=core.triangle_shapes(b);selected=faces[np.all(abs(X[faces[:,:3],2])<1e-14,axis=1)]
    vertices=X[selected[:,:3]];area=np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1)/2
    q=np.einsum('qa,fai->fqi',b,vertices);c=np.asarray(spec['center_m'][:2]);s=np.asarray(spec['std_m'])
    g=np.exp(-.5*np.sum(((q[:,:,:2]-c)/s)**2,axis=2));qw=area[:,None]*w[None,:]*g/normalizer
    values=np.einsum('fq,qa,qb->fab',qw,N,N)
    rows=np.repeat(selected,6,axis=1).ravel();cols=np.tile(selected,(1,6)).ravel()
    W=sparse.coo_matrix((values.ravel(),(rows,cols)),shape=(len(X),len(X))).tocsr();W.sum_duplicates()
    weights=np.asarray(W.sum(axis=1)).ravel();mass=float(W.sum());first=X.T@weights
    return W,dict(layout=identity,order=order,exact_normalization_m2=normalizer,integrated_mass=mass,normalization_error=abs(mass-1),canonical_centroid_m=centroid.tolist(),integrated_first_moment_m=first.tolist(),first_moment_error_m=float(np.max(abs(first-centroid))),PSD_by_positive_quadrature=True,signed_T6_entries_allowed=True,negative_entries=int(np.count_nonzero(W.data<0)),renormalized=False)


def make_forward(n,identity,*,bulk_order=24,surface_order=16):
    # Fresh instance only. Never mutate a frozen module or cached topology.
    f=core.SparseStaticForward(n,quadrature_order=bulk_order)
    W,info=consistent_fixture(f.X,f.faces,identity,order=surface_order)
    f.W_scalar=W;f.W=sparse.kron(W,sparse.eye(3),format='csr');f.weights=np.asarray(W.sum(axis=1)).ravel()
    f.W_free=f.W[f.free][:,f.free].tocsc();f.support_quadrature=info
    return f


def load_specs():
    result=[dict(id='prefix-center-down',center_m=[.18,.04,.08],direction=[0.,0.,-1.],force_N=PREFIX_FORCE_N)]
    for x in(.09,.27):
        for y in(.02,.06):
            for sign,name in((1.,'plusY'),(-1.,'minusY')):
                result.append(dict(id=f'probe-x{round(x*1000):03d}-y{round(y*1000):03d}-{name}',center_m=[x,y,.08],direction=(np.array([0.,sign,-1.])/np.sqrt(2)).tolist(),force_N=FORCE_N))
    result += [dict(id='held-offaxis-down',center_m=[.135,.055,.08],direction=[0.,0.,-1.],force_N=FORCE_N),dict(id='held-offaxis-axial',center_m=[.225,.025,.08],direction=[1.,0.,0.],force_N=FORCE_N)]
    return result


def surface_patch(n,spec,*,order=16):
    X,_,faces,_,_=core.geometry(n);c=np.asarray(spec['center_m'],float);d=np.asarray(spec['direction'],float);F=float(spec['force_N']);width=np.asarray(PATCH_WIDTH_M)
    if c.shape!=(3,)or d.shape!=(3,)or not np.isfinite(c).all()or not np.isfinite(d).all()or not np.isclose(np.linalg.norm(d),1,atol=1e-13,rtol=0)or not np.isfinite(F)or F<0 or c[2]!=.08 or np.any(c[:2]-width/2 < -1e-14)or np.any(c[:2]+width/2 > np.array([.36,.08])+1e-14):raise ValueError('Known top rectangular patch, unit direction and finite nonnegative force')
    b,w=core.triangle_rule(order);weights=np.zeros(len(X));integral=float(np.prod(width)/4);pieces=0
    for ids in faces:
        vertices=X[ids[:3]]
        if not np.all(abs(vertices[:,2]-.08)<1e-14):continue
        poly=np.eye(3)
        for axis in(0,1):
            for bound,greater in((c[axis]-width[axis]/2,True),(c[axis]+width[axis]/2,False)):
                if len(poly)<3:break
                poly=core._clip(poly,vertices[:,axis],bound,greater)
        if len(poly)<3:continue
        for j in range(1,len(poly)-1):
            sub=np.array([poly[0],poly[j],poly[j+1]]);xyz=sub@vertices;area=np.linalg.norm(np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0]))/2
            if area<1e-24:continue
            parent=b@sub;q=parent@vertices;profile=np.prod(np.sin(np.pi*(q[:,:2]-(c[:2]-width/2))/width)**2,axis=1)
            local=area*np.einsum('q,q,qa->a',w,profile,core.triangle_shapes(parent))/integral;np.add.at(weights,ids,local);pieces+=1
    force=F*weights[:,None]*d;moment=np.cross(X-ORIGIN_M,force).sum(0);expected=np.cross(c-ORIGIN_M,F*d)
    return force,dict(id=spec['id'],order=order,integrated_weight_mass=float(weights.sum()),normalization_error=abs(float(weights.sum())-1),net_force_N=force.sum(0).tolist(),net_force_error_N=float(np.max(abs(force.sum(0)-F*d))),moment_Nm=moment.tolist(),analytic_moment_Nm=expected.tolist(),moment_error_Nm=float(np.max(abs(moment-expected))),equal_candidate_continuum_traction_norm_N=F,normalization_integral_m2=integral,subtriangles=pieces,signed_T6_coefficients=True,world_dead_load=True)


def load_catalog(n,*,order=16):
    loads={};info={}
    for spec in load_specs():loads[spec['id']],info[spec['id']]=surface_patch(n,spec,order=order)
    return loads,info


def measurement_specs():
    specs=[]
    for i,p in enumerate(PREFIX_POINTS):specs.append(dict(id=f'prefix-marker-{i}',role='prefix',load='prefix-center-down',point_m=p.tolist(),projection=[0.,0.,1.]))
    for load in load_specs()[1:9]:specs.append(dict(id=load['id']+'-marker',role='candidate',load=load['id'],point_m=load['center_m'],projection=load['direction']))
    for load in load_specs()[9:]:
        for i,p in enumerate(TARGET_POINTS):
            for a,name in enumerate('xyz'):specs.append(dict(id=load['id']+f'-point{i}-{name}',role='target',load=load['id'],point_m=p.tolist(),projection=np.eye(3)[a].tolist()))
    return specs


def measurement_points():return np.array(list(dict.fromkeys(tuple(s['point_m'])for s in measurement_specs())))


def measurement_response(n,fields,labels=None):
    specs={s['id']:s for s in measurement_specs()};labels=list(specs)if labels is None else list(labels);columns=[]
    for label in labels:
        s=specs[label];u=np.asarray(fields[s['load']],float)
        if u.ndim!=3:raise ValueError('Explicit [batch,Nnodes,3] displacement fields')
        value=core.field_queries(n,u,np.array([s['point_m']]))[:,0]@np.asarray(s['projection'])
        columns.append(value)
    return np.stack(columns,axis=1)[:,:,None]


def make_measurement_modal(forward,loads):
    points=measurement_points();base=LabelledModal.from_forward(forward,loads=loads,points=points,condensed=True,query_matrix=core.point_query_matrix);specs=measurement_specs();rows=[]
    for s in specs:
        i=next(i for i,p in enumerate(points)if np.array_equal(p,s['point_m']));j=base.labels.index(s['load'])
        rows.append(np.asarray(s['projection'])@base.coefficients[3*i:3*i+3,j,:])
    modal=SimpleNamespace(eigenvalues=base.modal.eigenvalues.copy(),coefficients=np.asarray(rows),output_shape=(1,len(specs)),diagnostics=dict(base.modal.diagnostics,measurement_selection='Exact public point/projection map; all modes retained'))
    return LabelledModal(modal,tuple(s['id']for s in specs))


def nonrigid_rms(forward,fields):
    bary,w=core.volume_rule(24);N=core.shapes(bary);x=np.einsum('qa,eai->eqi',N,forward.X[forward.elements]).reshape(-1,3);weights=(forward.volumes[:,None]*w).ravel();weights=weights/weights.sum();center=weights@x;xc=x-center
    u=np.einsum('qa,leai->leqi',N,fields[:,forward.elements]).reshape(len(fields),-1,3);answer=[]
    for displacement in u:
        y=x+displacement;yc=y-weights@y;A,_,B=np.linalg.svd(xc.T@(weights[:,None]*yc));R=A@np.diag([1.,1.,np.linalg.det(A@B)])@B
        residual=yc-xc@R;answer.append(float(np.sqrt(np.sum(weights[:,None]*residual**2))))
    return np.asarray(answer)


def design():
    return dict(schema='active-fixture-bank-development-v01',layouts=[layout_spec(s['id'])for s in LAYOUTS],hard_boundary='Public exact BOTH end-face clamps, separate from one unknown compliant bottom patch',material=dict(E_Pa=[2000.,8000.],K_N_m=[5.,30.],poisson=.35,conditional_prior='Independent normalized loguniform E,K, identical per layout'),loads=load_specs(),patch_width_m=list(PATCH_WIDTH_M),measurements=measurement_specs(),sigma_m=SIGMA_M,probe_semantics='One-stage choice of known reset-static load, not a physical transient or accumulated poke history',gates=dict(core.GATES),screen_only='n1/n2 linear development; no NH/spatial/continuous-inference/calibration/active-win admission',no_automatic_execution=True)
