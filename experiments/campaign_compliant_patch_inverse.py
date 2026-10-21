"""Tiny manufactured inverse with unknown MATERIAL-FRAME compliant fixture.

Four unknowns: E, fixture center coordinate, Gaussian width, finite stiffness.
All15 nodal DOFs are free; known loads never depend on candidate parameters.
Reference posterior is explicit low-order prior quadrature, not a claim of
converged/calibrated inverse physics or Isaac/generalization performance.
"""
from __future__ import annotations
import argparse,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import torch
try:
    from campaign_compliant_gaussian_linear import GaussianFixtureMatrix,tiny_geometry
    from campaign_linear_fem import assemble_tetrahedra
    from campaign_boundary_gaussians import SurfaceQuadrature,GaussianPatch,normalized_surface_weights
    from campaign_material_gaussians import MaterialGaussians
except ImportError:
    from experiments.campaign_compliant_gaussian_linear import GaussianFixtureMatrix,tiny_geometry
    from experiments.campaign_linear_fem import assemble_tetrahedra
    from experiments.campaign_boundary_gaussians import SurfaceQuadrature,GaussianPatch,normalized_surface_weights
    from experiments.campaign_material_gaussians import MaterialGaussians

ROOT=Path(__file__).resolve().parents[1]
PARAMETERS=('E_Pa','center_x_material_m','width_m','fixture_stiffness_N_m')
LOWER=np.array([np.log(2000.),.005,np.log(.018),np.log(5.)])
UPPER=np.array([np.log(8000.),.065,np.log(.05),np.log(30.)])
TRUTH=np.array([4300.,.029,.031,16.5])
H=1/120
OBS_FRAMES=np.array([12,24])
OBS_NODES=np.array([1,4])
SIGMA=20e-6


def prior_grid(order):
    """Tensor Gauss-Legendre for independent UNIFORM INTRINSIC priors.

    Intrinsic coordinates are logE, center, logwidth, logK. Nonuniform product
    weights integrate their normalized prior, never a uniform particle count.
    """
    if type(order) is not int or not 2<=order<=5:raise ValueError('Order2..5 only')
    nodes,weights=np.polynomial.legendre.leggauss(order)
    rows=np.array(list(itertools.product(range(order),repeat=4)))
    intrinsic=LOWER+(nodes[rows]+1)*(UPPER-LOWER)/2
    theta=intrinsic.copy();theta[:,[0,2,3]]=np.exp(theta[:,[0,2,3]])
    mass=np.prod((weights/2)[rows],axis=1)
    if abs(mass.sum()-1)>64*np.finfo(float).eps:raise ValueError('Prior quadrature mass mismatch')
    return theta,mass


def known_force(program,*,Q=None):
    """ONE fixed120Hz/.5s material-distributed control, shared across theta."""
    if program not in ('A','B'):raise ValueError('Known A or reset B program required')
    X,_,faces=tiny_geometry()
    weights,_=normalized_surface_weights(X,faces,np.array([[.09,.075,.075]]),
        np.eye(3)[None]*.03**2,np.array([True]))
    t=(np.arange(60)+.5)/120
    phase=np.clip((t-.025)/.30,0,1)
    amplitude=.02*np.sin(np.pi*phase)**2
    amplitude[(t<=.025)|(t>=.325)]=0.
    direction=np.array([.6,1.,.3]) if program=='A' else np.array([-.2,.4,1.])
    direction/=np.linalg.norm(direction)
    if Q is not None:direction=direction@np.asarray(Q).T
    return amplitude[:,None,None]*weights[0,None,:,None]*direction


def matrices(theta,*,Q=None,shift=None):
    theta=np.asarray(theta,dtype=np.float64)
    if (theta.ndim!=2 or theta.shape[1]!=4 or not len(theta) or not np.isfinite(theta).all()
            or (theta[:,[0,2,3]]<=0).any()):raise ValueError('Finite physical theta[B,4] required')
    Q=np.eye(3) if Q is None else np.asarray(Q,dtype=float)
    shift=np.zeros(3) if shift is None else np.asarray(shift,dtype=float)
    if Q.shape!=(3,3) or not np.allclose(Q.T@Q,np.eye(3),atol=1e-13,rtol=0) or np.linalg.det(Q)<0:
        raise ValueError('Proper passive frame rotation required')
    X0,tets,faces=tiny_geometry();X=X0@Q.T+shift
    unit,mass,_,_=assemble_tetrahedra(X,tets,1.,.35,625.)
    quadrature=SurfaceQuadrature.from_mesh(X,faces);cache={};fixture_K=[];fixture_C=[];weights=[]
    for _,center,width,k in theta:
        key=(float(center),float(width))
        if key not in cache:
            b=np.array([center,.02,0.])@Q.T+shift
            patches=[GaussianPatch(b,Q@(np.eye(3)*width**2)@Q.T,'candidate_boundary_hypothesis'),
                GaussianPatch(X[0],np.eye(3)*.03**2,'candidate_boundary_hypothesis',active=False)]
            cache[key]=GaussianFixtureMatrix.from_patches(quadrature,patches,
                [np.eye(3),np.zeros((3,3))],[np.eye(3)*.05,np.zeros((3,3))])
        fixture=cache[key];fixture_K.append(fixture.K*k);fixture_C.append(fixture.C);weights.append(fixture.weights[0])
    elastic=torch.tensor(theta[:,0,None,None])*unit
    return dict(X=X,tets=tets,mass=mass,elastic_K=elastic,
        fixture_K=torch.stack(fixture_K),C=torch.stack(fixture_C),weights=torch.stack(weights))


def forward(theta,program='A',*,Q=None,shift=None):
    """Batched CPU float64 implicit Euler; same equations as unchanged bridge."""
    data=matrices(theta,Q=Q,shift=shift);mass=data['mass'];K=data['elastic_K']+data['fixture_K'];C=data['C']
    L=torch.diag(mass/H**2)[None]+C/H+K
    chol=torch.linalg.cholesky(L) # physical dynamic SPD; no regularizer/pins
    u=torch.zeros((len(K),15),dtype=torch.float64);v=torch.zeros_like(u);states=[u.reshape(-1,5,3).clone()]
    force=known_force(program,Q=Q).reshape(60,15)
    maximum_residual=0.
    for F in force:
        old_u,old_v=u,v
        rhs=torch.tensor(F)[None]+mass*(old_u/H**2+old_v/H)+torch.einsum('bij,bj->bi',C,old_u)/H
        u=torch.cholesky_solve(rhs[:,:,None],chol)[:,:,0];v=(u-old_u)/H
        residual=mass*(v-old_v)/H+torch.einsum('bij,bj->bi',K,u)+torch.einsum('bij,bj->bi',C,v)-torch.tensor(F)[None]
        maximum_residual=max(maximum_residual,float(residual.abs().max()))
        states.append(u.reshape(-1,5,3).clone())
    return torch.stack(states,1).numpy(),dict(maximum_physical_force_residual_N=maximum_residual,
        maximum_weight_sum_error=float((data['weights'].sum(1)-1).abs().max()),
        all_nodal_DOFs_free=True,hard_constraints=0,CPU_float64=True)


def selected_observations(u):
    return np.asarray(u)[:,OBS_FRAMES][:,:,OBS_NODES].reshape(len(u),-1)


def condition(predicted,observation,prior_mass,*,sigma=SIGMA,observed_mask=None):
    x,y=np.asarray(predicted,dtype=float),np.asarray(observation,dtype=float)
    mass=np.asarray(prior_mass,dtype=float)
    mask=np.ones(len(y),bool) if observed_mask is None else np.asarray(observed_mask)
    if (x.ndim!=2 or y.shape!=(x.shape[1],) or mask.shape!=y.shape or mask.dtype!=bool
            or mass.shape!=(len(x),) or (mass<=0).any() or abs(mass.sum()-1)>64*np.finfo(float).eps
            or not np.isfinite(x).all() or not np.isfinite(y[mask]).all() or not np.isfinite(sigma) or sigma<=0):
        raise ValueError('Finite candidate observations, normalized positive prior and explicit noise required')
    ll=-.5*np.sum(((x[:,mask]-y[mask])/sigma)**2,axis=1)-mask.sum()*np.log(sigma*np.sqrt(2*np.pi))
    logjoint=np.log(mass)+ll;maximum=logjoint.max();normalizer=maximum+np.log(np.exp(logjoint-maximum).sum())
    weights=np.exp(logjoint-normalizer)
    return weights,dict(log_evidence=float(normalizer),ESS=float(1/np.square(weights).sum()),
        maximum_weight=float(weights.max()),observed_channels=int(mask.sum()))


def moments(u,weights):
    anchor=u[0];delta=u-anchor;mean_delta=np.einsum('b,btnc->tnc',weights,delta)
    centered=delta-mean_delta
    return anchor+mean_delta,np.sqrt(np.einsum('b,btnc->tnc',weights,centered**2))


def parameter_summary(theta,weights):
    result=[]
    for j,name in enumerate(PARAMETERS):
        order=np.argsort(theta[:,j],kind='stable');x=theta[order,j];w=weights[order]
        mean=float(weights@theta[:,j]);cdf=np.cumsum(w);cdf[-1]=1.
        q=x[np.searchsorted(cdf,[.05,.5,.95],side='left')]
        result.append(dict(name=name,mean=mean,std=float(np.sqrt(weights@((theta[:,j]-mean)**2))),
            q05=float(q[0]),median=float(q[1]),q95=float(q[2])))
    return result


def design():
    return dict(schema='manufactured-compliant-patch-inverse-v0.1',status='FIXED_BEFORE_PRIOR_SWEEP',
        grid_orders=[3,5],prior_intrinsic_order=['logE','center_x','logwidth','logfixtureK'],
        prior_lower=LOWER.tolist(),prior_upper=UPPER.tolist(),truth_physical=TRUTH.tolist(),
        truth_off_grid=True,geometry='Two positively wound tets/5nodes from unchanged tiny_geometry',
        fixture='One active material-frame Gaussian, center[x,.02,0] on reference surface; isotropic finiteK,knownD=.05Ns/m; second slot inactive',
        force='Known .02N smooth pulse on one fixed material Gaussian; physically same across all hypotheses',
        density_kg_m3=625.,poissons_ratio=.35,body_drag=0.,seconds=.5,hz=120,frames=61,
        observed_frames=OBS_FRAMES.tolist(),observed_nodes=OBS_NODES.tolist(),observed_XYZ=True,
        observation_noise_std_m=SIGMA,observation_seed=2026121000,
        targets=dict(A_future_frames=list(range(25,61)),B_reset_frames=list(range(61))),
        method='Tensor Gauss-Legendre quadrature of normalized independent intrinsic priors; discrete Gaussian likelihood; no fit/adaptive grid/held-score tuning',
        scope='Manufactured finite small-strain verification only; no physical/Isaac/generalization/calibration claim')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare(path):
    path=Path(path)
    if path.exists():raise ValueError('Fresh synthetic declaration required')
    sources={name:sha(ROOT/name) for name in ['experiments/campaign_compliant_patch_inverse.py',
        'experiments/campaign_compliant_gaussian_linear.py','experiments/campaign_boundary_gaussians.py',
        'experiments/campaign_material_gaussians.py','experiments/campaign_linear_fem.py']}
    value=dict(**design(),source_sha256=sources);path.parent.mkdir(parents=True,exist_ok=True);write(path,value)
    return dict(path=str(path),sha256=sha(path))


def run(design_path,design_sha,output):
    if sha(design_path)!=design_sha:raise ValueError('Synthetic design changed')
    specification=json.loads(Path(design_path).read_text())
    if {k:specification[k] for k in design()}!=design():raise ValueError('Fixed design required')
    for name,digest in specification['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed')
    out=Path(output)
    if out.exists():raise ValueError('Fresh synthetic output; preserve failures')
    out.mkdir(parents=True);torch.set_num_threads(2);started=time.monotonic()
    write(out/'protocol.json',dict(design=dict(path=str(design_path),sha256=design_sha),source_sha256=specification['source_sha256']))
    try:
        truth_a,truth_check=forward(TRUTH[None],'A');truth_b,_=forward(TRUTH[None],'B')
        exact=selected_observations(truth_a)[0]
        observation=exact+np.random.default_rng(2026121000).normal(0,SIGMA,len(exact))
        rows=[];means={}
        for order in specification['grid_orders']:
            theta,mass=prior_grid(order)
            if np.any(np.min(np.abs(theta-TRUTH),axis=0)<1e-12):raise ValueError('Truth must stay off every grid coordinate')
            a,checks=forward(theta,'A');b,_=forward(theta,'B');predicted=selected_observations(a)
            weights,inference=condition(predicted,observation,mass)
            no_data,_=condition(predicted,observation,mass,observed_mask=np.zeros(len(observation),bool))
            targets={};payload={}
            for label,u,truth,frames in [('A',a,truth_a[0],np.arange(25,61)),('B',b,truth_b[0],np.arange(61))]:
                prior_mean,prior_std=moments(u,mass);mean,std=moments(u,weights)
                targets[label]=dict(prior_mean_RMSE_m=float(np.sqrt(np.mean((prior_mean[frames]-truth[frames])**2))),
                    posterior_mean_RMSE_m=float(np.sqrt(np.mean((mean[frames]-truth[frames])**2))),
                    prior_latent_std_RMS_m=float(np.sqrt(np.mean(prior_std[frames]**2))),
                    posterior_latent_std_RMS_m=float(np.sqrt(np.mean(std[frames]**2))))
                payload.update({label+'_mean_displacement_m':mean,label+'_latent_std_m':std})
                means[(order,label)]=mean
            X,tets,_=tiny_geometry();kernels=MaterialGaussians.from_tetrahedra(torch.tensor(X),torch.tensor(tets),625.)
            transported=kernels.transport(torch.tensor(X+a));minimum_J=float(transported.jacobians.min())
            if minimum_J<=0:raise ValueError('Entire candidate sweep fails on inverted geometry; never filter')
            np.savez_compressed(out/f'posterior-order{order}.npz',theta=theta,prior_mass=mass,posterior_weights=weights,
                observation=observation,exact_observation=exact,**payload)
            rows.append(dict(order=order,particles=len(theta),inference=inference,targets=targets,
                prior_parameters=parameter_summary(theta,mass),posterior_parameters=parameter_summary(theta,weights),
                posterior_physical_parameter_covariance=((theta-weights@theta).T@(weights[:,None]*(theta-weights@theta))).tolist(),
                no_observation_prior_weight_max_error=float(np.abs(no_data-mass).max()),minimum_candidate_J=minimum_J,
                maximum_mass_error_kg=float((transported.volumes*transported.bulk_density-kernels.masses).abs().max()),
                mechanics=checks,physical_shape_covariance_is_not_posterior_uncertainty=True))
        result=dict(completed=True,manufactured_only=True,unknown_finite_gaussian_fixture_not_exact_pin=True,
            truth_physical=TRUTH.tolist(),truth_off_grid=True,levels=rows,
            grid_refinement_predictive_mean_RMSE_m={label:float(np.sqrt(np.mean((means[(3,label)]-means[(5,label)])**2))) for label in ('A','B')},
            truth_mechanics=truth_check,elapsed_s=time.monotonic()-started,
            limitations=['Same finite linear model generates truth and hypotheses; no calibrated physics claim',
                'Only625primary quadrature points; levelcomparison is not posterior convergence proof',
                'One surface-line center/isotropic width/scalarcompliance, fixeddamping/materialnu/rho/knownload',
                'No nonlinearcontact/rigidhardware/RGB/Isaac/currentcohort/generalization result'],raw_arrays_LOCAL=True)
        for name,digest in specification['source_sha256'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed during sweep')
        write(out/'summary.json',result);return result
    except BaseException as error:
        write(out/'failure.json',dict(completed=False,error=type(error).__name__,message=str(error),no_filter_or_retry=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True)
    a=sub.add_parser('run');a.add_argument('--design',required=True);a.add_argument('--design-sha256',required=True);a.add_argument('--output',required=True)
    a=p.parse_args()
    value=prepare(a.output) if a.mode=='prepare' else run(a.design,a.design_sha256,a.output)
    print(json.dumps(value if a.mode=='prepare' else dict(completed=value['completed'],elapsed_s=value['elapsed_s'])))
