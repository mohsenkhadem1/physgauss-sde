"""Numerical/observability audit for the finite spatial-field experiment.

No population forecast metrics are selected here. Fixed representative cases,
mesh refinement, exact scaling symmetry and total-covariance checks are reported.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, time, gc
import numpy as np
import torch
from .mechanics import LinearGaussianBody
from .inference import fit_posterior
from .bank import points, load_cov, NOISE, PROBE_FORCE

ROOT=Path(__file__).resolve().parents[2]


def rigid_decomposition(rest, displacement):
    X=np.asarray(rest);Y=X+np.asarray(displacement)
    xc=X-X.mean(0);yc=Y-Y.mean(0)
    U,_,Vt=np.linalg.svd(xc.T@yc)
    correction=np.eye(3);correction[-1,-1]=np.linalg.det(U@Vt)
    R=U@correction@Vt
    aligned=xc@R+Y.mean(0)
    def rms(v):return float(np.sqrt(np.mean(np.sum(v*v,axis=-1))))
    total=rms(displacement);nonrigid=rms(Y-aligned)
    return dict(total_vector_rms_mm=total*1000,rigid_vector_rms_mm=rms(aligned-X)*1000,
                nonrigid_vector_rms_mm=nonrigid*1000,nonrigid_to_total_rms=nonrigid/max(total,1e-30))


def observation_map(unit):
    H=[]
    for t in [0,3,7]:
        for p in [1,5,9,13]:
            for axis in [0,2]:
                h=np.zeros(16);h[2*t:2*t+2]=unit[p,axis];H.append(h)
    return np.stack(H)


def symmetry_and_covariance(device):
    body=LinearGaussianBody('block',(8,4,4),device=device)
    theta=np.array([np.log(16000),.6,-.025,np.log(8000),.025])
    scale=2.5;scaled=theta.copy();scaled[[0,3]]+=np.log(scale)
    with torch.no_grad():
        solved=body.solve(np.stack([theta,scaled]))
        unit=solved['unit_response'].cpu().numpy()
        pointunit=body.point_response(solved['unit_response'],points()).cpu().numpy()
        probe=solved['probe_response'].cpu().numpy()
    force=np.column_stack([.025*np.sin(np.linspace(.3,2.4,8)),.02*np.cos(np.linspace(.2,2.,8))])
    motion=np.einsum('ncl,tl->tnc',unit[0],force)
    scaled_motion=np.einsum('ncl,tl->tnc',unit[1],force*scale)
    # Three independent orthographic cameras, including previously hidden y.
    cameras=np.array([[[1,0,0],[0,0,1]],[[0,1,0],[0,0,1]],
                      [[2**-.5,2**-.5,0],[0,0,1]]])
    images=np.einsum('aki,tn i->atnk',cameras,motion)
    scaled_images=np.einsum('aki,tn i->atnk',cameras,scaled_motion)
    calibration=PROBE_FORCE*probe[0,:,:,0]
    calibration_scaled=PROBE_FORCE*probe[1,:,:,0]
    H=np.stack([observation_map(p) for p in pointunit]);y=H[0]@force.flatten()
    fixed=fit_posterior(H,y,load_cov(),NOISE**2)
    scaled_prior=fit_posterior(H,y,np.stack([load_cov(),load_cov(scale)]),NOISE**2)
    # Actual marginal force posterior includes conditional and between-state terms.
    mean=fixed.weights@fixed.means
    conditional=np.einsum('h,hij->ij',fixed.weights,fixed.covariances)
    centered=fixed.means-mean
    between=np.einsum('h,hi,hj->ij',fixed.weights,centered,centered)
    _,draws=fixed.sample(200000,seed=4421)
    empirical=np.cov(draws,rowvar=False,bias=True);total=conditional+between
    result=dict(scale_factor=scale,
        original_theta=theta.tolist(),scaled_theta=scaled.tolist(),
        displacement_max_absolute_difference_m=float(np.max(np.abs(motion-scaled_motion))),
        displacement_relative_l2=float(np.linalg.norm(motion-scaled_motion)/np.linalg.norm(motion)),
        all_three_camera_projection_max_difference_m=float(np.max(np.abs(images-scaled_images))),
        fixed_probe_response_ratio=float(np.linalg.norm(calibration_scaled)/np.linalg.norm(calibration)),
        expected_probe_ratio=1/scale,
        fixed_probe_vector_difference_rms_mm=float(np.sqrt(np.mean(np.sum((calibration_scaled-calibration)**2,axis=-1)))*1000),
        same_fixed_force_prior_hypothesis_weights=fixed.weights.tolist(),
        co_scaled_force_prior_hypothesis_weights=scaled_prior.weights.tolist(),
        interpretation='E,k,F common scaling leaves every displacement and camera projection identical. A calibrated absolute force breaks this symmetry. A fixed force-amplitude prior can favor a scale statistically; marginalization creates no new mechanical information.',
        covariance=dict(samples=200000,seed=4421,conditional_trace=float(np.trace(conditional)),
                        between_hypothesis_trace=float(np.trace(between)),total_trace=float(np.trace(total)),
                        empirical_trace=float(np.trace(empirical)),
                        relative_frobenius_error=float(np.linalg.norm(empirical-total)/np.linalg.norm(total)),
                        force_mean_relative_error=float(np.linalg.norm(draws.mean(0)-mean)/max(np.linalg.norm(mean),1e-30))))
    return result


def run(output,device):
    torch.set_num_threads(1);start=time.time();output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(scope='Small-strain linear quasistatic 3D FEM; known reference geometry/density/cameras; no gravity, inertia, unilateral contact, nonlinear truth or continuous posterior claim.',
                device=device,representative_force_N=PROBE_FORCE,mesh_sensitivity=[],
                strain_definition='Maximum absolute principal infinitesimal strain and maximum strain tensor component; node vector RMS is not coordinate RMS.',
                mesh_field_matching='Theta log-E adjusted by contrast times the mesh-dependent volume-centering change, preserving the same continuous material field across meshes.',
                scaling=symmetry_and_covariance(device))
    def save():output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    base_cases=[np.array([np.log(16000),.6,-.025,np.log(8000),.025]),
                np.array([np.log(8000),-.8,.025,np.log(4000),-.025])]
    meshes=[(8,4,4),(12,6,6),(16,8,8)]
    query=points()
    for shape in ['block','step']:
        center_reference=None;previous={};first={}
        for mesh in meshes:
            body=LinearGaussianBody(shape,mesh,device=device)
            center=float(body.material_log_center)
            if center_reference is None:center_reference=center
            for case,base in enumerate(base_cases):
                theta=base.copy();theta[0]+=theta[1]*(center-center_reference)
                began=time.time()
                with torch.no_grad():
                    out=body.solve(theta,include_operators=True)
                    response=out['unit_response'][0]
                    # Top-only, side-only, and simultaneous loads at declared calibrated force.
                    amplitudes=response.new_tensor([[PROBE_FORCE,0],[0,PROBE_FORCE],[PROBE_FORCE,PROBE_FORCE]])
                    disp=torch.einsum('ncl,kl->knc',response,amplitudes)
                    strain=body.strain(disp)
                    eig=torch.linalg.eigvalsh(strain)
                    gradient=torch.einsum('keai,eaj->keij',disp[:,body.elements,:],body.grad)
                    determinants=torch.linalg.det(torch.eye(3,dtype=disp.dtype,device=disp.device)+gradient)
                    atquery=torch.einsum('pn,knc->kpc',body.interpolation(query),disp).cpu().numpy()
                    rhs=out['unknown_forces'][0]
                    uflat=response.reshape(body.ndof,2)
                    residual=out['stiffness'][0]@uflat-rhs
                    top_sum=rhs.reshape(body.n,3,2).sum(0).cpu().numpy()
                    rest=body.points.cpu().numpy();actual=disp.cpu().numpy()
                    strains=eig.abs().amax(dim=(1,2)).cpu().numpy()
                    components=strain.abs().amax(dim=(1,2,3)).cpu().numpy()
                row=dict(shape=shape,cells=list(mesh),nodes=body.n,elements=len(body.elements),case=case,
                         theta=theta.tolist(),seconds=time.time()-began,
                         relative_equilibrium_residual=float(residual.norm()/rhs.norm()),
                         unit_load_resultants=top_sum.tolist(),
                         previous_grid_query_vector_rms_difference_mm=None,
                         first_grid_query_vector_rms_difference_mm=None,loads=[])
                for k,name in enumerate(['top_calibrated','side_calibrated','both_calibrated_each']):
                    row['loads'].append(dict(name=name,maximum_absolute_principal_strain=float(strains[k]),
                                            maximum_absolute_strain_component=float(components[k]),
                                            minimum_deformation_determinant=float(determinants[k].min()),
                                            **rigid_decomposition(rest,actual[k])))
                if case in previous:
                    row['previous_grid_query_vector_rms_difference_mm']=(np.sqrt(np.mean(np.sum((atquery-previous[case])**2,axis=-1),axis=1))*1000).tolist()
                    row['first_grid_query_vector_rms_difference_mm']=(np.sqrt(np.mean(np.sum((atquery-first[case])**2,axis=-1),axis=1))*1000).tolist()
                else:first[case]=atquery.copy()
                previous[case]=atquery.copy();report['mesh_sensitivity'].append(row);save()
                print(shape,mesh,'case',case,'strain',strains,'elapsed',time.time()-start,flush=True)
                del out,response,disp,strain,gradient,determinants,uflat,rhs,residual
                gc.collect()
            del body;gc.collect()
            if device.startswith('cuda'):torch.cuda.empty_cache()
    check=subprocess.run([sys.executable,'-m','unittest','tests.test_joint_field_inference','tests.test_joint_field_mechanics','-v'],cwd=ROOT,capture_output=True,text=True)
    report['tests']=dict(command='python -m unittest tests.test_joint_field_inference tests.test_joint_field_mechanics -v',returncode=check.returncode,output=check.stdout+check.stderr)
    sources=[Path(__file__),Path(__file__).with_name('mechanics.py'),Path(__file__).with_name('inference.py'),Path(__file__).with_name('bank.py'),ROOT/'tests/test_joint_field_inference.py',ROOT/'tests/test_joint_field_mechanics.py']
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report['seconds']=time.time()-start
    report['interpretation']='Grid differences are empirical sensitivity, not a certified continuum error bound. No automatic claim that these loads remain small-strain: inspect reported strains. The scale diagnostic is exact for the declared no-gravity linear problem only.'
    save();print('Saved',output,flush=True)
    if check.returncode:raise RuntimeError('Numerical unit tests failed; report retained')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--output',type=Path,default=ROOT/'reports/joint-field-identification-v01/qualification.json');args=ap.parse_args()
    run(args.output,args.device)
