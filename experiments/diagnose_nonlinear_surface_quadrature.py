"""Geometry-only meshquality and independent surface-Gaussian quadrature audit."""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from scipy.integrate import cubature
try:
    from campaign_nonlinear_excitation import geometry as old_geometry
    from campaign_compliant_gaussian_linear import candidate_patches
    import campaign_boundary_gaussians as old_boundary
    import campaign_quality_tet_refinement as quality
    from campaign_consistent_surface_gaussians import assemble_surface_gaussians
except ImportError:
    from experiments.campaign_nonlinear_excitation import geometry as old_geometry
    from experiments.campaign_compliant_gaussian_linear import candidate_patches
    from experiments import campaign_boundary_gaussians as old_boundary,campaign_quality_tet_refinement as quality
    from experiments.campaign_consistent_surface_gaussians import assemble_surface_gaussians
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def adaptive_moments(points,faces,centers,covariances):
    """Independent adaptive GK21 on ORIGINAL6triangles; no FE assembly used."""
    chol=np.linalg.cholesky(covariances);total=np.zeros((len(centers),13));checks=[]
    for face in faces:
        triangle=points[face];jac=np.linalg.norm(np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]))
        def func(uv):
            u,v=uv.T;x=(1-u[:,None])*triangle[0]+u[:,None]*((1-v[:,None])*triangle[1]+v[:,None]*triangle[2])
            features=np.concatenate([np.ones((len(x),1)),x,np.einsum('qi,qj->qij',x,x).reshape(-1,9)],axis=1);columns=[]
            for center,L in zip(centers,chol):
                white=np.linalg.solve(L,(x-center).T).T;density=np.exp(-.5*np.sum(white**2,axis=1))
                columns.append(features*(jac*u*density)[:,None])
            return np.concatenate(columns,axis=1)
        result=cubature(func,np.zeros(2),np.ones(2),rule='gk21',rtol=1e-11,atol=1e-15,max_subdivisions=128)
        checks.append(dict(status=result.status,subdivisions=int(result.subdivisions),max_raw_error_estimate=float(np.max(result.error))))
        if result.status!='converged':raise RuntimeError('Independent adaptive Gaussian reference did not converge')
        total+=result.estimate.reshape(len(centers),13)
    Z=total[:,0];return dict(normalizers_m2=Z,centroids_material_m=total[:,1:4]/Z[:,None],second_moments_material_m2=total[:,4:].reshape(-1,3,3)/Z[:,None,None],checks=checks)


def affine_energy(mean,second,A,b,K):
    return float(.5*np.sum((A.T@K@A)*second)+(A.T@K@b)@mean+.5*b@K@b)


def run(path,digest,output):
    protocol=read(ROOT/path)
    if sha(ROOT/path)!=digest:raise ValueError('Changed geometrydiagnostic protocol')
    for name,value in protocol['source_sha256'].items():
        if sha(ROOT/name)!=value:raise ValueError('Changed source '+name)
    if any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2 required')
    out=ROOT/output
    if out.exists():raise ValueError('Retain existing geometry audit')
    out.mkdir(parents=True);start=time.monotonic();patches,K,D=candidate_patches();K=2*K;centers=np.array([p.center_material_m for p in patches]);covariances=np.array([p.covariance_material_m2 for p in patches]);X0,T0,F0=old_geometry(0)
    reference=adaptive_moments(X0,F0,centers,covariances);axis=(centers[1]-centers[0]);axis/=np.linalg.norm(axis)
    continuous_moment=np.cross(reference['centroids_material_m'][0]-reference['centroids_material_m'][1],axis)
    A=np.array([[.01,.005,0.],[-.003,.02,.004],[0.,-.002,-.01]]);b=np.array([.001,-.002,.0005]);rows=[]
    for family,geometry in [('old_fixed_diagonal',old_geometry),('new_shortest_diagonal',quality.geometry)]:
        for level in range(5):
            if time.monotonic()-start>120:raise TimeoutError('Geometry-only120s cap')
            X,T,F=geometry(level);quad=old_boundary.SurfaceQuadrature.from_mesh(X,F);bank=old_boundary.patch_bank(quad,patches,2,{'candidate_boundary_hypothesis'});w=bank['normalized_node_weights'];mean=w@X;second=np.einsum('sn,ni,nj->sij',w,X,X)
            exact8=assemble_surface_gaussians(X,F,centers,covariances,order=8);exact16=assemble_surface_gaussians(X,F,centers,covariances,order=16)
            oldZ=[]
            for p in patches:
                delta=X-p.center_material_m;white=np.linalg.solve(np.linalg.cholesky(p.covariance_material_m2),delta.T).T;oldZ.append(float(quad.node_area_m2@np.exp(-.5*np.sum(white**2,axis=1))))
            quality_row=quality.shape_metrics(X,T);topology=quality.topology_certificate(X,T,X0,F0)
            if not topology['passed']:raise ValueError('Mesh topology certificate failed')
            moment=np.cross(mean[0]-mean[1],axis);patch_rows=[]
            for j in range(2):
                refenergy=affine_energy(reference['centroids_material_m'][j],reference['second_moments_material_m2'][j],A,b,K[j])
                oldenergy=affine_energy(mean[j],second[j],A,b,K[j]);consistentenergy=affine_energy(exact16.centroids_material_m[j],exact16.second_moments_material_m2[j],A,b,K[j])
                lump_second=np.einsum('n,ni,nj->ij',exact16.load_vectors[j],X,X);rowsumlumpenergy=affine_energy(exact16.centroids_material_m[j],lump_second,A,b,K[j])
                patch_rows.append(dict(patch=j,old_normalizer_relative_error=float(oldZ[j]/reference['normalizers_m2'][j]-1),old_centroid_m=mean[j].tolist(),
                    old_centroid_error_norm_m=float(np.linalg.norm(mean[j]-reference['centroids_material_m'][j])),
                    consistent16_centroid_error_norm_m=float(np.linalg.norm(exact16.centroids_material_m[j]-reference['centroids_material_m'][j])),
                    consistent8_to16_centroid_difference_m=float(np.linalg.norm(exact8.centroids_material_m[j]-exact16.centroids_material_m[j])),
                    old_secondmoment_error_frobenius_m2=float(np.linalg.norm(second[j]-reference['second_moments_material_m2'][j])),
                    affine_fixture_energy_reference_J=refenergy,old_affine_fixture_energy_J=oldenergy,old_affine_fixture_energy_relative_error=float(oldenergy/refenergy-1),
                    consistent_affine_fixture_energy_error_J=float(consistentenergy-refenergy),
                    accurate_row_sum_lumping_added_affine_fixture_energy_J=float(rowsumlumpenergy-consistentenergy),
                    consistent_Gram_rowsum_maxerror=float(np.max(abs(exact16.fixture_grams[j]@np.ones(len(X))-exact16.load_vectors[j])))))
            rows.append(dict(family=family,level=level,shape=quality_row,topology=topology,patches=patch_rows,
                rest_balanced1N_moment_Nm=moment.tolist(),rest_balanced1N_moment_norm_Nm=float(np.linalg.norm(moment)),
                rest_moment_difference_from_accurate_surface_Nm=float(np.linalg.norm(moment-continuous_moment))))
    ref={key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in reference.items()}
    report=dict(schema='mesh-quality-and-surface-Gaussian-quadrature-diagnosis-v0.1',completed=True,source_protocol=dict(path=path,sha256=digest),
        reference=ref,continuous_rest_balanced1N_moment_Nm=continuous_moment.tolist(),continuous_rest_balanced1N_moment_norm_Nm=float(np.linalg.norm(continuous_moment)),
        prescribed_center_axis=axis.tolist(),rows=rows,affine_fixture_probe=dict(A=A.tolist(),b_m=b.tolist(),K_world_N_per_m=K.tolist()),
        no_physical_trajectory_or_observation_reads=True,new_dynamical_solves=0,elapsed_s=time.monotonic()-start,
        qualifications=['Old vertex-lumped Gaussian rule is an explicitly stated approximation, not algebraically the continuous integral.','Consistent P1 fixture Gram is a NEW discretization; it does not repair/relabel frozenold dynamics.','Geometry/quad evidence alone does not attribute all dynamical mesh error.'])
    write(out/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps(dict(completed=r['completed'],elapsed_s=r['elapsed_s'])))
