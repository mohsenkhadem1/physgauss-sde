"""Frozen prefix-only state assimilation/relaxation diagnostic; no forecast scores."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.gaussian_boundary_flow_v02.object_backend_wide import WideGaussianTableBoundaryBackend
from scripts.external_benchmark.data import rigid_map


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    torch.set_num_threads(1)
    source=Path('runs/gaussian-boundary-flow-v03-retrieval/object-soft-task1-demo0')
    out=Path('reports/gaussian-boundary-flow-v02/cutoff-relaxation')
    if out.exists():raise FileExistsError('Immutable diagnostic directory already exists')
    plan=json.loads((source/'plan.json').read_text());seal=json.loads((source/'inference-sealed.json').read_text());complete=json.loads((source/'complete.json').read_text())
    if sha(source/'plan.json')!=seal['plan_sha256']or sha(source/'inference-sealed.npz')!=seal['selection_sha256']or sha(source/'prediction.npz')!=complete['prediction_sha256']:raise ValueError('Changed frozen selection or prediction')
    package=Path(plan['input_package'])
    if sha(package/'manifest.json')!=plan['input_package_manifest_sha256']:raise ValueError('Changed input package')
    for file,digest in plan['source_sha256'].items():
        if ('object_backend' in file or file.endswith('/fem.py'))and sha(file)!=digest:raise ValueError('Changed frozen mechanics '+file)
    data=load_package(package);frames=np.asarray(plan['prefix_frames']);ids=np.asarray(plan['point_ids']);last=int(data['train_end'])-1
    if frames[-1]!=last or len(frames)!=12 or len(ids)!=48:raise ValueError('Expected fixed12frame48point prefix')
    with np.load(source/'inference-sealed.npz')as a:z=a['selected_latents'][0];selected=int(a['selected_ids'][0])
    with np.load(source/'prediction.npz')as a:raw=a['nearest_raw'][last].copy()
    frozen={'case':data['name'],'source':str(source),'source_plan_sha256':sha(source/'plan.json'),'source_prediction_sha256':sha(source/'prediction.npz'),'source_selection_sha256':sha(source/'inference-sealed.npz'),'package':str(package),'cutoff':last,'prefix_frames':frames.tolist(),'point_ids':ids.tolist(),'selected_bank_index':selected,'new_parameter_selection':False,'maximum_newton_iterations_per_start':30,'solver_tolerance':1e-7,'assimilation_svd_rcond':1e-10,'inversion_rejection_minJ':.03,'scope':'Same-cutoff observation correction and unchanged static equilibrium relaxation; no suffix targets or scores','script_sha256':sha(__file__)}
    out.mkdir(parents=True);write(out/'plan.json',frozen)
    b=WideGaussianTableBoundaryBackend(data,'cpu');m=b.model;theta=b.bind(z);A=data['observation_matrix'][ids];initial=data['observed_initial'][ids];target=data['observed'][last,ids];valid=data['observed_mask'][last,ids]&np.isfinite(target).all(1)
    def observed(world):return initial+A@(world-data['nodes'])
    def stats(world):
        x=m.t((world-m.origin)/m.scale);pred=observed(world);R,t=rigid_map(pred[valid],target[valid]);J=m.min_jacobian(x)
        d={'coordinate_rmse_mm':float(np.sqrt(np.mean((pred[valid]-target[valid])**2))*1000),'aligned_shape_rmse_mm':float(np.sqrt(np.mean((pred[valid]@R+t-target[valid])**2))*1000),'minJ':J,'observation_correction_is_mechanical':False}
        try:
            E,r,_=m.evaluate(x,theta,last);d.update(energy_dimensionless=float(E),force_residual_norm=float(torch.linalg.vector_norm(r)),net_residual=r.sum(0).tolist())
        except(ValueError,RuntimeError)as e:d['evaluation_failure']=str(e)
        return d
    R,t=rigid_map(observed(raw)[valid],target[valid]);rigid=raw@R+t
    delta,_,rank,s=np.linalg.lstsq(A[valid],target[valid]-observed(rigid)[valid],rcond=1e-10);corrected=rigid+delta
    stages={'raw':stats(raw),'rigid_tracking_correction':stats(rigid),'affine_tracking_correction':stats(corrected)}
    results={'plan_sha256':sha(out/'plan.json'),'stages':stages,'observation_rank':int(rank),'observation_singular_values':s.tolist(),'maximum_geometric_correction_mm':float(np.linalg.norm(corrected-raw,axis=1).max()*1000),'relaxations':{},'future_target_access':False}
    relaxed={}
    for name,world in [('raw_start',raw),('corrected_start',corrected)]:
        try:
            x=m.t((world-m.origin)/m.scale)
            if m.min_jacobian(x)<=.03:raise ValueError('Geometric correction rejected: element J<=.03; no clipping or strain repair')
            with torch.no_grad():q,_,diagnostic=m.solve(theta,last,x,tolerance=1e-7,maxiter=30)
            end=q.detach().numpy()*m.scale+m.origin;relaxed[name]=end
            results['relaxations'][name]={'completed':True,'solver':diagnostic,'state':stats(end),'distance_from_original_equilibrium_mm':float(np.sqrt(np.mean((end-raw)**2))*1000)}
        except(ValueError,RuntimeError)as e:results['relaxations'][name]={'completed':False,'failure':str(e)}
    if len(relaxed)==2:results['relaxed_start_difference_mm']=float(np.sqrt(np.mean((relaxed['raw_start']-relaxed['corrected_start'])**2))*1000)
    write(out/'results.json',results)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
