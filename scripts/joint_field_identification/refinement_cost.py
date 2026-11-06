"""Sparse CPU refinement cost and discrepancy, isolated from frozen v01 results.

Uses the identical element stiffness and normalized support/traction definitions.
No dense global stiffness is assembled except the tiny parity check. The existing
surface interpolation allocation remains dense and is included in peak memory.
"""
from pathlib import Path
import argparse,gc,hashlib,json,os,resource,threading,time
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu
from .mechanics import LinearGaussianBody
from . import bank

ROOT=bank.ROOT

def rss_gb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):return int(line.split()[1])/1024**2
    return 0.


def sparse_solve(body,theta,query,parity=False):
    start=time.perf_counter();th=torch.as_tensor(theta,dtype=torch.float64)[None]
    E=body.material_modulus(th)[0].numpy();values=(E[:,None,None]*body.local_unit_stiffness.numpy()).ravel()
    indices=body._assembly_index.numpy();rows=indices//body.ndof;cols=indices%body.ndof
    elastic=sparse.coo_matrix((values,(rows,cols)),shape=(body.ndof,body.ndof)).tocsc()
    support=body.footprint(th[:,2],'bottom')[0].numpy()
    stiffness=elastic+sparse.diags(np.repeat(np.exp(theta[3])*support,3),format='csc')
    top=body.footprint(th[:,4],'top')[0].numpy();side=body.footprint(th[:,4],'side')[0].numpy()
    unknown=np.stack([(top[:,None]*[0,0,-1]).ravel(),(side[:,None]*[0,-1,0]).ravel()],-1)
    probes=body.custom_probe_forces(bank.probe_specs()).numpy();rhs=np.concatenate([unknown,probes],axis=1)
    assembly_seconds=time.perf_counter()-start
    start=time.perf_counter();factor=splu(stiffness,permc_spec='MMD_AT_PLUS_A');factor_seconds=time.perf_counter()-start
    start=time.perf_counter();solution=factor.solve(rhs);solve_seconds=time.perf_counter()-start
    residual=stiffness@solution-rhs
    interpolation=body.interpolation(query).numpy();response=np.einsum('pn,ncl->pcl',interpolation,solution.reshape(body.n,3,-1))
    symmetry=stiffness-stiffness.T
    meta=dict(assembly_seconds=assembly_seconds,factor_seconds=factor_seconds,solve_12_rhs_seconds=solve_seconds,total_seconds=assembly_seconds+factor_seconds+solve_seconds,stiffness_nnz=int(stiffness.nnz),factor_nnz=int(factor.L.nnz+factor.U.nnz),relative_equilibrium_residual=float(np.linalg.norm(residual)/np.linalg.norm(rhs)),maximum_stiffness_asymmetry=float(np.max(np.abs(symmetry.data))) if symmetry.nnz else 0.,current_rss_GiB=rss_gb(),process_peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    if parity:
        dense=body.solve(th,probe_specs=bank.probe_specs());dense=np.concatenate([dense['unit_response'][0].numpy(),dense['probe_response'][0].numpy()],-1).reshape(body.ndof,-1)
        meta['dense_solution_relative_l2_difference']=float(np.linalg.norm(dense-solution)/np.linalg.norm(dense))
        meta['dense_solution_max_absolute_difference_m_per_N']=float(np.abs(dense-solution).max())
        assert meta['dense_solution_relative_l2_difference']<1e-9
    assert meta['relative_equilibrium_residual']<1e-8
    return response,meta


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--threads',type=int,default=2);ap.add_argument('--timeout',type=float,default=600);ap.add_argument('--memory-gib',type=float,default=20);args=ap.parse_args();torch.set_num_threads(args.threads)
    output=ROOT/'reports/joint-field-identification-v01/refinement-cost.json';raw=ROOT/'runs/joint-field-identification-v01/refinement-cost';raw.mkdir(exist_ok=True,parents=True)
    started=time.monotonic();done=threading.Event()
    def guard():
        while not done.wait(2):
            if time.monotonic()-started>args.timeout or rss_gb()>args.memory_gib:
                print('Resource guard: stopping refinement experiment',flush=True);os._exit(124)
    threading.Thread(target=guard,daemon=True).start()
    document=dict(scope='Cost-only isolated sparse solve refinement for two representative parameter cases, not a population accuracy certificate or new inference result.',threads=args.threads,time_limit_seconds=args.timeout,memory_limit_GiB=args.memory_gib,query_points_m=bank.points().tolist(),force_per_rhs_N=.08,rhs=['unknown_top','unknown_side']+[str(x) for x in bank.probe_specs()],source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scripts/joint_field_identification/mechanics.py':hashlib.sha256(Path(__file__).with_name('mechanics.py').read_bytes()).hexdigest()},parity=[],rows=[],status='running')
    def save():output.write_text(json.dumps(document,indent=2))
    save();base=[np.array([np.log(16000),.6,-.025,np.log(8000),.025]),np.array([np.log(8000),-.8,.025,np.log(4000),-.025])]
    for shape in ['block','step']:
        body=LinearGaussianBody(shape,(4,2,2));_,check=sparse_solve(body,base[0],bank.points(),True);document['parity'].append(dict(shape=shape,**check));del body;gc.collect();save()
        centerref=float(LinearGaussianBody(shape,(8,4,4)).material_log_center)
        previous={};first={}
        for cells in [(16,8,8),(24,12,12),(32,16,16)]:
            start=time.perf_counter();body=LinearGaussianBody(shape,cells);construction=time.perf_counter()-start
            print(shape,cells,'mesh constructed',round(construction,3),'RSS',round(rss_gb(),3),flush=True)
            for case,base_theta in enumerate(base):
                theta=base_theta.copy();theta[0]+=theta[1]*(float(body.material_log_center)-centerref)
                response,meta=sparse_solve(body,theta,bank.points());response*=.08
                row=dict(shape=shape,cells=list(cells),nodes=body.n,elements=len(body.elements),case=case,theta=theta.tolist(),construction_seconds=construction,**meta)
                row['response_vector_rms_mm']=(np.sqrt(np.mean(np.sum(response**2,axis=1),axis=0))*1000).tolist()
                row['previous_refinement_vector_rms_difference_mm']=None
                row['previous_refinement_max_point_vector_difference_mm']=None
                if case in previous:
                    delta=response-previous[case]
                    row['previous_refinement_vector_rms_difference_mm']=(np.sqrt(np.mean(np.sum(delta**2,axis=1),axis=0))*1000).tolist()
                    row['previous_refinement_max_point_vector_difference_mm']=(np.sqrt(np.sum(delta**2,axis=1)).max(0)*1000).tolist()
                    row['previous_refinement_max_absolute_coordinate_difference_mm']=(np.abs(delta).max(axis=(0,1))*1000).tolist()
                    row['previous_refinement_coordinate_rms_difference_mm']=(np.sqrt(np.mean(delta**2,axis=(0,1)))*1000).tolist()
                else:first[case]=response.copy()
                previous[case]=response.copy();np.savez(raw/f'{shape}-{cells[0]}-case{case}.npz',response_m=response,theta=theta,points=bank.points())
                document['rows'].append(row);save();print(shape,cells,case,'total solve',round(meta['total_seconds'],3),'s; peakGiB',round(meta['process_peak_rss_GiB'],3),'last delta',row['previous_refinement_vector_rms_difference_mm'],flush=True)
            del body;gc.collect()
    document['status']='completed';document['elapsed_seconds']=time.monotonic()-started;document['peak_rss_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2;save();done.set();print('Completed',document['elapsed_seconds'],flush=True)

if __name__=='__main__':main()
