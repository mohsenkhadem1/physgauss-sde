"""Bounded same-law CPU equilibria for NewtonVBD comparison; no GPU calls."""
import argparse,json,signal,time
from pathlib import Path
import numpy as np
from experiments.campaign_newton_stable_material import StableP1Block,StableP2Block
from experiments.run_static_supported_block import probes
from experiments.run_static_p2_axial_continuation import sha,dump,record
JOBS=[('P1',(4,2,2),None),('P1',(8,4,4),None),('P1',(16,8,8),None),('P2',(8,4,4),3),('P2',(16,8,8),3),('P2',(16,8,8),4)]
RUNTIME=Path('runs/operator-dev/newton-runtime-v01/env/lib/python3.12/site-packages/newton')
def prepare(path):
    sources=['experiments/campaign_newton_stable_material.py','experiments/run_newton_material_alignment.py','experiments/campaign_static_p2_block.py','experiments/campaign_static_supported_block.py','experiments/campaign_static_uniaxial_bar.py','experiments/run_static_supported_block.py','experiments/run_static_p2_axial_continuation.py','experiments/boundary_block.py','experiments/diagnose_nonlinear_physical_probes.py',str(RUNTIME/'_src/solvers/vbd/particle_vbd_kernels.py'),str(RUNTIME/'_src/sim/builder.py')]
    old=[]
    for stage in ['static-supported-block-v01','static-p2-block-v01']:
        root=Path('runs/gaussian-campaign-v01')/stage;summary=json.loads((root/'summary.json').read_text())
        for r in summary['cases']:
            if r['case_id']=='left_broad_axial' and r['completed']:
                cap=r.get('trajectory',r.get('capture'));old.append({'basis':'P1' if stage=='static-supported-block-v01' else'P2','tetrahedra':r['tetrahedra'],'order':r.get('order'),**cap})
    d={'schema':'Newton-stable-law-CPU-alignment-v0.1','status':'FROZEN_BEFORE_MATCHED_BLOCK_SOLVES','CPU_threads':2,'wall_cap_s':600,'jobs':[{'basis':b,'cells':list(c),'order':o} for b,c,o in JOBS],'physics':{'size_m':[.24,.08,.08],'young_Pa':5000.,'poisson':.35,'density':625.,'traction_N':1.,'gravity':0.,'boundary':'exactleftfaceall3axes','initial':'RESTforeverysolve','damping':0.,'force_tolerance_N':1e-11},'constitutive':{'authored_mu':'E/(2*(1+nu))','authored_lambda':'E*nu/((1+nu)*(1-2nu))','kappa':'lambda+mu','W':'mu/2*(||F||²-3)-mu*(J-1)+kappa/2*(J-1)²','P':'mu*F+(kappa*(J-1)-mu)*cof(F)','H':'FullmuI+kappacofouter+s*d²J;selfP1vertexlasttermcancels','positive_J_branch_guard':.1,'native_stable_law_permits_inversion':'Referenceguarddoesnotclaimnativebarrier'},'query_groups':{k:v.tolist() for k,v in probes().items()},'gates':{'all_fixed_query_max_Euclidean_m':20e-6,'force_balance_N':1e-8,'moment_balance_Nm':1e-8,'constraint_m':0.,'excitation':[.02,.08,.001]},'source_sha256':{f:sha(f) for f in sources},'old_logJ_captures':old,'nonclaims':['EquilibriumCPUreferenceisnottransientVBDstep','P1samemeshsolvercomparisonNOTcontinuumqualification','OldC36lawandfailuresunchanged','NoGPU/nativeaccuracy/speed/inferenceclaim']}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    dump(path,d)
def run(protocol,digest,output):
    if sha(protocol)!=digest:raise ValueError('Changed protocol')
    p=json.loads(Path(protocol).read_text())
    for f,h in p['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changedsource '+f)
    for r in p['old_logJ_captures']:
        if sha(r['path'])!=r['sha256']:raise ValueError('Changedoldcapture')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);begin=time.monotonic();groups={k:np.array(v) for k,v in p['query_groups'].items()};points=np.concatenate(list(groups.values()));d={'schema':'Newton-material-alignment-results-v0.1','completed':False,'protocol':record(protocol),'cases':[],'comparisons':[],'old_failures_unchanged':True,'GPU_or_inference_qualified':False};fields={}
    def save():d['elapsed_s']=time.monotonic()-begin;dump(out/'summary.json',d)
    signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('Frozen600sCPUbudget')));signal.alarm(600)
    try:
        for basis,cells,order in JOBS:
            start=time.monotonic();row={'basis':basis,'cells':list(cells),'tetrahedra':int(6*np.prod(cells)),'order':order,'completed':False};label=f'{basis}-tets{row["tetrahedra"]}'+('' if order is None else f'-order{order}')
            try:
                m=(StableP1Block(cells,young=5000.,gravity=0.,traction=1.) if basis=='P1' else StableP2Block(cells,young=5000.,gravity=0.,traction=1.,order=order));a=m.solve();row.update(m.diagnostics(a));row['completed']=True;row['history']=a['history']
                if basis=='P1':
                    from experiments.diagnose_nonlinear_physical_probes import interpolate
                    q=interpolate(m.rest,m.tets,a['positions'][None],points)[0][0]
                else:q=m.query(a['positions'],points)
                U,g,H,J=m.elastic(a['positions']);file=out/(label+'.npz');data={'rest_m':m.rest,'tets':m.tets,'faces':m.faces,'positions_m':a['positions'],'applied_forces_N':m.load,'elastic_forces_N':-g,'constraint_mask':m.mask,'material_probes_m':points,'probe_positions_m':q}
                if basis=='P2':data['elements']=m.elements
                np.savez_compressed(file,**data);row['capture']=record(file);row['physics_passed']=row['minimum_J']>.1 and row['maximum_constraint_error_m']==0 and row['original_free_residual_N']<=1e-11 and row['force_balance_defect_N']<=1e-8 and row['actual_moment_balance_defect_Nm']<=1e-8
                old=next((r for r in p['old_logJ_captures'] if r['basis']==basis and r['tetrahedra']==row['tetrahedra'] and r['order']==order),None)
                if old:
                    with np.load(old['path'],allow_pickle=False) as z:oldq=z['probe_positions_m']
                    row['finite_law_difference_same_mesh_max_query_m']=float(np.linalg.norm(q-oldq,axis=1).max())
                prior=(basis,row['tetrahedra']//8,order) if order!=4 else(basis,row['tetrahedra'],3)
                if prior in fields:
                    delta=np.linalg.norm(q-fields[prior],axis=1);d['comparisons'].append({'basis':basis,'fine_tetrahedra':row['tetrahedra'],'kind':'quadrature' if order==4 else'spatial','max_query_error_m':float(delta.max()),'passed20um':bool(delta.max()<=20e-6)})
                fields[(basis,row['tetrahedra'],order)]=q
            except TimeoutError as e:row['failure']={'exception':repr(e),'history':getattr(e,'history',[])};raise
            except Exception as e:row['failure']={'exception':repr(e),'history':getattr(e,'history',[])}
            finally:row['elapsed_s']=time.monotonic()-start;d['cases'].append(row);dump(out/(label+'.json'),row);save()
        d['completed']=True
    except Exception as e:d['execution_failure']=repr(e)
    finally:signal.alarm(0);save()
    return d
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a=a.parse_args()
    if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
    else:
        d=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps({k:d.get(k) for k in ['completed','elapsed_s','comparisons','execution_failure']},indent=2));raise SystemExit(0 if d['completed'] else 1)
