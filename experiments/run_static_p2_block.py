"""Prospective bounded P2 block spatial/quadrature qualification, no P1 edits."""
import argparse,hashlib,json,signal,time
from pathlib import Path
import numpy as np
import torch
from experiments.campaign_static_p2_block import QuadraticBlock
from experiments.run_static_supported_block import probes
MESHES=((4,2,2),(8,4,4),(16,8,8))
CASES=({'id':'left_broad_axial','young':5000.,'gravity':0.,'traction':1.},{'id':'left_gravity_cantilever','young':300000.,'gravity':9.81,'traction':0.})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def prepare(path):
    sources=['experiments/campaign_static_p2_block.py','experiments/run_static_p2_block.py','experiments/campaign_static_supported_block.py','experiments/run_static_supported_block.py','experiments/campaign_static_uniaxial_bar.py','experiments/boundary_block.py','experiments/diagnose_nonlinear_physical_probes.py']
    d={'schema':'P2-static-block-protocol-v0.1','status':'FROZEN_BEFORE_BENCHMARK','CPU_threads':2,'hard_wall_cap_s':600,'estimated_memory_cap_bytes':8*1024**3,'cases':list(CASES),'mesh_cells':MESHES,'tetrahedra':[96,768,6144],'spatial_quadrature_order':3,'finest_quadrature_parity_order':4,'quadrature_points':[27,64],'quadrature_parity_requires_finest6144':True,'source_sha256':{p:sha(p) for p in sources},'probe_groups_material_m':{k:v.tolist() for k,v in probes().items()},'loads':'Identical P1 continuum rho625 gravity9.81 and uniform1N right reference traction; P2 consistent integrals (signed vertex body coefficients, not negative physical masses)','supports':'All displacement components exact on every P2 node at x=0','geometry':'Shared global edge midpoints; affine tetrahedron rest mapping','solver':'Unmodified NH tangent, exact free-DOF elimination, positiveJ/Armijo staticNewton40; fixed1e-11N original residual; everycase fromREST; no repair/retry/warmstart','Jacobians':'Guard>.1 at volume quadrature and10elementnodes. Also report global sufficient bound (1-maxvertex||F-I||2)^3 when max<1, by affineF/convex spectral norm; no certificate otherwise','spatial_and_quadrature_gates':{'max_Euclidean_fixed_query_error_m':20e-6,'same_groups_including_edges_corners':True,'noise_reference_sigma_m':200e-6,'comparison_order':'P2 96→768→6144 at order3, then finest order3→4 independently'},'physics_gates':{'minimum_sampled_J':.1,'exact_constraint_error_m':0.,'original_force_residual':1e-11,'global_force_balance_N':1e-8,'actual_moment_balance_Nm':1e-8},'excitation_goal':{'max_sampled_absolute_principal_strain':[.02,.08],'rigid_removed_volume_coordinate_RMS_min_m':.001,'not_replaced_by_volume99thpercentile':True},'limits':['P1failuresunchanged','Staticfixedprobesonly;nodynamic/native/contact/inferenceclaim','Allfailed/unlaunchedcellsretained;hardcapstopsnewcases']}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    dump(path,d)

def run(protocol,digest,output):
    if sha(protocol)!=digest:raise ValueError('Protocol changed')
    p=json.loads(Path(protocol).read_text())
    for f,h in p['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Source changed '+f)
    torch.set_num_threads(2);out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();groups={k:np.array(v) for k,v in p['probe_groups_material_m'].items()};points=np.concatenate(list(groups.values()))
    jobs=[(c,cells,3) for cells in MESHES for c in CASES]+[(c,MESHES[-1],4) for c in CASES]
    d={'schema':'P2-static-block-results-v0.1','completed':False,'protocol':{'path':str(protocol),'sha256':digest},'cases':[],'comparisons':[],'planned':[{'id':c['id'],'cells':list(m),'order':o,'status':'QUEUED'} for c,m,o in jobs],'qualification':{},'dynamic_or_inference_qualified':False};saved={}
    def save():d['elapsed_s']=time.monotonic()-start;dump(out/'summary.json',d)
    def alarm(s,f):raise TimeoutError('Frozen600sP2budget exceeded')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    try:
        for case,cells,order in jobs:
            row={'case_id':case['id'],'cells':list(cells),'tetrahedra':int(6*np.prod(cells)),'order':order,'completed':False};label=case['id']+'-tets'+str(row['tetrahedra'])+'-order'+str(order);slot=next(r for r in d['planned'] if r['id']==case['id'] and r['cells']==list(cells) and r['order']==order);slot['status']='RUNNING';save();begin=time.monotonic()
            try:
                m=QuadraticBlock(cells,order=order,**{k:case[k] for k in ['young','gravity','traction']});a=m.solve();row.update(m.diagnostics(a));row['nodes']=len(m.rest);row['completed']=True;row['history']=a['history'];field=m.query(a['positions'],points)
                row['physics_passed']=row['minimum_J']>.1 and row['maximum_constraint_error_m']==0 and row['original_free_residual_N']<=row['residual_tolerance_N'] and row['force_balance_defect_N']<=1e-8 and row['actual_moment_balance_defect_Nm']<=1e-8
                row['excitation_goal_met']=.02<=row['maximum_absolute_principal_stretch_strain']<=.08 and row['bulk_nonrigid_coordinate_RMS_m']>=.001
                path=out/(label+'.npz');np.savez_compressed(path,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=a['positions'],material_probes_m=points,probe_positions_m=field,force_N=m.load,constraint_mask=m.mask)
                row['capture']={'path':str(path),'sha256':sha(path)}
                prev=(case['id'],row['tetrahedra']//8,3) if order==3 else(case['id'],row['tetrahedra'],3)
                if prev in saved:
                    old,oldfield=saved[prev];offset=0;metrics={}
                    for name,P in groups.items():
                        e=field[offset:offset+len(P)]-oldfield[offset:offset+len(P)];offset+=len(P);metrics[name]={'maximum_Euclidean_error_m':float(np.linalg.norm(e,axis=1).max()),'coordinate_RMS_m':float(np.sqrt(np.mean(e*e)))}
                    d['comparisons'].append({'case_id':case['id'],'kind':'spatial' if order==3 else'quadrature','coarse_tetrahedra':old['tetrahedra'],'fine_tetrahedra':row['tetrahedra'],'orders':[old['order'],order],'errors':metrics,'passed':all(v['maximum_Euclidean_error_m']<=20e-6 for v in metrics.values())})
                saved[(case['id'],row['tetrahedra'],order)]=(row,field);slot['status']='COMPLETED'
            except TimeoutError as e:row['failure']={'exception':repr(e),'history':getattr(e,'history',[])};slot['status']='BUDGET_INTERRUPTED';raise
            except Exception as e:row['failure']={'exception':repr(e),'history':getattr(e,'history',[])};slot['status']='FAILED'
            finally:row['elapsed_s']=time.monotonic()-begin;d['cases'].append(row);dump(out/(label+'.json'),row);save()
        d['completed']=True
        for c in CASES:
            rows=[r for r in d['cases'] if r['case_id']==c['id'] and r['tetrahedra']==6144];comp=[r for r in d['comparisons'] if r['case_id']==c['id'] and r['fine_tetrahedra']==6144]
            spatial=any(r['kind']=='spatial' and r['passed'] for r in comp);quad=any(r['kind']=='quadrature' and r['passed'] for r in comp);physics=len(rows)==2 and all(r.get('physics_passed',False) for r in rows)
            d['qualification'][c['id']]={'finest_spatial_passed':spatial,'finest_quadrature_passed':quad,'finest_physics_passed':physics,'static_fixed_probe_qualified':spatial and quad and physics,'finest_excitation_goal_met':len(rows)==2 and all(r.get('excitation_goal_met',False) for r in rows)}
    except Exception as e:d['execution_failure']=repr(e)
    finally:signal.alarm(0);save()
    return d
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a=a.parse_args()
    if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
    else:
        d=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps({k:d.get(k) for k in ['completed','elapsed_s','qualification','execution_failure']},indent=2));raise SystemExit(0 if d['completed'] else 1)
