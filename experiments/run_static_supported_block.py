"""Bounded static block BC/gravity qualification; no dynamic/drop claim."""
from __future__ import annotations
import argparse,hashlib,itertools,json,signal,time
from pathlib import Path
import numpy as np
import torch
try:
    from .campaign_static_supported_block import StaticSupportedBlock,CASES,SIZE
    from .diagnose_nonlinear_physical_probes import interpolate
except ImportError:
    from campaign_static_supported_block import StaticSupportedBlock,CASES,SIZE
    from diagnose_nonlinear_physical_probes import interpolate
MESHES=((4,2,2),(8,4,4),(16,8,8))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')


def probes():
    L,W,H=SIZE;xs=L*np.array([.125,.375,.625,.875]);yz=np.array([-.25,0,.25]);groups={}
    groups['volume']=np.array(list(itertools.product(xs,yz*W,yz*H)))
    face=[]
    for x in [0,L]:face.extend([[x,y,z] for y,z in itertools.product(yz*W,yz*H)])
    for z in [-H/2,H/2]:face.extend([[x,y,z] for x,y in itertools.product(xs,yz*W)])
    for y in [-W/2,W/2]:face.extend([[x,y,z] for x,z in itertools.product(xs,yz*H)])
    groups['surface_face_interior']=np.asarray(face)
    groups['edge_midpoints']=np.array([[L/2,y,z] for y,z in itertools.product([-W/2,W/2],[-H/2,H/2])]+[[x,0,z] for x,z in itertools.product([0,L],[-H/2,H/2])]+[[x,y,0] for x,y in itertools.product([0,L],[-W/2,W/2])])
    groups['corners']=np.array(list(itertools.product([0,L],[-W/2,W/2],[-H/2,H/2])))
    return groups


def prepare(path):
    sources=['experiments/campaign_static_supported_block.py','experiments/run_static_supported_block.py','experiments/campaign_static_uniaxial_bar.py','experiments/campaign_sparse_nonlinear_gaussian.py','experiments/boundary_block.py','experiments/diagnose_nonlinear_physical_probes.py']
    beam_tip=1.5*625*9.81*SIZE[0]**4/(300000*SIZE[2]**2);beam_strain=3*625*9.81*SIZE[0]**2/(300000*SIZE[2])
    d={'schema':'static-supported-block-forward-qualification-v0.1','status':'FROZEN_BEFORE_BENCHMARK_CASES','CPU_threads':2,'hard_wall_cap_s':600,'domain_size_m':list(SIZE),'density_kg_m3':625.,'poisson':.35,'meshes_cells':[list(c) for c in MESHES],'tetrahedra':[96,768,6144],'cases':list(CASES),
      'boundary_definition':'Exactall-axisnodalconstraints,physicalcoordinatesindependentmesh;no springs/handles','separated_bottom_patches':{'z_m':-.04,'x_intervals_m':[[0,.06],[.18,.24]],'y_interval_m':[-.04,.04],'edges_conform_every_mesh':True},
      'loads':'Bodygravityrho0*V/4nodeweights;uniformknownreference-rightfaceP1tractionintegralsexact;nosurfacepointload','initialization':'EverycasefromREST;originalstaticNewtoncap40,no warmstart/retry,positiveJ>.1,unmodifiedtangent/objective',
      'pre_design_beam_estimate_NOT_validation':{'leftgravity_tip_sag_m':beam_tip,'root_axial_strain':beam_strain,'note':'EulerBernoullismallstrainestimateonly;full3DclampcornerstrainsmaydifferandrigidremovedbulkRMSmaymissgoal'},
      'deformation_goal':{'applies_only_expected_deformation_cases':True,'maximum_absolute_principal_stretch_strain':[.02,.08],'bulk_material_volume_proper_rigid_removed_coordinate_RMS_min_m':.001,'controls':'Bothend/bottom/separatedpatchesexpectedrestrictmotion;recordstrain/RMSbutdonotrequire1mm','auxiliary_strain':'Volumeweighted99thpercentileoftetmaximumabsoluteprincipalstrain,andvolumeweightedprincipalstrainRMS;NEVERreplacemaxstrainadmission'},
      'probe_groups_material_m':{k:v.tolist() for k,v in probes().items()},'spatial_accuracy':{'maximum_Euclidean_position_error_m':20e-6,'all_volume_surface_edge_corner_groups_retained':True,'maximum_strain_NOT_spatial_gate':True,'noise_sigma_reference_m':200e-6,'interpretation':'0.1sigmafixedphysicalsurfacepointcoordinateerror;notrendereddepthorallcontinuumpositions','qualifies_only_finest_adjacent_case_comparison':True},
      'physics_gates':{'minimum_J':.1,'original_free_force_residual':'FrozenstaticNewtonmax(1e-10,1e-10*resultantloadnorm)N','exact_constraint_error_m':0.,'force_balance_N':1e-8,'actual_moment_balance_Nm':1e-8},
      'source_sha256':{p:sha(p) for p in sources},'limits':['StaticBC/gravityandtractiononly;freefall/dropseparateIsaacexperiment','Boundaryedgesingularities/mesh/strainfailuresretained','No inverseorneuraltraininginthisstage','AlloldGaussianforwardfailuresunchanged']}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    dump(path,d);return d


def run(protocol,digest,output):
    if sha(protocol)!=digest:raise ValueError('Changedstaticprotocol')
    p=json.loads(Path(protocol).read_text())
    for path,expected in p['source_sha256'].items():
        if sha(path)!=expected:raise ValueError('Changedpinnedsource '+path)
    torch.set_num_threads(2);out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();groups={k:np.asarray(v) for k,v in p['probe_groups_material_m'].items()};P=np.concatenate(list(groups.values()))
    d={'schema':'static-supported-block-result-v0.1','completed':False,'protocol':{'path':str(protocol),'sha256':digest},'cases':[],'mesh_comparisons':[],'planned':[{'case_id':case['id'],'cells':list(c),'status':'QUEUED'} for c in MESHES for case in CASES],'inverse_or_network_authorized':False,'static_only':True}
    def save():d['elapsed_s']=time.monotonic()-start;dump(out/'summary.json',d)
    def alarm(s,f):raise TimeoutError('Prospective600sCPU2staticblockbudget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(600);previous={};final={}
    try:
        for cells in MESHES:
            for case in CASES:
                label=case['id']+'-cells'+'x'.join(map(str,cells));row={'id':label,'case_id':case['id'],'cells':list(cells),'tetrahedra':6*np.prod(cells).item(),'completed':False,'expected_deformation':case['expected_deformation']};cellstart=time.monotonic();slot=next(r for r in d['planned'] if r['case_id']==case['id'] and r['cells']==list(cells));slot['status']='RUNNING';save()
                try:
                    model=StaticSupportedBlock(cells,boundary=case['boundary'],young=case['young_Pa'],gravity=case['gravity'],traction=case['traction_N']);answer=model.solve();row.update(model.diagnostics(answer));row['completed']=True
                    row['history']=answer['history'];row['physics_passed']=row['minimum_J']>.1 and row['maximum_constraint_error_m']==0 and row['original_free_residual_N']<=row['residual_tolerance_N'] and row['force_balance_defect_N']<=1e-8 and row['actual_moment_balance_defect_Nm']<=1e-8
                    row['deformation_goal_met']=.02<=row['maximum_absolute_principal_stretch_strain']<=.08 and row['bulk_material_volume_nonrigid_coordinate_RMS_m']>=.001
                    row['deformation_goal_applicable']=case['expected_deformation'];fields=interpolate(model.reference_local,model.tets,answer['positions'][None],P)[0][0]
                    np.savez_compressed(out/(label+'.npz'),rest_m=model.rest,tets=model.tets,faces=model.faces,positions_m=answer['positions'],displacement_m=answer['displacement_local_m'],applied_forces_N=model.load,reaction_forces_N=np.where(model.mask,answer['residual'],0.),constraint_mask=model.mask,material_probes_m=P,probe_positions_m=fields)
                    row['trajectory']={'path':str(out/(label+'.npz')),'sha256':sha(out/(label+'.npz'))}
                    if case['id'] in previous:
                        old,oldfields=previous[case['id']];errors={};offset=0
                        for name,points in groups.items():
                            e=fields[offset:offset+len(points)]-oldfields[offset:offset+len(points)];errors[name]={'maximum_Euclidean_error_m':float(np.max(np.linalg.norm(e,axis=1))),'maximum_coordinate_error_m':float(np.max(abs(e))),'coordinate_RMS_m':float(np.sqrt(np.mean(e**2)))};offset+=len(points)
                        passed=all(r['maximum_Euclidean_error_m']<=20e-6 for r in errors.values());comparison={'case_id':case['id'],'coarse_tetrahedra':old['tetrahedra'],'fine_tetrahedra':row['tetrahedra'],'errors':errors,'threshold_m':20e-6,'passed':passed};d['mesh_comparisons'].append(comparison);final[case['id']]=comparison
                    previous[case['id']]=(row,fields);slot['status']='COMPLETED'
                except TimeoutError as exc:
                    row['failure']={'exception':repr(exc),'history':getattr(exc,'history',[])};slot['status']='BUDGET_INTERRUPTED';raise
                except Exception as exc:row['failure']={'exception':repr(exc),'history':getattr(exc,'history',[])};slot['status']='FAILED'
                finally:row['elapsed_s']=time.monotonic()-cellstart;d['cases'].append(row);dump(out/(label+'.json'),row);save()
        d['completed']=True;d['per_case_static_spatial_qualification']={}
        for case in CASES:
            finest=next((r for r in reversed(d['cases']) if r['case_id']==case['id']),{})
            fine_complete=finest.get('completed',False) and finest.get('tetrahedra')==6144
            last=final.get(case['id'],{})
            d['per_case_static_spatial_qualification'][case['id']]={'finest_case_completed':fine_complete,'finest_adjacent_20um_query_gate_passed':fine_complete and last.get('fine_tetrahedra')==6144 and last.get('passed',False),'finest_physics_passed':fine_complete and finest.get('physics_passed',False),'deformation_goal_applicable':case['expected_deformation'],'finest_deformation_goal_met':fine_complete and finest.get('deformation_goal_met',False),'scope':'Declaredstaticcase/fixedprobesonly;nostrainuniformityorcontinuumcertificate'}
    except Exception as exc:d['execution_failure']=repr(exc)
    finally:signal.alarm(0);save()
    return d

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('prepare');a.add_argument('--protocol',required=True)
    a=s.add_parser('run');a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256',required=True);a.add_argument('--output',required=True)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
    else:
        d=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps({k:d.get(k) for k in ['completed','elapsed_s','per_case_static_spatial_qualification','execution_failure']},indent=2));raise SystemExit(0 if d['completed'] else 1)
