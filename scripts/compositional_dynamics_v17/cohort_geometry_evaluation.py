"""Evaluate frozen fresh forecasts only after all16 method/query attempts commit."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write,face_grid
from scripts.compositional_dynamics_v17.cohort_generator import guard
from scripts.compositional_dynamics_v17.forecast_metrics import compare_surface_forecast
from scripts.compositional_dynamics_v17.surface_physics_metrics import diagnostics,compare_diagnostics


def committed(bindings):
    protocol=guard(**bindings);base=ROOT/'runs/compositional-dynamics-v17/fresh-cohort';rows=[];evidence={}
    for world in protocol['generator_settings']['world_ids']:
        for role in ('forecast-40mm','forecast-80mm'):
            for method,folder in [('ours',base/'shared-forecast'/world/role),('native_phystwin',base/'native'/world/'forecasts'/role)]:
                path=folder/'prediction-commit.json';record=json.loads(path.read_text());evidence[str(path.relative_to(ROOT))]=sha(path)
                if record.get('protocol_sha256')!=bindings['protocol_sha'] or record.get('method_freeze_sha256')!=bindings['freeze_sha'] or record.get('world_id')!=world or record.get('role')!=role or not isinstance(record.get('completed'),bool):raise ValueError('Exact terminal method/world/query binding required')
                array=folder/('surface-prediction.npz'if method=='ours'else'prediction.npz')
                if record['completed']:
                    key='surface_prediction_sha256'if method=='ours'else'prediction_sha256'
                    if sha(array)!=record[key]:raise ValueError('Forecast array changed')
                    evidence[str(array.relative_to(ROOT))]=sha(array)
                    complete=folder/'complete.json';expected='complete_sha256'if method=='ours'else'native_complete_sha256'
                    if sha(complete)!=record[expected]:raise ValueError('Forecast completion changed')
                    status=json.loads(complete.read_text())
                    if not status['completed']:raise ValueError('Inconsistent completion')
                    evidence[str(complete.relative_to(ROOT))]=sha(complete)
                    if method=='native_phystwin':
                        plan=base/'pipeline/native-plans'/f'{world}.json'
                        if status['plan_sha256']!=sha(plan):raise ValueError('Native world plan changed')
                        evidence[str(plan.relative_to(ROOT))]=sha(plan)
                rows.append(dict(world_id=world,role=role,method=method,record=record,array=array))
    # Returning occurs only after all16 seals. No reference path has been opened.
    return protocol,rows,evidence


def main(bindings):
    out=REPORT/'fresh-cohort-geometry-results.json'
    if out.exists():raise ValueError('Preserve previous evaluation')
    p,commits,evidence=committed(bindings);base=ROOT/'runs/compositional-dynamics-v17/fresh-cohort'
    generation_path=REPORT/'fresh-cohort-generation.json';generation=json.loads(generation_path.read_text())
    if not generation['terminal'] or generation['protocol_sha256']!=bindings['protocol_sha'] or generation['method_freeze_sha256']!=bindings['freeze_sha']:raise ValueError('Wait for terminal declared reference census')
    evidence[str(generation_path.relative_to(ROOT))]=sha(generation_path);rows=[];dense=face_grid(17);dims=np.array(p['generator_settings']['dimensions_m'])
    for world in p['generator_settings']['world_ids']:
        metadata_path=base/'public'/world/'metadata.json';status_path=base/'public'/world/'complete.json';export=json.loads(status_path.read_text())
        evidence[str(status_path.relative_to(ROOT))]=sha(status_path)
        if export.get('world_id')!=world or export.get('protocol_sha256')!=bindings['protocol_sha'] or export.get('method_freeze_sha256')!=bindings['freeze_sha']:raise ValueError('Public export identity mismatch')
        if not export['completed']:
            for role in ('forecast-40mm','forecast-80mm'):rows.append(dict(world_id=world,role=role,reference_complete=False,reason='Calibration unavailable',methods={r['method']:dict(completed=False,error=r['record'].get('error'))for r in commits if r['world_id']==world and r['role']==role}))
            continue
        if sha(metadata_path)!=export['public_metadata_sha256']:raise ValueError('Public metadata seal changed')
        evidence[str(metadata_path.relative_to(ROOT))]=sha(metadata_path)
        metadata=json.loads(metadata_path.read_text());q=np.array(metadata['material_queries']);rest=q*dims+[0,-.03,-.03];indices=[np.flatnonzero(np.all(dense==point,axis=1))[0]for point in q]
        for role in ('forecast-40mm','forecast-80mm'):
            folder=base/'references'/world/role/'n4-dt0.00002500';record_path=folder/'complete.json';row=dict(world_id=world,role=role,reference_complete=False,methods={r['method']:dict(completed=r['record']['completed'],error=r['record'].get('error')) for r in commits if r['world_id']==world and r['role']==role})
            if not record_path.exists():row['reason']='Declared reference missing at terminal census';rows.append(row);continue
            record=json.loads(record_path.read_text());evidence[str(record_path.relative_to(ROOT))]=sha(record_path)
            if record.get('n')!=4 or record.get('dt_s')!=25e-6 or record['world_id']!=world or record['role']!=role or record['protocol_sha256']!=bindings['protocol_sha'] or record['method_freeze_sha256']!=bindings['freeze_sha']:raise ValueError('Reference identity mismatch')
            if not record['completed']:row['reason']=record.get('error','Incomplete reference');rows.append(row);continue
            array=folder/'trajectory.npz'
            if sha(array)!=record['trajectory_sha256']:raise ValueError('Reference array changed')
            evidence[str(array.relative_to(ROOT))]=sha(array)
            with np.load(array)as f:
                ids=np.arange(0,len(f['times_s']),10);t=f['times_s'][ids].copy();x=f['surface_frames'][ids].reshape(len(ids),-1,3)[:,indices].copy()
            if len(t)!=31 or not np.isfinite(t).all() or not np.allclose(t,np.arange(31)*.01,atol=1e-12,rtol=0):raise ValueError('Declared31frame reference clock required')
            floor=metadata['forecast_public_setups'][role]['floor_z_m'];refdiag=diagnostics(t,x,q,dims,floor)
            row.update(reference_complete=True,reference_qualified=False,reference_scope='Finite-resolution n4 generator; sensitivity reported separately, no automatic continuum qualification.')
            for r in commits:
                if r['world_id']!=world or r['role']!=role:continue
                result=dict(completed=r['record']['completed'],error=r['record'].get('error'))
                if result['completed']:
                    with np.load(r['array'])as f:pt=f['times_s'].copy();pred=f['positions_m'if r['method']=='ours'else'tracked_positions_m'].copy()
                    result['geometry']=compare_surface_forecast(rest,q,dims,t,x,pt,pred)
                    if not result['geometry']['complete']:raise ValueError('Completed prediction has missing frames')
                    result['physical_diagnostics']=compare_diagnostics(refdiag,diagnostics(pt,pred,q,dims,floor))
                row['methods'][r['method']]=result
            rows.append(row)
    fields=['world_point_rms_m','shape_point_rms_m','surface_centroid_error_m']
    paired=[r for r in rows if r['reference_complete'] and all(r['methods'].get(m,{}).get('completed')for m in ['ours','native_phystwin'])]
    summaries={}
    for method in ['ours','native_phystwin']:
        values={field:float(np.mean([r['methods'][method]['geometry'][field]for r in paired]))if paired else None for field in fields}
        summaries[method]=dict(primary_all_eight_mean=values if len(paired)==8 else None,conditional_paired_complete_case_mean=values,paired_complete_forecasts=len(paired),expected_forecasts=8)
    write(out,dict(protocol_sha256=bindings['protocol_sha'],method_freeze_sha256=bindings['freeze_sha'],scope='Fresh four-world finite-resolution load-magnitude extrapolation; oracle3D observation inputs, known initial geometry/mass. No raw-video, unseeninteractiontype, continuum or real-world claim.',rows=rows,summary=summaries,paired_world_error_difference_ours_minus_native_m=[dict(world_id=r['world_id'],role=r['role'],difference=r['methods']['ours']['geometry']['world_point_rms_m']-r['methods']['native_phystwin']['geometry']['world_point_rms_m'])for r in paired],source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__)},evidence_sha256=evidence,rendering_evaluation='Separate native-Gaussian/common-procedural image reports; this report establishes no rendering parity.'))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['protocol-path','freeze-path','approval-path','protocol-sha','freeze-sha']:parser.add_argument('--'+key,required=True)
    main(vars(parser.parse_args()))
