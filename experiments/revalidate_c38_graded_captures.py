"""NEW postincident evaluation, NOT recovery of the lost original case report.

Hash-bound original captures/checkpoints supply accepted displacements. The
original polynomial energy/forces are evaluated at those SAME displacements;
no Newton, CG, new load, observation or physical trajectory is computed.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, os, resource, signal, subprocess, time
from pathlib import Path
import numpy as np
from experiments.run_campaign_c38_static_graded_pilot import safe, GATES, GPU_UUID
from experiments.campaign_c38_supported_grading import SupportedBoundaryGradedGeometry
from experiments.campaign_c38_static_family import array_record, wrench
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/revalidate_c38_graded_captures.py'
SUMMARY='runs/operator-dev/c38-static-graded-pilot-v01/completed-summary.json'
INTEGRITY='runs/operator-dev/c38-static-graded-refinement-v01/raw-integrity-check.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(path):
    p=Path(path).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Repository-local artifact required')
    return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def sources():
    pending=[SOURCE,'experiments/campaign_p2_supported_preconditioner.py','experiments/campaign_p2_jacobian_certificate.py'];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);tree=ast.parse((ROOT/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('experiments.'):
                dep='experiments/'+node.module.split('.')[1]+'.py'
                if (ROOT/dep).is_file():pending.append(dep)
    return sorted(seen)
def exclusive(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as stream:json.dump(safe(value),stream,indent=2,allow_nan=False);stream.write('\n')
def prepare(protocol_path,output):
    destination=Path(protocol_path).resolve();out=Path(output).resolve()
    if destination.exists() or out.exists() or not out.is_relative_to(ROOT):raise ValueError('Fresh protocol/output required')
    summary=json.loads((ROOT/SUMMARY).read_text());integrity=json.loads((ROOT/INTEGRITY).read_text())
    if not summary['completed'] or not integrity['completed'] or not integrity['all6_capture_hashes_match_preincident_completed_summary']:raise ValueError('Original summary and postincident raw hash verification required')
    p=dict(schema='C38-graded-postincident-readonly-revalidation-v01',status='FROZEN_BEFORE_GPU_EVALUATION',source_sha256={n:sha(ROOT/n) for n in sources()},output=str(out.relative_to(ROOT)),prior_summary=record(ROOT/SUMMARY),raw_integrity=record(ROOT/INTEGRITY),records=integrity['records'],gates=GATES,comparison_tolerances=dict(original_checkpoint_R_N=1e-12,original_checkpoint_potential_J=1e-11,finest_summary_R_N=1e-12,finest_summary_wrench=1e-10,finest_summary_strain=1e-10,finest_summary_bulk_m=1e-12,original_saved_query_m=1e-14),resources=dict(wall_cap_s=180,CPU_threads=2,physical_GPU_UUID=GPU_UUID,GPU_allocated_cap_bytes=8*1024**3,RSS_cap_bytes=12*1024**3,disk_free_reserve_bytes=30*1024**3),scope='Evaluate SAME original accepted displacement under original24/80 operators; NO Newton/CG, new observations, dynamics, physical trajectory or caseJSON recovery',original_lost_case_path='runs/gaussian-campaign-v01/c38-static-graded-pilot-v01/Elo__both_ends__down2/result.json',original_lost_case_sha256='eddc3f42c063e4f1ecd5f0d5f3b0aaa2b890f6c9ac3dd8248a94336eded07f1b',overwritten_bytes=record(ROOT/'runs/operator-dev/c38-static-graded-refinement-v01/failed-preparation-overwrite-bytes.json'),original_case_JSON_recovered=False,raw_local=True)
    exclusive(destination,p);return sha(destination)
def validate(protocol_path,digest):
    if sha(protocol_path)!=digest:raise ValueError('Protocol changed')
    p=json.loads(Path(protocol_path).read_text())
    if p['schema']!='C38-graded-postincident-readonly-revalidation-v01' or p['gates']!=GATES or set(p['source_sha256'])!=set(sources()):raise ValueError('Exact evaluation scope required')
    for n,d in p['source_sha256'].items():
        if sha(ROOT/n)!=d:raise ValueError('Source changed:'+n)
    for k in ('prior_summary','raw_integrity','overwritten_bytes'):
        r=p[k]
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Evidence changed')
    integrity=json.loads((ROOT/p['raw_integrity']['path']).read_text())
    if p['records']!=integrity['records'] or len(p['records'])!=6:raise ValueError('All6 fixed fields required')
    for r in p['records']:
        for k in ('capture','latest_pointer','latest_metadata','latest_data'):
            a=r[k]
            if sha(ROOT/a['path'])!=a['sha256']:raise ValueError('Original raw artifact changed')
    return p

def evaluate(p,protocol_path):
    import shutil
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID or any(os.environ.get(k)!='2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Exact GPU2UUID+CPU2 required')
    out=ROOT/p['output'];out.mkdir(parents=True,exist_ok=False);start=time.monotonic();result=dict(completed=False,passed=False,protocol=record(protocol_path),pid=os.getpid(),rows=[],raw_local=True,original_case_JSON_recovered=False,physics_solves=0)
    def save():
        result['seconds']=time.monotonic()-start
        temporary=out/'summary.json.tmp';temporary.write_text(json.dumps(safe(result),indent=2,allow_nan=False)+'\n');temporary.replace(out/'summary.json')
    def expired(*_):raise TimeoutError('Frozen180s read-only evaluation cap')
    signal.signal(signal.SIGALRM,expired);signal.alarm(180);save()
    try:
        import torch
        from experiments.campaign_p2_supported_preconditioner import StreamingSupportedP2
        from experiments.campaign_p2_jacobian_certificate import whole_p2_jacobian_bounds
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(min(1.,8*1024**3/torch.cuda.get_device_properties(0).total_memory),0)
        torch.zeros(1,device='cuda:0').sum().item();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True);own=[a.split(',')[1].strip() for a in apps.splitlines() if a.split(',')[0].strip()==str(os.getpid())]
        if set(own)!={GPU_UUID}:raise ValueError('OwncomputePID must use only physicalGPU2')
        result['GPU_provenance']={'physical_UUIDs':own}
        summary=json.loads((ROOT/p['prior_summary']['path']).read_text());bycase={r['case']['id']:r for r in summary['rows']};tol=p['comparison_tolerances'];gates=p['gates']
        for r in p['records']:
            # Every raw artifact hash checked again immediately before read.
            for key in ('capture','latest_metadata','latest_data'):
                if sha(ROOT/r[key]['path'])!=r[key]['sha256']:raise ValueError('Artifact changed before evaluation')
            original=bycase[r['case']];case=original['case'];meta=json.loads((ROOT/r['latest_metadata']['path']).read_text())
            with np.load(ROOT/r['capture']['path'],allow_pickle=False) as a:
                arrays={k:a[k].copy() for k in a.files}
            with np.load(ROOT/r['latest_data']['path'],allow_pickle=False) as a:accepted=a['displacement_m'].copy()
            if not np.array_equal(accepted,arrays['displacement_m']):raise ValueError('Capture differs from original accepted displacement')
            q=arrays['positions_m'];u=torch.tensor(accepted,dtype=torch.float64,device='cuda:0');evaluations=[];models=[]
            for rule in ('keast24','exact80'):
                m=SupportedBoundaryGradedGeometry(tuple(r['cells']),case['young_Pa'],case['support'],case['Fz_N'],rule=rule)
                for name,key in [('rest','rest_m'),('tets','tets'),('elements','elements'),('faces','faces'),('mask','constraint_mask'),('load','known_total_force_N')]:
                    if not np.array_equal(getattr(m,name),arrays[key]):raise ValueError('Same original geometry/mask/fullforce required')
                if rule=='keast24':
                    for name,identity in meta['identity']['model_arrays'].items():
                        if array_record(getattr(m,name))!=identity:raise ValueError('Accepted geometry identity differs')
                operator=StreamingSupportedP2(m,'cuda:0',4000,8,support_kind=case['support']);cache=operator.evaluate(u)
                evaluations.append(dict(E=float(cache['energy']),gradient=cache['gradient'].detach().cpu().numpy(),J=float(cache['minimum_J'])));models.append(m);del cache,operator;torch.cuda.empty_cache()
            m=models[-1];v24,v80=evaluations;r24=float(np.linalg.norm((v24['gradient']-m.load)[~m.mask]));residual=v80['gradient']-m.load;r80=float(np.linalg.norm(residual[~m.mask]));diag=m.diagnostics(dict(positions=q,residual=residual,J=np.array([v80['J']]),tolerance_N=1e-10,iterations=meta['iteration']));cert=whole_p2_jacobian_bounds(m,q);query=m.query(q,arrays['material_probes_m']);query_error=float(np.max(abs(query-arrays['probe_positions_m'])));potential=v24['E']-float(np.sum(m.load*accepted));gd=float(np.linalg.norm(v24['gradient']-v80['gradient']));ed=abs(v24['E']-v80['E'])
            physics=bool(r24<=gates['free_residual24_N'] and r80<=gates['free_residual80_N'] and gd<=gates['gradient24_80_difference_N'] and ed<=gates['energy24_80_difference_J'] and diag['force_balance_defect_N']<=gates['force_balance_N'] and diag['actual_moment_balance_defect_Nm']<=gates['actual_moment_balance_Nm'] and diag['maximum_constraint_error_m']==0 and v80['J']>.1 and cert['all_elements_positive'])
            checkpoint=abs(r24-meta['residual_N'])<=tol['original_checkpoint_R_N'] and abs(potential-meta['potential_J'])<=tol['original_checkpoint_potential_J']
            prior=True
            if r['cells'][0]==24:
                prior=bool(abs(r80-original['finest_original_free_R80_N'])<=tol['finest_summary_R_N'] and abs(diag['force_balance_defect_N']-original['finest_force_balance_N'])<=tol['finest_summary_wrench'] and abs(diag['actual_moment_balance_defect_Nm']-original['finest_moment_balance_Nm'])<=tol['finest_summary_wrench'] and abs(diag['maximum_absolute_principal_stretch_strain']-original['finest_peak_sampled_principal_strain'])<=tol['finest_summary_strain'] and abs(diag['bulk_nonrigid_coordinate_RMS_m']-original['finest_bulk_nonrigid_coordinate_RMS_m'])<=tol['finest_summary_bulk_m'])
            row=dict(case=case,cells=r['cells'],capture=r['capture'],checkpoint_metadata=r['latest_metadata'],revalidated_physics_passed=physics,checkpoint_metrics_agree=bool(checkpoint),preincident_finest_summary_agrees=prior,original_saved_query_max_error_m=query_error,R24_N=r24,R80_N=r80,energy24_J=v24['E'],energy80_J=v80['E'],gradient24_80_difference_N=gd,energy24_80_difference_J=ed,diagnostics=diag,whole_J={'all_elements_positive':cert['all_elements_positive'],'minimum_lower_bound':cert['minimum_lower_bound']},hmax_m=m.maximum_reference_tet_edge_m,new_provenance_not_original_report=True)
            row['passed']=bool(physics and checkpoint and prior and query_error<=tol['original_saved_query_m']);result['rows'].append(row);save()
            if not row['passed']:raise ValueError('New evaluation failed original physical/parity criterion')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>12*1024**3 or torch.cuda.max_memory_allocated()>8*1024**3 or shutil.disk_usage(ROOT).free<30*1024**3:raise MemoryError('Frozen evaluation resource guard')
            del models,m,u;torch.cuda.empty_cache();print(r['case'],r['cells'][0],'evaluationPASS',flush=True)
        result.update(completed=True,passed=True)
    except BaseException as error:result['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:signal.alarm(0);save()
    return result

def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');args=a.parse_args()
    if args.mode=='prepare':print(prepare(args.protocol,args.output));return
    p=validate(args.protocol,args.protocol_sha256);result=evaluate(p,args.protocol);raise SystemExit(0 if result['passed'] else 1)
if __name__=='__main__':main()
