"""Additive read-only audit of the complete frozen75-candidate calibration library.

Reads only protocol/freeze/source and library artifacts; no reserved observations,
forecast targets, inference selections or predictor/scorer calls.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
BASE=ROOT/'runs/coupled-surface-v13';REPORT=ROOT/'reports/coupled-surface-v13'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def audit(base=BASE,report=REPORT,root=ROOT):
    p=read(report/'protocol.json');freeze=read(report/'method-freeze.json');errors=[];rows=[]
    protocol_sha=digest(report/'protocol.json')
    if protocol_sha!=freeze['protocol_sha256']:errors.append('Protocol differs from freeze')
    source_checks={name:digest(root/name)==sha for name,sha in freeze['source_sha256'].items()}
    errors.extend('Frozen source changed: '+name for name,okay in source_checks.items() if not okay)
    expected_settings=dict(cells=p['coarse_cells'],dt=p['dt_coarse'],duration_seconds=p['duration_seconds'],scene_source='engineered_scenarios.py',mass_operator='consistent-P1',contact_formulation='clipped-normal-refined-friction-sparse-sphere',surface_spacing_m=p['surface_spacing_m'],friction_quadrature_level=p['friction_quadrature_level'])
    groups=[('fem',p['fem_young_grid'],35),('native',p['native_scale_grid'],35),('rigid',[6000.],5)];expected_paths=set();summaries={}
    for family,scales,count in groups:
        family_rows=[]
        if len(scales)*len(p['friction_grid'])!=count:errors.append(family+' declared grid count mismatch')
        for i,scale in enumerate(scales):
            for j,mu in enumerate(p['friction_grid']):
                path=base/'library'/family/f'{i}-{j}';statuspath=path/'complete.json';expected_paths.add(statuspath)
                job=dict(path=f'library/{family}/{i}-{j}',family=family,scenario='squeeze',scale=scale,friction=mu,reference=False)
                row=dict(family=family,index=[i,j],scale=scale,friction=mu,expected_job=job,present=statuspath.exists(),completed=False,valid_completed=False,failure=None,file_sha256={},audit_errors=[])
                if not statuspath.exists():row['audit_errors'].append('Missing completion record')
                else:
                    status=read(statuspath);row['file_sha256'][str(statuspath.relative_to(root))]=digest(statuspath)
                    if status.get('job')!=job:row['audit_errors'].append('Declared grid job mismatch')
                    if status.get('simulation_settings')!=expected_settings:row['audit_errors'].append('Simulation settings mismatch')
                    if status.get('protocol_sha256')!=protocol_sha:row['audit_errors'].append('Protocol hash mismatch')
                    if type(status.get('success')) is not bool:row['audit_errors'].append('Boolean status required')
                    row['completed']=status.get('success') is True
                    row['failure']=status.get('error') if not row['completed'] else None
                    row['element_guard_failure']=bool(row['failure'] and ('inverted/degenerate' in row['failure'] or 'Invalid native spring mesh' in row['failure']))
                    if row['completed']:
                        try:
                            for name in ['diagnostics.json','trajectory.npz']:row['file_sha256'][str((path/name).relative_to(root))]=digest(path/name)
                            d=read(path/'diagnostics.json');row['diagnostics']=d
                            if d!=status['diagnostics']:row['audit_errors'].append('Diagnostic record mismatch')
                            if not np.isclose(d['dt'],p['dt_coarse'],atol=1e-15,rtol=0):row['audit_errors'].append('Actual timestep mismatch')
                            with np.load(path/'trajectory.npz') as a:
                                required=['frames','velocities','rest','tets','centers','reactions','times','tool_positions','active']
                                if not all(np.isfinite(a[key]).all() for key in required):row['audit_errors'].append('Nonfinite library arrays')
                                expected_frames=round(p['duration_seconds']*p['fps'])+1
                                if a['frames'].shape!=(expected_frames,int(np.prod(np.array(p['coarse_cells'])+1)),3):row['audit_errors'].append('Forecast frame/node shape mismatch')
                                if a['times'].shape!=(expected_frames,) or not np.allclose(a['times'],np.arange(expected_frames)/p['fps'],atol=1e-10,rtol=0):row['audit_errors'].append('Saved time grid mismatch')
                                if not np.array_equal(a['frames'][0],a['rest']):row['audit_errors'].append('Initial geometry mismatch')
                                if a['velocities'].shape!=a['frames'].shape or a['reactions'].shape!=a['centers'].shape:row['audit_errors'].append('Velocity/reaction shape mismatch')
                            if not np.isfinite(d['min_j']) or d['min_j']<=.02:row['audit_errors'].append('Integrated J guard invalid')
                            row['penetration_within_1mm']=bool(np.isfinite(d['max_penetration_m']) and d['max_penetration_m']<=.001)
                        except (OSError,ValueError,KeyError) as exc:row['audit_errors'].append('Completed library artifact invalid: '+str(exc))
                    row['valid_completed']=row['completed'] and not row['audit_errors']
                errors.extend(job['path']+': '+e for e in row['audit_errors']);rows.append(row);family_rows.append(row)
        summaries[family]=dict(expected=count,present=sum(r['present'] for r in family_rows),completed=sum(r['completed'] for r in family_rows),valid_completed=sum(r['valid_completed'] for r in family_rows),failed=sum(r['present'] and not r['completed'] for r in family_rows),element_guard_failures=sum(r.get('element_guard_failure',False) for r in family_rows),successful_parameters=[dict(scale=r['scale'],friction=r['friction']) for r in family_rows if r['valid_completed']])
    extras=set((base/'library').rglob('complete.json'))-expected_paths
    if extras:errors.append('Undeclared library statuses: '+str(sorted(str(p) for p in extras)))
    return dict(passed=not errors,errors=errors,scope=__doc__,expected_total=75,present=sum(r['present'] for r in rows),families=summaries,candidates=rows,protocol_sha256=protocol_sha,method_freeze_sha256=digest(report/'method-freeze.json'),frozen_source_sha256=freeze['source_sha256'],frozen_source_checks=source_checks,audit_source_sha256=digest(Path(__file__)),baseline_scope='Native is unchanged author PhysTwin spring-force kernel inside this study shared graph/mass/contact/integration and positive-J guard. It is not the full authors optimizer/model/pipeline; guarded adaptation failures are not full-method PhysTwin failures.',validity_definition='Completed full calibration trajectory with consistent frozen job/settings, matching finite arrays/times/initial geometry and positive integrated J. Penetration threshold is reported separately, not used to silently remove a completed calibration hypothesis. No forecast performance follows from library coverage.')


def main():
    output=audit()
    path=REPORT/'library-coverage-audit.json';path.write_text(json.dumps(output,indent=2)+'\n')
    lines=['FROZEN CALIBRATION LIBRARY COVERAGE',output['baseline_scope'],output['validity_definition']]
    for family,s in output['families'].items():lines.append(f"{family}: {s['valid_completed']}/{s['expected']} valid complete; {s['failed']} failed, {s['element_guard_failures']} element-guard failures; {s['present']} accounted")
    lines+=['Audit passed: '+str(output['passed']),'Errors: '+json.dumps(output['errors']),'Complete per-candidate parameters/status/file hashes and frozen source/protocol evidence are in library-coverage-audit.json.']
    (REPORT/'library-coverage-audit.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
    return 0 if output['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
