"""Read-only volume-weighted motion audit after ALL50 base-resolution reference jobs finish.

No filtering by numerical qualification, no changes to predictions or cohorts.
The deformation/world metric is an RMS ratio, not a variance fraction or an
identifiability test. Failed references remain in case counts; numeric summaries
are conditional on readable completed trajectories.
"""
from pathlib import Path
import sys,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.consistent_mass import assemble_mass
REPORT=ROOT/'reports/coupled-surface-v13'
BASE=ROOT/'runs/coupled-surface-v13/reference'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def motion_metrics(rest,tets,frames,times,density=1000.):
    rest=np.asarray(rest,float);frames=np.asarray(frames,float);times=np.asarray(times,float)
    if rest.ndim!=2 or rest.shape[1]!=3 or frames.ndim!=3 or frames.shape[1:]!=rest.shape:
        raise ValueError('Expected rest[N,3] and frames[T,N,3]')
    if len(frames)<2 or times.shape!=(len(frames),) or not np.all(np.diff(times)>0):
        raise ValueError('At least two strictly time-ordered frames required')
    if not all(np.isfinite(x).all() for x in [rest,frames,times]):raise ValueError('Nonfinite motion input')
    if not np.allclose(frames[0],rest,rtol=0,atol=1e-10):raise ValueError('Initial frame differs from undeformed rest geometry')
    mass=assemble_mass(rest,tets,density);weights=np.asarray(mass.sum(axis=1)).ravel();total=float(weights.sum())
    rest_com=weights@rest/total;A=rest-rest_com
    def ms(value):return max(0.,float(np.sum(value*(mass@value)))/(3*total))
    per_frame=[]
    for t,x in zip(times[1:],frames[1:]):
        com=weights@x/total;B=x-com
        U,_,Vt=np.linalg.svd(A.T@(mass@B))
        correction=np.eye(3);correction[-1,-1]=1. if np.linalg.det(U@Vt)>=0 else -1.
        rotation=U@correction@Vt  # row-vector proper rotation: rest_centered@rotation
        fit=A@rotation+com
        world=ms(x-rest);deform=ms(x-fit);rigid_fit=ms(fit-rest)
        per_frame.append(dict(time_s=float(t),world_coordinate_rms_m=float(np.sqrt(world)),
            deformation_coordinate_rms_m=float(np.sqrt(deform)),rigid_fit_coordinate_rms_m=float(np.sqrt(rigid_fit)),
            deformation_world_rms_ratio=None if world<=1e-30 else float(np.sqrt(deform/world)),
            physical_com_m=com.tolist(),com_translation_m=float(np.linalg.norm(com-rest_com)),
            proper_rotation_determinant=float(np.linalg.det(rotation))))
    world=float(np.sqrt(np.mean([r['world_coordinate_rms_m']**2 for r in per_frame])))
    deform=float(np.sqrt(np.mean([r['deformation_coordinate_rms_m']**2 for r in per_frame])))
    return dict(mass_kg=total,rest_physical_com_m=rest_com.tolist(),postinitial_frames=len(per_frame),
        world_coordinate_rms_m=world,deformation_coordinate_rms_m=deform,
        rigid_fit_coordinate_rms_m=float(np.sqrt(np.mean([r['rigid_fit_coordinate_rms_m']**2 for r in per_frame]))),
        deformation_world_rms_ratio=None if world<=1e-15 else deform/world,
        max_frame_deformation_coordinate_rms_m=max(r['deformation_coordinate_rms_m'] for r in per_frame),
        max_com_translation_m=max(r['com_translation_m'] for r in per_frame),per_frame=per_frame)


def summarize(rows):
    valid=[r['motion'] for r in rows if r['audit_complete']]
    ratios=[m['deformation_world_rms_ratio'] for m in valid if m['deformation_world_rms_ratio'] is not None]
    def spread(values):
        return dict(min=float(np.min(values)),median=float(np.median(values)),max=float(np.max(values))) if values else None
    world=float(np.sqrt(np.mean([m['world_coordinate_rms_m']**2 for m in valid]))) if valid else None
    deform=float(np.sqrt(np.mean([m['deformation_coordinate_rms_m']**2 for m in valid]))) if valid else None
    return dict(expected=len(rows),audited=len(valid),failed_or_unreadable=len(rows)-len(valid),
        generated_failures=sum(not r['reference_generation_success'] for r in rows),
        pooled_equal_case_world_coordinate_rms_m=world,pooled_equal_case_deformation_coordinate_rms_m=deform,
        pooled_deformation_world_rms_ratio=None if world is None or world<=1e-15 else deform/world,
        case_deformation_coordinate_rms_m=spread([m['deformation_coordinate_rms_m'] for m in valid]),
        case_deformation_world_rms_ratio=spread(ratios),undefined_ratio_cases=len(valid)-len(ratios))


def main():
    protocol_path=REPORT/'protocol.json';freeze_path=REPORT/'method-freeze.json'
    protocol=json.loads(protocol_path.read_text());freeze=json.loads(freeze_path.read_text())
    if digest(protocol_path)!=freeze['protocol_sha256']:raise RuntimeError('Protocol differs from frozen protocol hash')
    scenes=[protocol['calibration_scenario']]+protocol['forecast_scenarios']
    if set(protocol['primary_forecast_scenarios']) & set(protocol['stress_forecast_scenarios']) or set(protocol['primary_forecast_scenarios']) | set(protocol['stress_forecast_scenarios']) != set(protocol['forecast_scenarios']):raise RuntimeError('Forecast cohorts must form a disjoint complete partition')
    expected=[(obj,scene,BASE/obj['id']/scene) for obj in protocol['objects'] for scene in scenes]
    if len(expected)!=50 or len({(o['id'],s) for o,s,_ in expected})!=50:raise RuntimeError('Exactly50 distinct declared base-resolution references required')
    missing=[str(path/'complete.json') for _,_,path in expected if not (path/'complete.json').exists()]
    if missing:raise RuntimeError(f'Population motion audit waits for ALL50 completion records; {len(missing)} still missing')
    source_names=['surface_motion_audit.py','consistent_mass.py']
    sources={str(Path('scripts/coupled_contact_v13')/name):digest(ROOT/'scripts/coupled_contact_v13'/name) for name in source_names}
    key='scripts/coupled_contact_v13/consistent_mass.py'
    if sources[key]!=freeze['source_sha256'][key]:raise RuntimeError('Frozen consistent-mass source changed')
    inputs={str(protocol_path.relative_to(ROOT)):digest(protocol_path),str(freeze_path.relative_to(ROOT)):digest(freeze_path)}
    rows=[]
    for obj,scene,path in expected:
        status_path=path/'complete.json';inputs[str(status_path.relative_to(ROOT))]=digest(status_path);status=json.loads(status_path.read_text())
        cohort='calibration' if scene==protocol['calibration_scenario'] else ('primary' if scene in protocol['primary_forecast_scenarios'] else 'stress')
        row=dict(object=obj['id'],split=obj['split'],scenario=scene,cohort=cohort,
                 reference_generation_success=status.get('success') is True,audit_complete=False)
        if status.get('success') is not True:
            row['failure']=status.get('error','Reference generation failed')
        else:
            try:
                job=status['job']
                if job['path']!=f"reference/{obj['id']}/{scene}" or job['scenario']!=scene or job['scale']!=obj['young'] or job['friction']!=obj['friction'] or not job['reference']:raise ValueError('Completion record does not match declared reference')
                trajectory=path/'trajectory.npz';inputs[str(trajectory.relative_to(ROOT))]=digest(trajectory)
                with np.load(trajectory,allow_pickle=False) as data:
                    expected_times=np.arange(round(protocol['duration_seconds']*protocol['fps'])+1)/protocol['fps']
                    if data['times'].shape!=expected_times.shape or not np.allclose(data['times'],expected_times,rtol=0,atol=1e-9):raise ValueError('Saved reference timeline is incomplete or differs from protocol')
                    row['motion']=motion_metrics(data['rest'],data['tets'],data['frames'],data['times'],protocol['density'])
                row['audit_complete']=True
            except Exception as error:row['failure']=repr(error)
        rows.append(row)
    grouped={}
    for split in ['all',*sorted({r['split'] for r in rows})]:
        selected=rows if split=='all' else [r for r in rows if r['split']==split]
        grouped[split]={cohort:summarize(selected if cohort=='all' else [r for r in selected if r['cohort']==cohort]) for cohort in ['all','calibration','primary','stress']}
    if any(digest(ROOT/p)!=h for p,h in {**inputs,**sources}.items()):raise RuntimeError('Audit inputs/source changed while reading')
    output=dict(scope='Descriptive motion audit of all50 base-resolution reference jobs, including failures and unqualified stress. Numeric RMS summaries are conditional on auditable completed trajectories; failed cases remain in counts. No quality filtering or inference changes.',
        definition='Uniform physical density and exact P1 consistent-volume inner product. Proper mass-weighted Kabsch per saved postinitial frame; equal-frame temporal averaging and equal-case group pooling. World and deformation coordinate RMS divide quadratic error by3*totalmass.',
        interpretation='Deformation/world is an RMS ratio, NOT a variance fraction. Rigid-fit and residual RMS are not asserted to form an additive variance decomposition. Physical COM uses mass-matrix row sums. Motion amplitude alone establishes neither material identifiability, calibrated uncertainty nor reference truth.',
        source_sha256=sources,input_sha256=inputs,expected=50,present=len(rows),summary=grouped,
        by_scenario={s:summarize([r for r in rows if r['scenario']==s]) for s in scenes},cases=rows)
    destination=REPORT/'reference-motion-audit.json';temporary=destination.with_suffix('.tmp');temporary.write_text(json.dumps(output,indent=2)+'\n');temporary.replace(destination)


if __name__=='__main__':main()
