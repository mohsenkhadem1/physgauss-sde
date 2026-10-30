"""Read-only exact-array reproduction check on two completed development cases.

Compares production moving-sphere primary references to prior qualification with
identical public scene/material/contact/inertia/grid/time settings. No reruns,
reserved cases, prediction targets or parameter selection.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
REPORT=ROOT/'reports/coupled-surface-v13';BASE=ROOT/'runs/coupled-surface-v13'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def array_digest(a):
    prefix=json.dumps(dict(shape=a.shape,dtype=a.dtype.str),sort_keys=True).encode()
    return hashlib.sha256(prefix+np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    errors=[];qpath=REPORT/'physics-development-qualification-unadopted.json';q=read(qpath)
    evidence=q['scenarios']['moving_sphere']['gates']['primary_time'];oldpath=ROOT/evidence['report'];old=read(oldpath)
    if digest(oldpath)!=evidence['report_sha256']:errors.append('Qualification evidence report changed')
    p=read(REPORT/'protocol.json');frozen=read(REPORT/'method-freeze.json')
    if digest(REPORT/'protocol.json')!=frozen['protocol_sha256']:errors.append('Frozen protocol changed')
    physics_names=['physics.py','surface_contact.py','sparse_surface_contact.py','clipped_surface_contact.py','clipped_surface_contact.cpp','consistent_mass.py','engineered_scenarios.py']
    source_checks={}
    for name in physics_names:
        rel='scripts/coupled_contact_v13/'+name
        source_checks[rel]=old['plan']['source_sha256'][rel]==frozen['source_sha256'][rel]==digest(ROOT/rel)
    errors.extend('Qualified/production physics source mismatch: '+name for name,okay in source_checks.items() if not okay)
    rows=[]
    for obj,E,mu in [('surface_dev0',6000.,.35),('surface_dev1',3000.,.1)]:
        spec=['moving_sphere',E,mu,12,25e-6,.0015,1]
        matches=[r for r in old['runs'] if r.get('spec')==spec]
        if len(matches)!=1:raise RuntimeError('Expected exactly one matching qualified trajectory')
        prior=matches[0];oldfolder=ROOT/prior['path'];newfolder=BASE/'reference'/obj/'moving_sphere';status=read(newfolder/'complete.json')
        row=dict(object=obj,scenario='moving_sphere',qualification_spec=spec,errors=[],comparisons={})
        expected_job=dict(path=f'reference/{obj}/moving_sphere',family='fem',scenario='moving_sphere',scale=E,friction=mu,reference=True)
        settings=dict(cells=[12]*3,dt=25e-6,duration_seconds=.6,scene_source='engineered_scenarios.py',mass_operator='consistent-P1',contact_formulation='clipped-normal-refined-friction-sparse-sphere',surface_spacing_m=.0015,friction_quadrature_level=1)
        if status.get('job')!=expected_job or status.get('simulation_settings')!=settings:row['errors'].append('Production scene/parameter/settings mismatch')
        if status.get('protocol_sha256')!=frozen['protocol_sha256']:row['errors'].append('Production status protocol hash mismatch')
        if not prior['success'] or not status['success']:row['errors'].append('One trajectory did not complete')
        for key in ['dt','min_j','max_penetration_m','friction_work_j','contact_samples']:
            if prior[key]!=status['diagnostics'][key]:row['errors'].append('Integrated diagnostic differs: '+key)
        files=[newfolder/'complete.json',newfolder/'diagnostics.json',newfolder/'trajectory.npz',oldfolder/'status.json',oldfolder/'trajectory.npz']
        row['file_sha256']={str(path.relative_to(ROOT)):digest(path) for path in files}
        with np.load(oldfolder/'trajectory.npz') as a,np.load(newfolder/'trajectory.npz') as b:
            for key in ['rest','tets','times','frames','centers','velocities','reactions','tool_positions','active']:
                x,y=a[key],b[key];same=x.shape==y.shape and x.dtype==y.dtype
                exact=bool(same and np.array_equal(x,y));finite=bool(np.isfinite(x).all() and np.isfinite(y).all())
                row['comparisons'][key]=dict(exact_equal=exact,finite=finite,qualification_array_sha256=array_digest(x),production_array_sha256=array_digest(y),max_absolute_difference=float(np.max(np.abs(x-y))) if same else None)
                if not exact or not finite:row['errors'].append('Array reproduction mismatch: '+key)
        row['passed']=not row['errors'];errors.extend(obj+': '+e for e in row['errors']);rows.append(row)
    result=dict(passed=not errors,errors=errors,scope=__doc__,cases=rows,physics_source_checks=source_checks,physics_source_sha256={name:frozen['source_sha256'][name] for name in source_checks},protocol_sha256=digest(REPORT/'protocol.json'),method_freeze_sha256=digest(REPORT/'method-freeze.json'),qualification_aggregate_sha256=digest(qpath),qualification_evidence_sha256=digest(oldpath),audit_source_sha256=digest(Path(__file__)),interpretation='Matching arrays establish faithful pipeline integration for these two development references only, not correctness of physical modeling or held-out forecast performance.')
    (REPORT/'production-reproduction-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],errors=errors,cases=[dict(object=r['object'],passed=r['passed'],array_max_differences={k:v['max_absolute_difference'] for k,v in r['comparisons'].items()}) for r in rows]),indent=2))
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
