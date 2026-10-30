"""Reuse the fair world-coordinate renderer in an isolated surface-study namespace."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
BASE=ROOT/'runs/coupled-surface-v13';REPORT=ROOT/'reports/coupled-surface-v13'
FIXED_CASES=[['surface_test0','curved'],['surface_test3','moving_sphere'],['surface_test7','edge_drop']]


def case_qualification(object_id,scenario):
    for split in ('development','reserved'):
        path=REPORT/f'{split}-results.json'
        if path.exists():
            for row in json.loads(path.read_text())['cases']:
                if row['object']==object_id and row['scenario']==scenario:
                    return {k:row.get(k) for k in ('reference_qualified','calibration_reference_qualified','joint_reference_qualified','reference_qualification_reasons','calibration_qualification_reasons')}
    return dict(joint_reference_qualified=False,reference_qualification_reasons=['Qualification unavailable'])


def reference_label_for_case(object_id,scenario):
    protocol_path=REPORT/'protocol.json'
    if protocol_path.exists():
        protocol=json.loads(protocol_path.read_text())
        if scenario in protocol.get('stress_forecast_scenarios',[]):
            return 'Reference FEM: Exploratory stress\n(development unqualified)'
    quality=case_qualification(object_id,scenario)
    return 'Reference FEM\n(checks passed)' if quality['joint_reference_qualified'] else 'Reference FEM\n(checks failed / unavailable)'


def implementation():
    name='scripts.coupled_contact_v13._surface_isolated_renderer'
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/coupled_contact_v13/render.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    from scripts.coupled_contact_v13.engineered_scenarios import build,command
    # Transform only the reference title in the private renderer; original source stays unchanged.
    source=(ROOT/'scripts/coupled_contact_v13/render.py').read_text()
    old=next(line for line in source.splitlines() if line.strip().startswith('reference_label='))
    source=source.replace(old,"        reference_label=reference_label_for_case(object_id,scenario)",1)
    exec(compile(source,str(ROOT/'scripts/coupled_contact_v13/render.py'),'exec'),module.__dict__)
    module.reference_label_for_case=reference_label_for_case
    module.BASE=BASE;module.REPORT=REPORT;module.build=build;module.command=command
    return module


def main():
    protocol=json.loads((REPORT/'protocol.json').read_text())
    if protocol['video_cases']!=FIXED_CASES:raise ValueError('Surface fixed-video declaration differs from renderer')
    if protocol['fps']!=100:raise ValueError('Four-times-slower label requires100Hz source at25fps')
    for obj,scene in FIXED_CASES:
        reference=BASE/'reference'/obj/scene/'trajectory.npz';prediction=BASE/'evaluation'/obj/scene/'predictions.npz'
        if not reference.exists() or not prediction.exists():raise RuntimeError('Fixed video source unavailable: '+obj+'/'+scene)
        with np.load(reference) as ref:times=ref['times'].copy()
        if len(times)!=61 or not np.allclose(times,np.arange(61)/100,atol=1e-10,rtol=0):raise ValueError('Unexpected saved source timeline')
        for family in ('fem','native','rigid'):
            path=BASE/'prediction'/obj/scene/family/'trajectory.npz'
            if path.exists():
                with np.load(path) as arr:
                    if arr['times'].shape!=times.shape or not np.allclose(arr['times'],times,atol=1e-10,rtol=0):raise ValueError('Prediction timeline differs: '+str(path))
    module=implementation();module.main()
    path=REPORT/'video-provenance.json';rows=json.loads(path.read_text())
    expected={f'{obj}-{scene}.mp4' for obj,scene in FIXED_CASES}
    if {Path(row['path']).name for row in rows}!=expected:raise RuntimeError('Fixed surface clips omitted')
    for row in rows:
        row['surface_renderer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        row['shared_renderer_sha256']=hashlib.sha256((ROOT/'scripts/coupled_contact_v13/render.py').read_bytes()).hexdigest()
        row['scene_source_sha256']=hashlib.sha256((ROOT/'scripts/coupled_contact_v13/engineered_scenarios.py').read_bytes()).hexdigest()
        obj,scene=next((o,s) for o,s in FIXED_CASES if Path(row['path']).name==f'{o}-{s}.mp4')
        row['reference_qualification']=case_qualification(obj,scene)
        row['reference_display_label']=reference_label_for_case(obj,scene)
        row['forecast_cohort']='stress' if scene in protocol.get('stress_forecast_scenarios',[]) else 'primary'
        row['development_scope_qualified']=scene not in protocol.get('stress_forecast_scenarios',[])
        row['reference_settings']=protocol['reference_settings_by_scenario'][scene]
        row['primary_status_sha256']=hashlib.sha256((BASE/'reference'/obj/scene/'complete.json').read_bytes()).hexdigest()
        row['native_scope']='Author spring-force core with shared scene/contact/inertia/integration; not full PhysTwin'
        row['study']='Separately frozen consistent-mass clipped contact, shared world scale and camera; quarter-speed playback'
    module.write(path,rows)


if __name__=='__main__':main()
