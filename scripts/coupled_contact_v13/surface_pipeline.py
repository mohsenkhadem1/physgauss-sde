"""Isolated driver for the separately frozen surface-contact extension.

The original pipeline source is reused in a private module namespace. Its global
BASE/REPORT/model_class/execute bindings are replaced only in that namespace;
the imported original pipeline and all original files/caches remain unchanged.
Public scene functions come from the separately declared engineered_scenarios module;
matched vertex-contact ablations must use these same scenes and recompute their
calibration libraries (the historical k=4800 library is not compatible).
Each spawned worker initializes these bindings independently, so no process can
silently fall back to the vertex-contact run directory or model classes.
The original campaign deadline remains binding; this is not a new time budget.
"""
from pathlib import Path
import concurrent.futures
import importlib.util
import json
import hashlib
import time
import traceback
import multiprocessing
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'runs/coupled-surface-v13'
REPORT=ROOT/'reports/coupled-surface-v13'
_IMPL=None
_CLASSES={}


def contact_settings(p):
    spacing=float(p['surface_spacing_m'])
    level=int(p['friction_quadrature_level'])
    if spacing<=0 or level not in (0,1,2):raise ValueError('Invalid frozen contact settings')
    return dict(surface_spacing_m=spacing,friction_quadrature_level=level)


def model_class(family):
    from scripts.coupled_contact_v13.physics import FEMSystem
    from scripts.coupled_contact_v13.clipped_surface_contact import with_clipped_surface_contact
    from scripts.coupled_contact_v13.consistent_mass import with_consistent_mass
    p=implementation().protocol()
    settings=contact_settings(p);key=(family,settings['surface_spacing_m'],settings['friction_quadrature_level'])
    if key not in _CLASSES:
        if family in ('fem','oracle'):base=FEMSystem
        elif family=='native':
            from scripts.coupled_contact_v13.native_adapter import NativeSpringSystem
            base=NativeSpringSystem
        elif family=='rigid':
            from scripts.coupled_contact_v13.rigid import RigidSystem
            base=RigidSystem
        else:raise ValueError(family)
        parent=with_consistent_mass(with_clipped_surface_contact(base))
        _CLASSES[key]=type('FrozenSurface'+family,(parent,),dict(surface_spacing=settings['surface_spacing_m'],friction_quadrature_level=settings['friction_quadrature_level']))
    return _CLASSES[key]


def simulation_settings(job,p):
    reference=job['reference']
    override=p.get('reference_settings_by_scenario',{}).get(job['scenario'],{}) if reference else {}
    cells=list(override.get('cells',p['reference_cells']) if reference else p['coarse_cells'])
    dt=float(override.get('dt',p['dt_reference']) if reference else p['dt_coarse'])
    if job.get('spatial',False):
        if not reference or job.get('refinement',False):raise ValueError('Spatial companion must be a base-step reference')
        primary_cells=cells
        cells=list(override['companion_cells'])
        if cells==primary_cells:raise ValueError('Spatial companion must differ from primary mesh')
    if reference and job.get('refinement',False):dt*=.5
    if len(cells)!=3 or any(int(n)!=n or n<1 for n in cells) or dt<=0:raise ValueError('Invalid frozen discretization')
    return dict(cells=cells,dt=dt,duration_seconds=p['duration_seconds'],
                scene_source='engineered_scenarios.py',mass_operator='consistent-P1',
                contact_formulation='clipped-normal-refined-friction-sparse-sphere',**contact_settings(p))


def implementation():
    global _IMPL
    if _IMPL is None:
        name='scripts.coupled_contact_v13._surface_isolated_pipeline'
        source=ROOT/'scripts/coupled_contact_v13/pipeline.py'
        spec=importlib.util.spec_from_file_location(name,source)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
        module.BASE=BASE;module.REPORT=REPORT
        from scripts.coupled_contact_v13 import engineered_scenarios
        for key in ('build','run','command','shared_node_indices'):
            setattr(module,key,getattr(engineered_scenarios,key))
        module.model_class=model_class;module.execute=execute;module.simulate_job=simulate_job;module.reference_jobs=reference_jobs
        original_check_freeze=module.check_freeze
        def check_freeze():
            original_check_freeze()
            from scripts.coupled_contact_v13.surface_scope import check_qualification
            check_qualification(module.REPORT,ROOT)
        module.check_freeze=check_freeze
        original_reference_quality=module.reference_quality
        from scripts.coupled_contact_v13.surface_body_quality import augment
        def coupled_reference_quality(object_id,scenario):
            body=augment(original_reference_quality(object_id,scenario),BASE,object_id,scenario,
                           module.protocol().get('body_temporal_coordinate_rms_limit_m',.00025))
            from scripts.coupled_contact_v13.surface_spatial_quality import augment as spatial_augment
            return spatial_augment(body,BASE,object_id,scenario)
        module.reference_quality=coupled_reference_quality
        _IMPL=module
    return _IMPL


def reference_jobs(split):
    p=implementation().protocol()
    jobs=[dict(path=f'reference/{o["id"]}/{scene}',family='fem',scenario=scene,scale=o['young'],friction=o['friction'],reference=True)
          for o in p['objects'] if o['split']==split for scene in ['squeeze']+p['forecast_scenarios']]
    temporal=[dict(j,path=j['path'].replace('reference/','reference-refinement/'),refinement=True) for j in jobs]
    spatial=[dict(j,path=j['path'].replace('reference/','reference-spatial/'),spatial=True) for j in jobs]
    return jobs+temporal+spatial


def simulate_job(job):
    relative=Path(job['path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Job path must stay within the surface campaign')
    m=implementation();p=m.protocol();settings=simulation_settings(job,p)
    protocol_sha=hashlib.sha256((m.REPORT/'protocol.json').read_bytes()).hexdigest()
    path=m.BASE/relative;path.mkdir(parents=True,exist_ok=True)
    if (path/'complete.json').exists():
        cached=json.loads((path/'complete.json').read_text())
        if cached.get('job')!=job or cached.get('simulation_settings')!=settings or cached.get('protocol_sha256')!=protocol_sha:
            raise RuntimeError('Surface cache provenance/settings mismatch: '+str(path))
        return cached
    start=time.monotonic()
    status=dict(job=job,simulation_settings=settings,protocol_sha256=protocol_sha)
    try:
        if m.remaining()<=0:raise RuntimeError('Campaign deadline reached')
        result=m.run(job['scenario'],job['scale'],job['friction'],tuple(settings['cells']),settings['dt'],settings['duration_seconds'],model_class(job['family']))
        m.save_result(path,result)
        status.update(success=True,diagnostics=json.loads((path/'diagnostics.json').read_text()))
    except Exception as exc:
        status.update(success=False,error=repr(exc),traceback=traceback.format_exc())
    status['seconds']=time.monotonic()-start;m.write(path/'complete.json',status)
    return status


def scheduled_jobs(jobs,p):
    """Deterministic largest-work-first scheduling; no simulation input changes.

    Cell count times integration steps is a cost proxy, not a runtime guarantee.
    Original index breaks ties; execute restores original caller result order.
    """
    def cost(item):
        settings=simulation_settings(item[1],p)
        nx,ny,nz=settings['cells']
        return nx*ny*nz*settings['duration_seconds']/settings['dt']
    return sorted(enumerate(jobs),key=lambda item:(-cost(item),item[0]))


def execute(jobs,workers):
    jobs=list(jobs)
    if not jobs:return []
    ordered=scheduled_jobs(jobs,implementation().protocol());rows=[None]*len(jobs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for (index,_),row in zip(ordered,pool.map(simulate_job,[job for _,job in ordered])):
            rows[index]=row
            print(json.dumps({k:row[k] for k in ('success','job','seconds')}),flush=True)
    return rows


def main():
    implementation().main()


if __name__=='__main__':main()
