"""Isolated driver for the separately frozen surface-contact extension.

The original pipeline source is reused in a private module namespace. Its global
BASE/REPORT/model_class/execute bindings are replaced only in that namespace;
the imported original pipeline and all original files/caches remain unchanged.
Public scene functions come from the separately declared surface_scenarios module;
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
import multiprocessing
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'runs/coupled-surface-v13'
REPORT=ROOT/'reports/coupled-surface-v13'
_IMPL=None
_CLASSES={}


def model_class(family):
    from scripts.coupled_contact_v13.surface_contact import SurfaceFEMSystem,with_surface_contact
    if family in ('fem','oracle'):return SurfaceFEMSystem
    if family not in _CLASSES:
        if family=='native':
            from scripts.coupled_contact_v13.native_adapter import NativeSpringSystem
            base=NativeSpringSystem
        elif family=='rigid':
            from scripts.coupled_contact_v13.rigid import RigidSystem
            base=RigidSystem
        else:raise ValueError(family)
        _CLASSES[family]=with_surface_contact(base)
    return _CLASSES[family]


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
        from scripts.coupled_contact_v13 import surface_scenarios
        for key in ('build','run','command','shared_node_indices'):
            setattr(module,key,getattr(surface_scenarios,key))
        module.model_class=model_class;module.execute=execute
        original_reference_quality=module.reference_quality
        from scripts.coupled_contact_v13.surface_body_quality import augment
        def coupled_reference_quality(object_id,scenario):
            return augment(original_reference_quality(object_id,scenario),BASE,object_id,scenario,
                           module.protocol().get('body_temporal_coordinate_rms_limit_m',.00025))
        module.reference_quality=coupled_reference_quality
        _IMPL=module
    return _IMPL


def simulate_job(job):
    # Job path is a protocol-generated relative key. Reject accidental cross-run
    # absolute paths or parent traversal before any cache read or write.
    path=Path(job['path'])
    if path.is_absolute() or '..' in path.parts:raise ValueError('Job path must stay within the surface campaign')
    return implementation().simulate_job(job)


def execute(jobs,workers):
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=[]
        for row in pool.map(simulate_job,jobs):
            rows.append(row)
            print(json.dumps({k:row[k] for k in ('success','job','seconds')}),flush=True)
    return rows


def main():
    implementation().main()


if __name__=='__main__':main()
