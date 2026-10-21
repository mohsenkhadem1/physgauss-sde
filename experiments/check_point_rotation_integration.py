"""Full-size development smoke on OLD object0; never a held performance test.

New-cohort data is never read. All old-model evaluation evidence stays intact.
The output is an execution check on a discrepancy-fitting object, not evidence
of generalization, calibration or policy benefit.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import torch
try:
    import run_point_rotation_analysis as r
    import run_point_prior_analysis as old
except ImportError:
    from experiments import run_point_rotation_analysis as r
    from experiments import run_point_prior_analysis as old


def run(output):
    out=r.resolve(output);out.mkdir(parents=True,exist_ok=False)
    status=dict(pid=os.getpid(),terminal=False,completed=False,new_cohort_values_opened=False,
        scientific_performance_claim=False)
    r.write(out/'terminal.json',status);started=time.monotonic()
    try:
        oldp,_,oldplan=old.validate_execution('runs/operator-dev/point-prior-analysis-v01/execution-protocol.json',
            'fabad32cb79a94bde93890506f8862f0ea00d9535f001bfad026236aaa08242c',check_environment=False)
        objects=old.eligible_objects(oldplan,'CALIBRATION',execution_sha256='',method_sha256=oldp['method']['sha256'])
        pairs=old.capture_pairs(oldplan,objects)
        method=r.read(r.checked(dict(path=r.METHOD_PATH,sha256=r.METHOD_SHA256)))
        newplan=r.read(r.checked(method['cohort_plan']))
        librec=dict(path='runs/gaussian-campaign-v01/point-rotation-library-v02/summary.json',
            sha256='d432f1bbae2bcda41a951051553047267b60454611856cfbadf3233fe55bfb97')
        library=r.read(r.checked(librec));fit=r.read(r.checked(method['discrepancy_fit']))
        sources=r.source_map(newplan,library,fit)
        sources['experiments/check_point_rotation_integration.py']=r.sha(__file__)
        declaration=dict(scope='DEVELOPMENT EXECUTION CHECK ONLY; OLD fitting object0, no new cohort values',
            method=dict(path=r.METHOD_PATH,sha256=r.METHOD_SHA256),library_summary=librec,
            source_sha256=sources,device='cuda:0',raw_outputs_LOCAL=True)
        r.write(out/'declaration.json',declaration);digest=r.sha(out/'declaration.json');torch.set_num_threads(2)
        cache=r.read(r.checked(library['plan']))['cache']
        packed,public,_,_=r.load_support_basis_cache(cache['path'],cache['sha256'])
        libraries=r.load_rotation_libraries(librec['path'],librec['sha256'])
        model=r.load_discrepancy(method['discrepancy_fit'])
        queries=r.read(r.checked(oldplan['public_acquired_queries']))
        candidates=r.make_queries(public['rest_points_m'],public['surface_faces'],frame_indices=(24,),
            offsets_deg=((-60.,10.),(-20.,10.),(20.,10.),(60.,10.)),grid_size=6)
        context=(r.PointRotationForward(packed,np.asarray(public['rest_points_m'],dtype=np.float64)),
            public,libraries,model,queries,candidates,method)
        obj={**objects[0],'role':'DEVELOPMENT_SMOKE'}
        r.analyze_episode(obj,pairs[0],context,declaration,digest,out/'old-object-000')
        status.update(terminal=True,completed=True,elapsed_s=time.monotonic()-started,
            result=r.record(out/'old-object-000/result.json'))
        r.write(out/'terminal.json',status);return status
    except BaseException as exc:
        status.update(terminal=True,completed=False,elapsed_s=time.monotonic()-started,error=repr(exc))
        r.write(out/'terminal.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    print(__import__('json').dumps(run(a.output)))
