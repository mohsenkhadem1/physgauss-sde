"""Additive operational-recovery view of the frozen cohort evaluation.
Original failed native launch attempts remain sealed and separately scoreable.
No numerical method, target selection, metric or aggregation change is permitted.
"""
import argparse,json
from pathlib import Path
from scripts.compositional_dynamics_v17 import cohort_geometry_evaluation as frozen
from scripts.compositional_dynamics_v17 import native_cohort_recovery as recovery
ROOT=frozen.ROOT;REPORT=frozen.REPORT;sha=frozen.sha;write=frozen.write


def committed(bindings):
    protocol,rows,evidence=frozen.committed(bindings)
    plan=recovery.guard();manifest_path=recovery.OUT/'recovery-manifest.json';manifest=json.loads(manifest_path.read_text())
    if not manifest['terminal'] or manifest['plan_sha256']!=sha(recovery.PLAN) or manifest['expected']!=8 or len(manifest['rows'])!=8:raise ValueError('All eight recovery attempts must commit before target access')
    evidence[str(manifest_path.relative_to(ROOT))]=sha(manifest_path);evidence[str(recovery.PLAN.relative_to(ROOT))]=sha(recovery.PLAN)
    expected={(f'world-{i}',role)for i in range(4)for role in ('forecast-40mm','forecast-80mm')}
    if {(r['world_id'],r['role'])for r in manifest['rows']}!=expected:raise ValueError('Exact fixed recovery query census required')
    mappings={(r['world_id'],r['role']):r for r in manifest['rows']};worlds={r['world_id']:r for r in plan['worlds']};result=[]
    for row in rows:
        if row['method']=='ours':result.append(row);continue
        name,role=row['world_id'],row['role'];mapping=mappings[name,role]
        old=ROOT/mapping['original_commit'];path=ROOT/mapping['recovery_commit'];world=worlds[name]
        expected_old=ROOT/world['original_output']/'forecasts'/role/'prediction-commit.json';expected_new=ROOT/world['recovery_output']/'forecasts'/role/'prediction-commit.json'
        if old!=expected_old or path!=expected_new or sha(old)!=mapping['original_commit_sha256'] or sha(path)!=mapping['recovery_commit_sha256']:raise ValueError('Recovery mapping integrity mismatch')
        record=json.loads(path.read_text());worldplan=ROOT/world['recovery_plan']
        if row['record']['completed'] or record.get('protocol_sha256')!=bindings['protocol_sha'] or record.get('method_freeze_sha256')!=bindings['freeze_sha'] or record.get('world_id')!=name or record.get('role')!=role or record.get('recovery_plan_sha256')!=sha(recovery.PLAN) or record.get('native_world_plan_sha256')!=sha(worldplan) or record.get('original_prediction_commit_sha256')!=sha(old) or not isinstance(record.get('completed'),bool) or record['completed']!=mapping['completed']:raise ValueError('Recovery identity mismatch')
        evidence[str(path.relative_to(ROOT))]=sha(path);evidence[str(worldplan.relative_to(ROOT))]=sha(worldplan);array=path.with_name('prediction.npz')
        if record['completed']:
            complete=path.with_name('complete.json');status=json.loads(complete.read_text())
            if sha(complete)!=record['native_complete_sha256'] or not status['completed'] or status['plan_sha256']!=sha(worldplan) or sha(array)!=record['prediction_sha256'] or status['prediction_sha256']!=sha(array):raise ValueError('Recovered native prediction changed')
            evidence[str(complete.relative_to(ROOT))]=sha(complete);evidence[str(array.relative_to(ROOT))]=sha(array)
        result.append(dict(row,record=record,array=array))
    return protocol,result,evidence


def main(bindings):
    import importlib.util
    source=Path(frozen.__file__);spec=importlib.util.spec_from_file_location('_recovery_frozen_evaluation',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    output=REPORT/'fresh-cohort-geometry-recovery-results.json'
    if output.exists():raise ValueError('Preserve recovered evaluation')
    module.committed=committed
    def save(path,data):
        if path!=REPORT/'fresh-cohort-geometry-results.json':raise ValueError('Unexpected frozen output route')
        data['operational_recovery']=dict(plan_sha256=sha(recovery.PLAN),original_native_attempts='All four fits stopped before optimization at missing GPU visibility guard; original eight failed forecast commits preserved.',repair='Explicit requiredGPU0 environment and fresh output/cwd routing only; same public data, numerical sources, settings, seeds and originaldeadline.',original_frozen_evaluator_sha256=sha(source),wrapper_sha256=sha(__file__))
        data['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(__file__)
        write(output,data)
    module.write=save;module.main(bindings)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['protocol-path','freeze-path','approval-path','protocol-sha','freeze-sha']:parser.add_argument('--'+key,required=True)
    main(vars(parser.parse_args()))
