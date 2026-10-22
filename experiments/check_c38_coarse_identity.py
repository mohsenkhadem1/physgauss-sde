"""Read-only CPU identity preflight for reusing the original14 coarse controls.

No solver, GPU, observation or response evaluation. Equality means identical
shape, dtype and bytes; no numerical tolerance substitutes for array identity.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from experiments.campaign_c38_static_family import build_geometry
from experiments.campaign_c38_supported_grading import SupportedBoundaryGradedGeometry
from experiments.campaign_p2_consistent_transient import scalar_mass
ROOT=Path(__file__).resolve().parents[1]
SUMMARY='runs/operator-dev/c38-static-prequalification-v01/completed-summary.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def equal(a,b):
    a=np.ascontiguousarray(a);b=np.ascontiguousarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def execute(destination):
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Fresh preflight output required')
    summary_path=ROOT/SUMMARY;summary=json.loads(summary_path.read_text())
    if not summary['completed'] or len(summary['rows'])!=14:raise ValueError('All14 original controls required')
    result=dict(schema='C38-coarse4-array-identity-v01',completed=False,passed=False,original_summary={'path':SUMMARY,'sha256':sha(summary_path)},equality='Exact shape/dtype/bytes, no tolerance',rows=[],solver_calls=0,GPU_calls=0,observations_drawn=False)
    for prior in summary['rows']:
        case=prior['case'];r=prior['result'];report_path=ROOT/r['path']
        if sha(report_path)!=r['sha256']:raise ValueError('Original full coarse-control report hash mismatch')
        report=json.loads(report_path.read_text())
        if report['case']!=case or report['coarse_CPU_GPU']['passed'] is not True:raise ValueError('Original independent control did not pass')
        rule_rows=[]
        for rule in ('keast24','exact80'):
            a=build_geometry((4,2,2),case['young_Pa'],case['support'],case['Fz_N'],rule=rule)
            b=SupportedBoundaryGradedGeometry((4,2,2),case['young_Pa'],case['support'],case['Fz_N'],rule=rule)
            fields=('rest','tets','elements','faces','mask','free','linear_gradients','volumes','bary','weights','N','load','support_masks','support_active')
            checks={name:equal(getattr(a,name),getattr(b,name)) for name in fields}
            for name in ('body_N','surface_N','total_N','body_integral_weights_m3'):
                checks['compiled_'+name]=equal(a.public_force_components[name],b.public_force_components[name])
            ma,mb=scalar_mass(a),scalar_mass(b)
            for name in ('data','indices','indptr'):checks['full_mass_CSR_'+name]=equal(getattr(ma,name),getattr(mb,name))
            # Exact80 implementations differ only by duplicated guard storage:
            # both evaluate identical10 vertex/edge +80 integration locations.
            sa=np.unique(np.vstack([a.check_bary,a.bary]),axis=0);sb=np.unique(np.vstack([b.check_bary,b.bary]),axis=0)
            checks['effective_J_guard_locations']=equal(sa,sb)
            rule_rows.append(dict(rule=rule,required_array_checks=checks,all_required_bytes_identical=all(checks.values()),raw_check_bary_storage_identical=equal(a.check_bary,b.check_bary),raw_body_integral_storage_identical=equal(a.body_integral_weights,b.body_integral_weights),auxiliary_storage_caveat='Exact80 adds duplicate checking locations in graded implementation; legacy body-integral cache uses quadrature arithmetic while actual compiled bodyforce and full mass are identical. No whole-object identity claim.'))
        result['rows'].append(dict(case=case,original_control_report=r,original_control=report['coarse_CPU_GPU'],rules=rule_rows,passed=all(v['all_required_bytes_identical'] for v in rule_rows)))
    result.update(completed=True,passed=all(r['passed'] for r in result['rows']),all14_original_independent_controls_hash_bound=True,entire_object_array_storage_identical=False,scope='Same physical geometry, quadrature, constraints, full forces/mass and effective guard locations. Auxiliary cache/duplicate-guard storage differences explicitly retained; root decides reuse. No solver/backend validation from this check alone.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();r=execute(args.output);print({'completed':r['completed'],'passed':r['passed'],'cases':len(r['rows'])});raise SystemExit(0 if r['passed'] else 1)
