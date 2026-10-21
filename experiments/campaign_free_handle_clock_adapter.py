"""Reanalyze frozen FREE-handle capture after an explicit clock-adapter audit.

World/manager dt GETTERS in this SDK retain nominal double precision, whereas
the recorded native world clock advances float32-rounded steps. The original
1ns double-dt analysis failure is preserved. This adapter changes exactly that
clock/reference block, with no simulation, no new forces, and no relaxed force
tolerance. Generated analysis lives in a fresh separate folder.
"""
import argparse
import difflib
import inspect
import json
import os
from pathlib import Path
import numpy as np
try:
    import campaign_free_handle_control as base
except ImportError:
    from experiments import campaign_free_handle_control as base


def validated_native_clock(data):
    times=np.asarray(data['world_times_s'],dtype=float)
    reported=np.asarray(data['world_dt_s'],dtype=float)
    manager=np.asarray(data['manager_dt_s'],dtype=float)
    if times.shape!=(len(reported)+1,) or reported.shape!=manager.shape:
        raise ValueError('Incomplete native clock arrays')
    np.testing.assert_array_equal(reported,manager)
    measured=np.diff(times)
    expected=reported.astype(np.float32).astype(np.float64)
    tolerance=32*np.finfo(float).eps*max(1.,float(abs(times).max()))
    step_error=float(abs(measured-expected).max())
    if np.any(measured<=0) or step_error>tolerance:
        raise ValueError('Measured native step interval differs from float32-rounded nominal dt')
    elapsed=times-times[0]
    cumulative=float(abs(elapsed-np.r_[0.,np.cumsum(expected)]).max())
    return {'measured_step_dt_s':measured,'maximum_step_error_vs_float32_getter_dt_s':step_error,
            'maximum_cumulative_error_vs_float32_getter_dt_s':cumulative,
            'nominal_getter_elapsed_error_s':float(abs(elapsed[-1]-reported.sum())),
            'maximum_measured_dt_minus_nominal_getter_s':float(abs(measured-reported).max()),
            'roundoff_tolerance_s':tolerance,
            'reference_clock':'Recorded native per-step world-time differences, independently verified against float32-rounded World/manager nominal getters. The getters are not called actual backend dt.'}


def derive_analyzer():
    original=inspect.getsource(base.analyze_cell)
    before="""    dt=data['world_dt_s'];elapsed=data['world_times_s']-data['world_times_s'][0]
    clock_error=float(abs(elapsed-np.r_[0.,np.cumsum(dt)]).max())
    if clock_error>1e-9:raise ValueError('Actual native dt/clock mismatch')
"""
    after="""    clock_audit=validated_native_clock(data)
    dt=clock_audit.pop('measured_step_dt_s')
    elapsed=data['world_times_s']-data['world_times_s'][0]
    clock_error=clock_audit['maximum_cumulative_error_vs_float32_getter_dt_s']
"""
    if original.count(before)!=1:raise ValueError('Frozen clock block no longer uniquely matches')
    derived=original.replace(before,after)
    field="'clock_error_s':clock_error,"
    if derived.count(field)!=1:raise ValueError('Frozen clock-result field no longer uniquely matches')
    derived=derived.replace(field,"'clock_error_s':clock_error,'clock_audit':clock_audit,")
    namespace={**vars(base),'validated_native_clock':validated_native_clock}
    exec(compile(derived,'free-handle-explicit-clock-adapter','exec'),namespace)
    return namespace['analyze_cell'],original,derived


def run(capture_root,output):
    capture_root=Path(capture_root);output=Path(output)
    protocol=json.loads((capture_root/'protocol.json').read_text())
    if base.sha(base.__file__)!=protocol['source_sha256']:
        raise ValueError('Capture worker source differs; do not apply this adapter')
    ledger=json.loads((capture_root/'execution.json').read_text())
    if not ledger['completed']:raise ValueError('Original capture driver still active/incomplete')
    output.mkdir(parents=True,exist_ok=False)
    analyzer,original,derived=derive_analyzer()
    (output/'adapter-source.py').write_bytes(Path(__file__).read_bytes())
    (output/'original-analyzer.py').write_text(original)
    (output/'derived-analyzer.py').write_text(derived)
    (output/'analyzer.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),derived.splitlines(True),fromfile='frozen-original-analyzer',tofile='explicit-measured-clock-adapter')))
    adapter_protocol={'capture_root':str(capture_root),'capture_protocol_sha256':base.sha(capture_root/'protocol.json'),
        'capture_execution_sha256':base.sha(capture_root/'execution.json'),
        'capture_source_sha256':protocol['source_sha256'],'adapter_source_sha256':base.sha(__file__),
        'derived_analyzer_sha256':base.sha(output/'derived-analyzer.py'),
        'modification':'Exact 3-line clock/reference block plus persisted clock-audit field. Force, mass, hardware, flags,100ppm/rotation gates and all physics remain unchanged.',
        'no_simulation_launched':True,'original_failures_preserved':True}
    base.write_json(output/'protocol.json',adapter_protocol)
    rows=[]
    for cell in protocol['config']['cells']:
        source=capture_root/cell['id'];target=output/cell['id'];target.mkdir()
        status=json.loads((source/'worker-status.json').read_text())
        failure=json.loads((source/'failure.json').read_text())
        if not status['terminal'] or status['step']!=3*cell['hz']-1 or failure['error']!='Actual native dt/clock mismatch':
            raise ValueError('Adapter only handles the recorded post-capture double-dt analysis failure')
        for name in ['native-state.json','trajectory.npz']:
            os.link(source/name,target/name)
        row=analyzer(target,cell,protocol)
        row.update(adapter_source_sha256=base.sha(__file__),capture_complete_verified=True,
                   original_capture_failure_sha256=base.sha(source/'failure.json'),
                   original_worker_status_sha256=base.sha(source/'worker-status.json'))
        base.write_json(target/'analysis.json',row);rows.append(row)
    result={'protocol':adapter_protocol,'all_cells_passed':all(row['passed'] for row in rows),'cells':rows,
            'qualification':protocol['config']['qualification'],
            'interpretation':'The original capture analysis failed on a nominal-double clock assumption. This separate explicit adapter proves complete captures and analyzes recorded native intervals; no rerun or hidden failure replacement.'}
    base.write_json(output/'analysis.json',result)
    print(json.dumps({'all_cells_passed':result['all_cells_passed'],'cells':[{k:row[k] for k in ['cell',
        'fitted_step_impulse_gain','maximum_relative_cumulative_momentum_error',
        'positions_coordinate_rmse_vs_exact_held_force_m','positions_coordinate_rmse_vs_kick_drift_m',
        'fitted_position_integration_theta','passed']} for row in rows]},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root',default='runs/gaussian-campaign-v01/native-free-handle-v01')
    parser.add_argument('--output',default='runs/gaussian-campaign-v01/native-free-handle-v01/clock-analysis-v01')
    args=parser.parse_args();run(args.capture_root,args.output)
