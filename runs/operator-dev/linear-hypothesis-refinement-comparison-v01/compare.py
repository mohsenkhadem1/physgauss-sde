"""Exactly three prospective no-fit comparisons after finite terminal success."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.campaign_hypothesis_dynamics import validate_protocol,validate_clock,checked
from experiments.campaign_hypothesis_inputs import known_hardware_from_public
from experiments.campaign_fixed_actuator_clock import FixedActuatorClock,KNOWN_ROLE
from experiments.campaign_linear_hypothesis import LinearHypothesisDynamics
spec=importlib.util.spec_from_file_location('fixed_baseline_metrics',ROOT/'runs/operator-dev/linear-hypothesis-smoke-comparison-v01/compare.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
DECLARATION=ROOT/'runs/operator-dev/linear-hypothesis-refinement-comparison-v01/declaration.json'
sha=common.sha;write=common.write;rmse=common.rmse

def run(cell_id):
    d=json.loads(DECLARATION.read_text());row=next(r for r in d['cells'] if r['id']==cell_id)
    for rel,digest in d['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    protocol=validate_protocol(row['protocol_path'],row['protocol_sha256'])
    finite_root=ROOT/protocol['output'];execution=json.loads((finite_root/'execution.json').read_text())
    summary=json.loads((finite_root/'summary.json').read_text())
    hz=protocol['integration_hz']
    assert execution['terminal'] and execution['completed'] and execution['completed_steps']==3*hz
    assert summary['completed'] and summary['frames']==61 and summary['steps']==3*hz
    assert sha(finite_root/'trajectory.npz')==summary['trajectory_sha256']
    if any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2 required')
    torch.set_num_threads(2)
    output=ROOT/row['output'];output.mkdir(parents=True,exist_ok=False)
    write(output/'execution.json',dict(pid=os.getpid(),terminal=False,completed=False,CPU_threads=2,GPU_used=False))
    try:
        started=time.monotonic()
        hardware=known_hardware_from_public(json.loads(checked(protocol['public_hardware']).read_text()))
        hypothesis=json.loads(checked(protocol['realized_hypothesis']).read_text())
        with np.load(checked(protocol['known_commands']),allow_pickle=False) as f:
            clock=FixedActuatorClock(actuator_hz=120,information_role=KNOWN_ROLE,**{k:f[k].copy() for k in f.files})
        assert clock.public_manifest()==protocol['public_command_manifest']==summary['public_command_manifest']
        refined=validate_clock(clock,hz,len(hardware.points),len(hardware.handles))
        model=LinearHypothesisDynamics.from_hypothesis(hardware,hypothesis)
        assembled=time.monotonic();prediction=model.predict(clock,integration_hz=hz);predicted=time.monotonic()
        # Frozen model has completed before any finite q/v array enters memory.
        with np.load(finite_root/'trajectory.npz',allow_pickle=False) as f:
            finite={k:f[k].copy() for k in ['times_s','actual_state_times_s','positions_m','velocities_point_chord_m_s','force_step_commands_N']}
        assert np.array_equal(finite['times_s'],clock.output_times_s)
        assert np.array_equal(finite['force_step_commands_N'],refined.handle_forces_world_N)
        np.testing.assert_allclose(prediction['actual_state_times_s'],finite['actual_state_times_s'],atol=1e-12,rtol=0)
        q=prediction['positions_m'].numpy();v=prediction['velocities_point_chord_m_s'].numpy()
        assert q.shape==finite['positions_m'].shape==(61,len(hardware.points),3)
        error=q-finite['positions_m'];verror=v-finite['velocities_point_chord_m_s']
        finite_motion=rmse(finite['positions_m']-hardware.points);linear_motion=rmse(q-hardware.points)
        arrays={k:value.numpy() for k,value in prediction.items() if isinstance(value,torch.Tensor)}
        arrays['requested_output_times_s']=prediction['requested_output_times_s']
        np.savez_compressed(output/'predictions.npz',**arrays)
        result=dict(id=cell_id,cell_id=protocol['cell_id'],boundary_kind=hypothesis['boundary_kind'],frames_compared=61,integration_hz=hz,
            position_coordinate_RMSE_m=rmse(error),position_maximum_absolute_coordinate_error_m=float(np.max(np.abs(error))),
            position_maximum_node_vector_error_m=float(np.max(np.linalg.norm(error,axis=-1))),
            velocity_coordinate_RMSE_m_s=rmse(verror),velocity_maximum_absolute_coordinate_error_m_s=float(np.max(np.abs(verror))),
            velocity_maximum_node_vector_error_m_s=float(np.max(np.linalg.norm(verror,axis=-1))),
            finite_motion_coordinate_RMS_m=finite_motion,linear_motion_coordinate_RMS_m=linear_motion,
            linear_to_finite_motion_RMS_ratio=linear_motion/finite_motion,error_to_finite_motion_RMS_ratio=rmse(error)/finite_motion,
            finite_J=common.jacobians(finite['positions_m'],hardware.points,hardware.tets),linear_J=common.jacobians(q,hardware.points,hardware.tets),
            finite_patch_distance=common.patch_defect(finite['positions_m'],hardware.points,[h.ordered_patch_nodes for h in hardware.handles]),
            linear_patch_distance=common.patch_defect(q,hardware.points,[h.ordered_patch_nodes for h in hardware.handles]),
            runtime=dict(baseline_construction_s=assembled-started,baseline_predict_s=predicted-assembled,
                baseline_construction_plus_prediction_s=predicted-started,finite_logged_elapsed_s=summary['elapsed_s'],comparison_is_controlled_benchmark=False),
            cached_effective_matrix_factorizations=model.factorizations,exact_repeated_force_bytes_and_public_clock_match=True,
            same_material_full_hardware_geometry_and_support_law=True,fitting_or_hyperparameter_selection_performed=False,
            prediction_sha256=sha(output/'predictions.npz'),finite_trajectory_sha256=sha(finite_root/'trajectory.npz'),
            finite_summary_sha256=sha(finite_root/'summary.json'),protocol_sha256=row['protocol_sha256'],
            declaration_sha256=sha(DECLARATION),source_sha256=d['source_sha256'],raw_prediction_storage='LOCAL_ONLY',
            limitation='Single matching-rate engineering comparison; no finite-SO3, continuum, nonlinear-exact, production-admission or predictive-generalization claim.')
        for rel,digest in d['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
        write(output/'comparison.json',result)
        write(output/'execution.json',dict(pid=os.getpid(),terminal=True,completed=True,CPU_threads=2,GPU_used=False,comparison_sha256=sha(output/'comparison.json')))
        print(json.dumps({k:result[k] for k in ['id','position_coordinate_RMSE_m','velocity_coordinate_RMSE_m_s','linear_to_finite_motion_RMS_ratio','runtime']}),flush=True)
    except BaseException as error:
        write(output/'failure.json',dict(type=type(error).__name__,message=str(error)))
        write(output/'execution.json',dict(pid=os.getpid(),terminal=True,completed=False,CPU_threads=2,GPU_used=False));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cell',required=True,choices=['pointpin240','pointpin480','moving120']);run(parser.parse_args().cell)
