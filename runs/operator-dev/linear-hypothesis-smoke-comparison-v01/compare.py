"""One fixed, no-fit small-strain comparison authorized before prediction."""
from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.campaign_hypothesis_dynamics import validate_protocol,checked
from experiments.campaign_hypothesis_inputs import known_hardware_from_public
from experiments.campaign_fixed_actuator_clock import FixedActuatorClock,KNOWN_ROLE
from experiments.campaign_linear_hypothesis import LinearHypothesisDynamics

DECLARATION=ROOT/'runs/operator-dev/linear-hypothesis-smoke-comparison-v01/declaration.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def rmse(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def jacobians(q,points,tets):
    ref=(points[tets[:,1:]]-points[tets[:,:1]]).transpose(0,2,1)
    edge=(q[:,tets[:,1:]]-q[:,tets[:,:1]]).transpose(0,1,3,2)
    J=np.linalg.det(edge@np.linalg.inv(ref))
    return dict(minimum=float(J.min()),maximum=float(J.max()),inverted_tetrahedron_frames=int((J<=0).sum()))
def patch_defect(q,points,patches):
    rows=[]
    for patch in patches:
        p=q[:,patch];r=points[patch]
        actual=np.linalg.norm(p[:,:,None]-p[:,None,:],axis=-1)
        reference=np.linalg.norm(r[:,None]-r[None,:],axis=-1)
        rows.append(float(np.max(np.abs(actual-reference))))
    return dict(maximum_pair_distance_defect_m=max(rows),each_handle_maximum_m=rows)

def run():
    d=json.loads(DECLARATION.read_text())
    for rel,digest in d['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    for record in d['completed_comparison_inputs'].values():assert sha(ROOT/record['path'])==record['sha256'],record['path']
    if any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2 required')
    torch.set_num_threads(2)
    protocol=validate_protocol(d['protocol']['path'],d['protocol']['sha256'])
    output=ROOT/d['output'];output.mkdir(parents=True,exist_ok=False)
    write(output/'execution.json',dict(pid=os.getpid(),terminal=False,completed=False,CPU_threads=2,GPU_used=False))
    try:
        wall=time.monotonic()
        hardware=known_hardware_from_public(json.loads(checked(protocol['public_hardware']).read_text()))
        spec=json.loads(checked(protocol['realized_hypothesis']).read_text())
        with np.load(checked(protocol['known_commands']),allow_pickle=False) as f:
            clock=FixedActuatorClock(actuator_hz=120,information_role=KNOWN_ROLE,**{k:f[k].copy() for k in f.files})
        assert clock.public_manifest()==protocol['public_command_manifest']
        model=LinearHypothesisDynamics.from_hypothesis(hardware,spec)
        assembled=time.monotonic();prediction=model.predict(clock,integration_hz=120);predicted=time.monotonic()
        # The finite response is opened only after the frozen baseline has
        # finished prediction. It is never an input to model construction.
        finite_root=ROOT/protocol['output'];summary=json.loads((finite_root/'summary.json').read_text())
        with np.load(finite_root/'trajectory.npz',allow_pickle=False) as f:
            finite={k:f[k].copy() for k in ['times_s','actual_state_times_s','positions_m','velocities_point_chord_m_s','force_step_commands_N']}
        assert summary['completed'] and summary['frames']==61 and summary['steps']==360
        assert summary['public_command_manifest']==clock.public_manifest()
        assert np.array_equal(finite['times_s'],clock.output_times_s)
        assert np.array_equal(finite['force_step_commands_N'],clock.handle_forces_world_N)
        np.testing.assert_allclose(prediction['actual_state_times_s'],finite['actual_state_times_s'],atol=1e-12,rtol=0)
        q=prediction['positions_m'].numpy();v=prediction['velocities_point_chord_m_s'].numpy()
        assert q.shape==finite['positions_m'].shape==(61,len(hardware.points),3)
        error=q-finite['positions_m'];verror=v-finite['velocities_point_chord_m_s']
        finite_motion=rmse(finite['positions_m']-hardware.points);linear_motion=rmse(q-hardware.points)
        arrays={k:value.numpy() for k,value in prediction.items() if isinstance(value,torch.Tensor)}
        arrays['requested_output_times_s']=prediction['requested_output_times_s']
        np.savez_compressed(output/'predictions.npz',**arrays)
        result=dict(cell_id=protocol['cell_id'],frames_compared=61,integration_hz=120,
            position_coordinate_RMSE_m=rmse(error),position_maximum_absolute_coordinate_error_m=float(np.max(np.abs(error))),
            position_maximum_node_vector_error_m=float(np.max(np.linalg.norm(error,axis=-1))),
            velocity_coordinate_RMSE_m_s=rmse(verror),velocity_maximum_absolute_coordinate_error_m_s=float(np.max(np.abs(verror))),
            velocity_maximum_node_vector_error_m_s=float(np.max(np.linalg.norm(verror,axis=-1))),
            finite_motion_coordinate_RMS_m=finite_motion,linear_motion_coordinate_RMS_m=linear_motion,
            linear_to_finite_motion_RMS_ratio=linear_motion/finite_motion,
            error_to_finite_motion_RMS_ratio=rmse(error)/finite_motion,
            finite_J=jacobians(finite['positions_m'],hardware.points,hardware.tets),
            linear_J=jacobians(q,hardware.points,hardware.tets),
            finite_patch_distance=patch_defect(finite['positions_m'],hardware.points,[h.ordered_patch_nodes for h in hardware.handles]),
            linear_patch_distance=patch_defect(q,hardware.points,[h.ordered_patch_nodes for h in hardware.handles]),
            runtime=dict(baseline_construction_s=assembled-wall,baseline_predict_s=predicted-assembled,
                baseline_construction_plus_prediction_s=predicted-wall,finite_logged_elapsed_s=summary['elapsed_s'],
                comparison_is_controlled_benchmark=False),
            cached_effective_matrix_factorizations=model.factorizations,
            exact_force_bytes_and_public_clock_match=True,same_material_full_hardware_and_geometry=True,
            information_role='supplied physical hypothesis forward baseline, zero observed response frames',
            fitting_or_hyperparameter_selection_performed=False,raw_prediction_storage='LOCAL_ONLY',
            declaration_sha256=sha(DECLARATION),source_sha256=d['source_sha256'],
            prediction_sha256=sha(output/'predictions.npz'),finite_trajectory_sha256=sha(finite_root/'trajectory.npz'),
            limitation='One120Hz engineering comparison only; finite response is not temporal/continuum-qualified. Linear strain and infinitesimal rigid rotations are approximate; no nonlinear-exact, finiteSO3, prediction-generalization or uncertainty claim.')
        for rel,digest in d['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
        write(output/'comparison.json',result)
        write(output/'execution.json',dict(pid=os.getpid(),terminal=True,completed=True,CPU_threads=2,GPU_used=False,comparison_sha256=sha(output/'comparison.json')))
        print(json.dumps({k:result[k] for k in ['cell_id','position_coordinate_RMSE_m','velocity_coordinate_RMSE_m_s','linear_to_finite_motion_RMS_ratio','runtime']}),flush=True)
    except BaseException as error:
        write(output/'failure.json',dict(type=type(error).__name__,message=str(error)))
        write(output/'execution.json',dict(pid=os.getpid(),terminal=True,completed=False,CPU_threads=2,GPU_used=False))
        raise

if __name__=='__main__':run()
