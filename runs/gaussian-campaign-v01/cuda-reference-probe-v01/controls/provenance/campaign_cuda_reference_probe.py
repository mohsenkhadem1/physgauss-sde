"""Separate float64 CUDA device-binding probe; immutable CPU physics is unchanged.

CPU assembly is retained. Private copies of original function objects keep their
code objects, but bind the CPU tensor helper/factories to the selected device.
No CPU module globals are patched. This is a parity/timing experiment, not a
qualified GPU reference, new force envelope or production data generator.
"""
from __future__ import annotations

import argparse
import copy
import cProfile
import hashlib
import io
import json
import marshal
import os
from pathlib import Path
import pstats
import subprocess
import time
import traceback
import types
import xml.etree.ElementTree as ET

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import numpy as np
import torch

try:
    import campaign_nonlinear_dynamics as core
    import campaign_nonlinear_fem as fem
    import campaign_linear_fem as linear
except ImportError:
    from experiments import campaign_nonlinear_dynamics as core
    from experiments import campaign_nonlinear_fem as fem
    from experiments import campaign_linear_fem as linear


PINNED = {'campaign_nonlinear_dynamics.py': 'da0db7018b4913020bb228c2fc4df3c240e7dd872f17db4868de147eaaf78fa2',
          'campaign_nonlinear_fem.py': '468daaafa9d02c9db27553e2cfb154190248500677c212d9dde25bffdf14e1ff',
          'campaign_linear_fem.py': 'c4a94946b8d0edd2246a889dddcf89b58e0a8c660d75250a89f95de85b31ca55',
          'campaign_rigid_inertia.py': 'e420d9cadaa97d1b1648227b7b28af1803952caa87432f819baf0827deecabf7'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sources():
    found = {name: sha(Path(__file__).with_name(name)) for name in PINNED}
    if found != PINNED:
        raise ValueError('Pinned CPU physical source changed')
    return found


class DeviceFactories:
    """Only allocation placement changes; arithmetic still uses original torch."""
    FACTORIES = {'as_tensor', 'tensor', 'zeros', 'ones', 'empty', 'eye', 'arange', 'full'}

    def __init__(self, device):
        self.device = torch.device(device)

    def __getattr__(self, name):
        original = getattr(torch, name)
        if name not in self.FACTORIES:
            return original
        def allocate(*args, **kwargs):
            if 'device' in kwargs and torch.device(kwargs['device']) != self.device:
                raise ValueError('Unexpected explicit foreign-device allocation')
            kwargs['device'] = self.device
            return original(*args, **kwargs)
        return allocate


def make_device_model(cpu_model, device):
    """Independent private method globals; never monkey-patch shared modules."""
    original_tensor = core.tensor
    namespace = dict(vars(core))
    namespace['torch'] = DeviceFactories(device)
    namespace['tensor'] = lambda value: torch.as_tensor(value, dtype=torch.float64, device=device)
    helpers = {'skew': linear.skew, 'determinant3': fem.determinant3,
               'rotation_matrix': fem.rotation_matrix, 'rotation_left_jacobian': fem.rotation_left_jacobian,
               'x_minus_log1p': fem.x_minus_log1p, 'neo_hookean_energy': fem.neo_hookean_energy,
               'equilibrium_metrics': fem.equilibrium_metrics, 'finite_array': core.finite_array}
    def clone(function):
        result = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
        result.__kwdefaults__ = copy.copy(function.__kwdefaults__)
        return result
    for name, function in helpers.items():
        namespace[name] = clone(function)
    methods = {name: clone(value) for name, value in vars(core.NonlinearDynamics).items()
               if isinstance(value, types.FunctionType) and name != '__init__'}
    cls = type('DeviceBoundNonlinearDynamics', (core.NonlinearDynamics,), methods)
    model = copy.deepcopy(cpu_model)
    model.__class__ = cls
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, fem.FiniteRigidPatchMap):
            value.__class__ = type('DeviceBoundFiniteRigidPatchMap', (fem.FiniteRigidPatchMap,),
                                   {'positions': clone(fem.FiniteRigidPatchMap.positions)})
            for key, item in vars(value).copy().items():
                setattr(value, key, move(item))
            return value
        return value
    for name, value in vars(model).copy().items():
        setattr(model, name, move(value))
    model.tets = torch.as_tensor(cpu_model.tets, dtype=torch.long, device=device)
    model._factor_cache = {}  # Rebuild the original physical metric on this device.
    buffers={name:getattr(model,name) for name in ['points','K','body_mass','gradients','volumes','auxiliary_offsets','auxiliary_mass','com_offsets']}
    buffers.update(mapping_A=model.mapping.A,mapping_points=model.mapping.points)
    if any(value.dtype!=torch.float64 or value.device!=torch.device(device) for value in buffers.values()):
        raise ValueError('Physical floating buffers must all be float64 on requested device')
    if core.tensor is not original_tensor:
        raise AssertionError('Shared CPU module binding changed')
    proof = {'method_code_identity': {name: getattr(cls, name).__code__ is getattr(core.NonlinearDynamics, name).__code__ for name in methods},
             'helper_code_identity': {name: namespace[name].__code__ is function.__code__ for name, function in helpers.items()},
             'cpu_tensor_binding_unchanged': True, 'device': str(device),
             'physical_buffer_dtype_devices': {name:{'dtype':str(value.dtype),'device':str(value.device)} for name,value in buffers.items()},
             'cpu_assembly_then_transfer': True, 'preconditioner_rebuilt_on_device': True}
    return model, proof


def numpy(value):
    return value.detach().cpu().numpy().copy()


def state_arrays(state):
    return {name: numpy(getattr(state, name)) for name in ['positions', 'velocities', 'auxiliary_positions',
        'auxiliary_velocities', 'handle_centers', 'handle_rotations']}


def compare_states(a, b, position_tolerance=1e-10, velocity_tolerance=1e-8):
    aa, bb = state_arrays(a), state_arrays(b)
    errors = {name: float(abs(aa[name]-bb[name]).max()) if aa[name].size else 0. for name in aa}
    for name, error in errors.items():
        limit = velocity_tolerance if 'velocities' in name else position_tolerance
        if error > limit:
            raise ValueError('Device mechanics parity failed: '+str((name, error, limit)))
    return errors


def unit_tet():
    return np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]), np.array([[0,1,2,3]])


def rigid_cube():
    points = np.array([[0.,0.,0.],[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                       [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=float)*.1
    tets = []
    for a,b,c,d in [[1,2,3,4],[5,6,7,8],[1,2,6,5],[2,3,7,6],[3,4,8,7],[4,1,5,8]]:
        for tri in [[a,b,c],[a,c,d]]:
            tet = [0,*tri]
            if np.linalg.det((points[tet[1:]]-points[tet[0]]).T)<0:
                tet[2],tet[3] = tet[3],tet[2]
            tets.append(tet)
    return points, np.array(tets)


def control_models():
    points,tets = unit_tet()
    rows = []
    for kind, fixed in [('free',[]),('point_pin',[0]),('fixed_patch',[0,1,2]),('two_patches',[0,1])]:
        rows.append((kind+'-rest', core.NonlinearDynamics(points,tets,1200.,.3,600.,fixture_nodes=fixed,boundary_kind=kind), {}, {}, 1))
    body = core.NonlinearDynamics(points,tets,1200.,.32,600.)
    acceleration = np.array([.03,-.02,.01])
    rows.append(('uniform-acceleration',body,{}, {'nodal_forces':numpy(body.body_mass)[:,None]*acceleration},4))
    pts,tet = rigid_cube()
    rigid = core.NonlinearDynamics(pts,tet,1500.,.3,20.,load_patches=[list(range(9))],load_nodes=[0],
        handle_masses=[.7],handle_inertias=[np.diag([.002,.003,.004])])
    velocity = np.array([.03,.02,-.01]);omega = np.array([.4,.7,-.9])
    initial = dict(velocities=velocity+np.cross(omega,pts),handle_linear_velocities=[velocity],handle_angular_velocities=[omega])
    rows.append(('finite-rigid-rotation',rigid,initial,{'handle_forces':np.array([[.003,-.002,.001]])},4))
    acceleration = np.array([.2,-.1,.15])
    moving = core.NonlinearDynamics(points,tets,1000.,.3,600.,fixture_nodes=list(range(4)),boundary_kind='moving_patch',
        fixture_target=lambda t: points+.5*acceleration*t*t,drag=.7)
    rows.append(('accelerating-prescribed-boundary',moving,{}, {},4))
    return rows


def run_control_set(device=None, compare=None):
    output = []
    for name, original, initial, forces, count in control_models():
        model, proof = make_device_model(original, device) if device is not None else (original, None)
        state = model.initial_state(**initial); states = []; ledgers = []
        for _ in range(count):
            state, ledger = model.step(state,.01,**forces,relative_tolerance=1e-10,absolute_tolerance_N=1e-10)
            states.append(state);ledgers.append(ledger)
            for value in state_arrays(state).values():
                if not np.isfinite(value).all() or value.dtype != np.float64:
                    raise ValueError('Nonfinite or non-float64 mechanics state')
        errors = []
        if compare is not None:
            reference = next(row for row in compare if row['name']==name)
            errors = [compare_states(a,b) for a,b in zip(reference['states'],states)]
            for cpu,gpu in zip(reference['ledgers'],ledgers):
                if gpu['nonlinear_history'][-1]['scaled_generalized_residual_norm_N'] > gpu['residual_tolerance_N']:
                    raise ValueError('Original nonlinear residual tolerance not met')
                if abs(cpu['minimum_J']-gpu['minimum_J'])>1e-9:
                    raise ValueError('MinimumJ parity failed')
                for field in ['support_work_J','body_viscous_dissipation_J','energy_algebra_identity_residual_J']:
                    if abs(cpu[field]-gpu[field])>1e-10:
                        raise ValueError('Work/energy parity failed: '+field)
        if name=='uniform-acceleration':
            x,_=unit_tet(); n=count;h=.01;g=np.array([.03,-.02,.01])
            np.testing.assert_allclose(numpy(state.positions),x+h*h*n*(n+1)/2*g,atol=1e-11,rtol=0)
            np.testing.assert_allclose(numpy(state.velocities),np.tile(n*h*g,(4,1)),atol=1e-10,rtol=0)
        if name=='accelerating-prescribed-boundary':
            x,_=unit_tet();np.testing.assert_allclose(numpy(state.positions),x+.5*np.array([.2,-.1,.15])*.04**2,atol=1e-14,rtol=0)
        output.append({'name':name,'states':states,'ledgers':ledgers,'errors':errors,'binding_proof':proof})
    return output


def gpu_assignment():
    tree=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True))
    rows=[{'gpu':int(g.findtext('minor_number')),'pid':os.getpid(),'type':p.findtext('type'),'memory':p.findtext('used_memory')}
          for g in tree.findall('gpu') for p in g.findall('./processes/process_info') if p.findtext('pid')==str(os.getpid())]
    if {row['gpu'] for row in rows}!={2}:
        raise ValueError('Torch probe must use physicalGPU2 only: '+str(rows))
    return rows


def controls(root):
    profiler=cProfile.Profile();start=time.perf_counter();profiler.enable();cpu=run_control_set();profiler.disable()
    cpu_wall=time.perf_counter()-start
    text=io.StringIO();pstats.Stats(profiler,stream=text).strip_dirs().sort_stats('cumulative').print_stats(35)
    (root/'cpu-profile.txt').write_text(text.getvalue())
    # CUDA initializes only after the CPU profile has completed.
    start=time.perf_counter();torch.cuda.set_device(0);torch.empty(1,device='cuda:0');torch.cuda.synchronize();startup=time.perf_counter()-start
    start=time.perf_counter();gpu=run_control_set('cuda:0',cpu);torch.cuda.synchronize();gpu_wall=time.perf_counter()-start
    serial=[]
    for row in gpu:
        serial.append({key:value for key,value in row.items() if key not in ['states']})
    result={'completed':True,'all_tiny_controls_passed':True,'physics_source_sha256':sources(),
            'probe_source_sha256':sha(__file__),'CPU_profile_wall_s':cpu_wall,'CUDA_startup_s':startup,
            'GPU_controls_including_assembly_transfer_telemetry_wall_s':gpu_wall,'controls':serial,
            'gpu_assignment':gpu_assignment(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,
            'device':torch.cuda.get_device_name(0),'float64':True,'production_equivalence_claim':False}
    result['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    result['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
    write(root/'summary.json',result)
    return result


def load_reference(folder):
    folder=Path(folder);protocol=json.loads((folder/'protocol.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    execution=json.loads((folder/'execution.json').read_text())
    if (protocol['episode_id'],protocol['physics_hz'],protocol['response'])!=('t0001',480,'diagonal_pulse_handle0') or not execution['terminal'] or not execution['completed']:
        raise ValueError('Only completed declared TRAINt0001/480 diagonal reference permitted')
    if sha(folder/'trajectory.npz')!=summary['trajectory_sha256'] or sha(folder/'steps.jsonl')!=summary['steps_sha256']:
        raise ValueError('Captured CPU response hash mismatch')
    if sha(folder/'provenance/static-reference.npz')!=protocol['saved_regenerated_static_reference_sha256']:
        raise ValueError('Captured CPU static-reference hash mismatch')
    if (protocol['relative_nonlinear_residual_tolerance'],protocol['absolute_residual_floor_N'],protocol['nonlinear_iteration_cap'])!=(1e-7,1e-10,250):
        raise ValueError('Require unchanged baseline residual tolerance/floor/cap')
    for old,expected in protocol['source_input_dependency_and_gate_sha256'].items():
        if sha(folder/'provenance'/Path(old).name)!=expected:
            raise ValueError('Frozen CPU reference provenance mismatch')
    with np.load(folder/'trajectory.npz',allow_pickle=False) as raw:
        trajectory={key:raw[key].copy() for key in raw.files}
    with np.load(folder/'provenance/static-reference.npz',allow_pickle=False) as raw:
        reference={key:raw[key].copy() for key in raw.files}
    fixture=json.loads((folder/'provenance/gaussian_independent_diagnostic_inputs_v01.json').read_text())
    case=next(row for row in fixture['cases'] if row['episode_id']=='t0001')
    if case['recipe']['split']!='train' or protocol['translation_m']!=[0.,0.,0.]:
        raise ValueError('Require untranslated TRAIN reference')
    for row in case['reference_static_arrays']:
        value=reference[row['key']]
        if value.dtype!=np.dtype(row['dtype']) or list(value.shape)!=row['shape'] or hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()!=row['sha256']:
            raise ValueError('Canonical reference array mismatch')
    material=protocol['declared_material'];spec=case['boundaries'];hardware=json.loads((folder/'hardware.json').read_text())
    if hardware!=protocol['hardware']:
        raise ValueError('Frozen hardware metadata mismatch')
    model=core.NonlinearDynamics(reference['rest_points_m'],reference['tetrahedra'],material['youngs_modulus_Pa'],material['poissons_ratio'],material['density_kg_m3'],
        fixture_nodes=spec['fixture_nodes'],load_patches=spec['load_patches'],load_nodes=spec['load_nodes'],
        handle_masses=[row['mass_kg'] for row in hardware],handle_inertias=[row['full_COM_inertia_kg_m2'] for row in hardware],
        drag=material['linear_damping_s_inverse'],boundary_kind=case['recipe']['boundary_type'])
    steps=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    return model,trajectory,protocol,summary,steps


def train_probe(root,reference,proof_path):
    proof=json.loads(Path(proof_path).read_text())
    if not proof['all_tiny_controls_passed'] or proof['probe_source_sha256']!=sha(__file__) or proof['physics_source_sha256']!=sources():
        raise ValueError('Matching successful tiny-control proof required')
    setup_start=time.perf_counter();cpu,truth,protocol,cpu_summary,cpu_steps=load_reference(reference)
    model,binding=make_device_model(cpu,'cuda:0');state=model.initial_state();torch.cuda.synchronize()
    setup=time.perf_counter()-setup_start
    times=[];states=[];step_rows=[];transfer_s=0.;io_s=0.;telemetry_s=0.;step_total=0.
    def capture():
        nonlocal transfer_s
        start=time.perf_counter();times.append(state.time_s);states.append(state_arrays(state));torch.cuda.synchronize();transfer_s+=time.perf_counter()-start
    capture();start=time.perf_counter()
    commands=truth['force_step_commands_N'];h=1/480
    if commands.shape!=(1440,2,3):raise ValueError('Unexpected exact captured command sequence')
    with (root/'steps.jsonl').open('x') as stream:
        for index,force in enumerate(commands):
            wall=time.perf_counter();state,row=model.step(state,h,handle_forces=force,relative_tolerance=protocol['relative_nonlinear_residual_tolerance'],absolute_tolerance_N=protocol['absolute_residual_floor_N'],maximum_iterations=protocol['nonlinear_iteration_cap']);torch.cuda.synchronize();seconds=time.perf_counter()-wall;step_total+=seconds
            row['GPU_step_wall_seconds']=seconds
            row.pop('held_nodal_forces_N');step_rows.append(row)
            wall=time.perf_counter();stream.write(json.dumps({'step_index':index,**row},allow_nan=False)+'\n');stream.flush();io_s+=time.perf_counter()-wall
            if (index+1)%48==0:capture()
            if (index+1)%240==0:
                wall=time.perf_counter();write(root/'progress.json',{'completed_steps':index+1,'elapsed_s':time.perf_counter()-start,'last_residual_N':row['nonlinear_history'][-1]['scaled_generalized_residual_norm_N'],'gpu_assignment':gpu_assignment()});telemetry_s+=time.perf_counter()-wall
                sources();print(json.dumps({'steps':index+1,'elapsed_s':time.perf_counter()-start}),flush=True)
    wall=time.perf_counter()-start
    arrays={name:np.array([row[name] for row in states]) for name in states[0]}
    np.savez_compressed(root/'trajectory.npz',times_s=times,**arrays)
    expected={'positions':'positions_m','velocities':'velocities_point_chord_m_s','handle_centers':'handle_centers_m','handle_rotations':'handle_rotations'}
    limits={'positions':1e-9,'velocities':1e-7,'handle_centers':1e-9,'handle_rotations':1e-8}
    errors={name:float(abs(arrays[name]-truth[key]).max()) for name,key in expected.items()}
    if len(times)!=31 or not np.array_equal(np.asarray(times),truth['actual_state_times_s']):raise ValueError('Saved CPU/GPU physical clocks differ')
    parity=all(errors[key]<=limits[key] for key in limits)
    J_error=max(abs(a['minimum_J']-b['minimum_J']) for a,b in zip(step_rows,cpu_steps))
    residuals_ok=all(row['nonlinear_history'][-1]['scaled_generalized_residual_norm_N']<=row['residual_tolerance_N'] for row in step_rows)
    tolerance_error=max(abs(a['residual_tolerance_N']-b['residual_tolerance_N']) for a,b in zip(step_rows,cpu_steps))
    applied=np.array([row['held_handle_forces_N'] for row in step_rows]);command_error=float(abs(applied-commands).max())
    intervals=np.array([[row['start_time_s'],row['end_time_s']] for row in step_rows]);impulse=np.sum(applied*np.diff(intervals,axis=1)[:,:,None],axis=0)
    impulse_error=float(abs(impulse-np.asarray(cpu_summary['command_impulse_Ns'])).max())
    energies={key:max(abs(a['new_invariants'][key]-b['new_invariants'][key]) for a,b in zip(step_rows,cpu_steps)) for key in ['elastic_energy_J','chord_kinetic_energy_J','energy_J']}
    passed=parity and J_error<=1e-8 and residuals_ok and tolerance_error<=1e-20 and command_error==0 and impulse_error<=1e-15
    result={'completed':True,'parity_passed':passed,'nodal_and_pose_maximum_absolute_errors':errors,'predeclared_tolerances':limits,
        'minimum_J_maximum_difference':J_error,'all_original_residual_tolerances_met':residuals_ok,'residual_tolerance_maximum_difference_N':tolerance_error,
        'exact_command_maximum_difference_N':command_error,'impulse_maximum_difference_Ns':impulse_error,'energy_maximum_differences_J':energies,
        'GPU_wall_including_capture_IO_telemetry_s':wall,'CPU_captured_wall_s':cpu_summary['elapsed_s'],'GPU_step_including_original_balance_telemetry_s':step_total,
        'CPU_captured_step_including_original_balance_telemetry_s':sum(row['step_wall_seconds'] for row in cpu_steps),
        'GPU_setup_CPU_assembly_transfer_s':setup,'sample_state_D2H_transfer_s':transfer_s,'JSON_IO_s':io_s,'OS_telemetry_s':telemetry_s,
        'timing_caveat':'Captured CPU run includes per-step geometry eigvalsh telemetry omitted here; execution hardware contention differs. This is initial-path timing, not matched benchmark speedup.',
        'binding_proof':binding,'GPU_assignment':gpu_assignment(),'physics_source_sha256':sources(),'probe_source_sha256':sha(__file__),
        'production_equivalence_claim':False,'no_reserved_data':True,'new_force_envelope':False,'float64':True}
    result['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    result['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
    write(root/'summary.json',result)
    if not passed:raise ValueError('Predeclared GPU/CPU parity gate failed')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['controls','train'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--control-proof',type=Path);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':p.error('Set CUDA_VISIBLE_DEVICES=2; logicalcuda0 is physicalGPU2')
    if args.output.exists():p.error('Use a fresh output; failures are preserved')
    if args.phase=='train' and (args.reference is None or args.control_proof is None):p.error('Train requires exact CPU reference and successful control proof')
    sources();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    root=args.output;root.mkdir(parents=True);(root/'provenance').mkdir()
    for name in [Path(__file__).name,*PINNED]:
        (root/'provenance'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    execution={'pid':os.getpid(),'phase':args.phase,'terminal':False,'completed':False,'source_sha256':sha(__file__),'started_unix_s':time.time()};write(root/'execution.json',execution)
    try:
        result=controls(root) if args.phase=='controls' else train_probe(root,args.reference,args.control_proof)
        execution.update(terminal=True,completed=True,finished_unix_s=time.time());write(root/'execution.json',execution)
        print(json.dumps({'completed':True,'phase':args.phase}),flush=True);return 0
    except BaseException as error:
        traceback.print_exc();write(root/'failure.json',{'type':type(error).__name__,'error':str(error)})
        execution.update(terminal=True,completed=False,finished_unix_s=time.time());write(root/'execution.json',execution);return 1


if __name__=='__main__':
    raise SystemExit(main())
