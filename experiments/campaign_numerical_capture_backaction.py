"""Paired physics-only removal of the display-update/synchronous-capture block.

This is NOT a sensor dataset: no images or ray-QC certificates are generated.
Reuse the frozen factorial's state getters and initialization, changing only the
explicitly persisted derived base-worker blocks below. All original files stay
intact. The intervention measures a combined block, not a renderer alone.
"""
import argparse
import copy
import difflib
import inspect
import json
import os
from pathlib import Path
import subprocess
import numpy as np
try:
    from campaign_numerical_convergence import sha,atomic,checked_write,command_clock_audit
    from campaign_numerical_force_polynomial import finite_strain
except ImportError:
    from experiments.campaign_numerical_convergence import sha,atomic,checked_write,command_clock_audit
    from experiments.campaign_numerical_force_polynomial import finite_strain


def derive_worker(source):
    replacements=[
        ("            action_id = action['action_id']; directory = root/'outcomes'/episode/action_id/'observations'\n            for folder in ['rgb', 'depth', 'masks']:\n                (directory/folder).mkdir(parents=True)\n",
         "            action_id = action['action_id']; directory = root/'physics-queries'/episode/action_id\n            directory.mkdir(parents=True)\n"),
        ("                env['display'].GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(q))\n",''),
        ("                image, depth, mask = sensor; name = f'{frame_index:05d}'\n                Image.fromarray(image).save(directory/'rgb'/f'{name}.png', compress_level=3)\n                Image.fromarray(mask.astype(np.uint8)*255).save(directory/'masks'/f'{name}.png', compress_level=3)\n                np.savez_compressed(directory/'depth'/f'{name}.npz', depth_m=depth)\n",''),
        ("'capture-audit.json'","'physics-observation-audit.json'"),
        ("                       'maximum_sensor_ray_error_m': max(a['maximum_error_m'] for a in capture_audits),\n                       'maximum_sensor_capture_attempts': max(a['attempts'] for a in capture_audits),\n",
         "                       'sensor_qc_performed': False, 'images_generated': False,\n                       'role': 'Physics-only numerical diagnostic; never fitting data',\n")]
    derived=source
    for old,new in replacements:
        if derived.count(old)!=1:raise ValueError('Frozen worker block not uniquely matched')
        derived=derived.replace(old,new)
    return derived


def worker(args,protocol):
    import isaac_calibrated_probes_v03 as base
    import campaign_numerical_solver_coordinates as factorial
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    if sha(base.__file__)!=protocol['base_worker_sha256'] or sha(factorial.__file__)!=protocol['factorial_wrapper_sha256']:
        raise ValueError('Frozen source changed')
    source=inspect.getsource(base.run_worker);derived=derive_worker(source)
    path=root/'provenance/derived_physics_only_worker.py';checked_write(path,derived.encode())
    checked_write(root/'provenance/worker.diff',''.join(difflib.unified_diff(source.splitlines(True),derived.splitlines(True),fromfile='base-v03-run_worker',tofile='physics-only-derived')).encode())
    exec(compile(derived,str(path),'exec'),vars(base))
    def no_sensor_capture(world,rep,rgb,depth_annotator,points,faces,pose,k,attempts=12):
        return None,{'mode':'NO_SENSOR_QC','intermediate_display_update':False,
                     'synchronous_render_or_annotation_readback':False,
                     'qualification':'Only factorial before/after physical-state getters are performed; no sensor/ray acceptance claim'}
    base.capture_current_state=no_sensor_capture
    config=copy.deepcopy(protocol['factorial_config'])
    config.update(protocol=protocol['protocol'],qualification=protocol['qualification'])
    config['coordinate_conditions']=[{'name':'original','translation_m':[0.,0.,0.],'gpu':1}]
    return factorial.worker(argparse.Namespace(condition='original',iterations=32,
        dataset=protocol['dataset'],captures=str(root/'captures')),config)


def analyze(output,protocol):
    root=Path(output);private=root/'captures/private/t0001'
    summary=json.loads((private/'summary.json').read_text())
    if not summary['completed'] or (private/'failure.json').exists():raise ValueError('Failed physics-only run')
    if list(root.rglob('*.png')) or list(root.rglob('capture-audit.json')):
        raise ValueError('Physics-only diagnostic must not contain images or sensor audit claims')
    with np.load(private/'reference.npz') as raw:rest,tets=raw['rest_points_m'],raw['tetrahedra']
    metadata=json.loads((private/'actual-solver-state.json').read_text())
    menu=json.loads((root/'captures/public/t0001/action-menu.json').read_text())
    baseline=Path(protocol['baseline'])/'original/i32/captures/private/t0001'
    rows=[];records={};hashes={}
    for action in ['zero_force','evaluation','reset_replay']:
        path=private/action/'trajectory.npz'
        with np.load(path) as raw:data={k:raw[k].copy() for k in raw.files}
        with np.load(baseline/action/'trajectory.npz') as raw:control={k:raw[k].copy() for k in raw.files}
        records[action]=data;hashes[str(path)]=sha(path);hashes[str(baseline/action/'trajectory.npz')]=sha(baseline/action/'trajectory.npz')
        for k in ['force_step_commands_N','force_step_midpoints_s','times_s','boundary_target_positions_m']:
            np.testing.assert_array_equal(data[k],control[k])
        np.testing.assert_allclose(data['positions_m'][0],rest,atol=1e-6)
        np.testing.assert_allclose(data['velocities_m_s'][0],0,atol=1e-7)
        strain=finite_strain(rest,tets,data['positions_m'])
        if strain['minimum_volume_ratio']<.1:raise ValueError('Inverted/compressed physics-only state')
        peak=menu['evaluation_query']['peak_force_N'] if action=='evaluation' else [0.,0.,0.]
        clock=command_clock_audit(data,960,metadata,peak)
        audit=json.loads((private/action/'physics-observation-audit.json').read_text())
        if len(audit)!=31 or not all(v.get('paused_physics_state_bitwise_unchanged') for v in audit):
            raise ValueError('State-only getter invariance check missing')
        dq=data['positions_m'].astype(float)-control['positions_m'].astype(float)
        dv=data['velocities_m_s'].astype(float)-control['velocities_m_s'].astype(float)
        rows.append({'action_id':action,'positions_bitwise_equal':bool(np.array_equal(data['positions_m'],control['positions_m'])),
            'velocities_bitwise_equal':bool(np.array_equal(data['velocities_m_s'],control['velocities_m_s'])),
            'handle_positions_bitwise_equal':bool(np.array_equal(data['handle_positions_m'],control['handle_positions_m'])),
            'coordinate_rmse_m':float(np.sqrt(np.mean(dq**2))),'velocity_rmse_m_s':float(np.sqrt(np.mean(dv**2))),
            'command_clock':clock,'strain':strain})
    for key in ['positions_m','velocities_m_s']:
        np.testing.assert_array_equal(records['zero_force'][key],records['reset_replay'][key])
    result={'rows':rows,'input_sha256':hashes,'source_sha256':sha(__file__),
        'sensor_qc_performed':False,'images_generated':False,
        'qualification':protocol['qualification']}
    atomic(root/'analysis.json',result);return result


def main(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    factorial=Path(__file__).with_name('campaign_numerical_solver_coordinates.py')
    config=Path(__file__).with_name('campaign_numerical_solver_coordinates_config.json')
    protocol={'protocol':'train-t0001-intermediate-capture-backaction-v0.1',
        'dataset':'runs/gaussian-boundary-neural-v01','gpu':1,
        'baseline':'runs/gaussian-campaign-v01/numerical-solver-coordinates-v02',
        'factorial_config':json.loads(config.read_text()),'factorial_wrapper_sha256':sha(factorial),
        'base_worker_sha256':'116d4717bff0c7780925bd4e6d85f5b2f1615d33f4993e356a8284199829a821',
        'source_sha256':sha(__file__),
        'qualification':'TRAIN-only physics diagnostic, NO_IMAGES and NO_SENSOR_QC. Remove combined per-sample display-update plus synchronous render/readback/image-writing block. Keep initialization/warmup, physical scene, commands, rates, resets and factorial state-getter timing. A difference cannot be attributed to the renderer alone; agreement does not test startup effects or all hidden backend state. Never fitting data.'}
    checked_write(root/'protocol.json',(json.dumps(protocol,indent=2)+'\n').encode())
    for p in [Path(__file__),factorial,config]:checked_write(root/'provenance'/p.name,p.read_bytes())
    if not args.launch:return {'prepared':True,'simulation_launched':False}
    if (root/'execution.json').exists():raise FileExistsError('Preserve existing backaction run')
    command=[protocol['factorial_config']['isaac_python'],'-u',str(Path(__file__)),'--worker','--output',str(root)]
    with (root/'worker.log').open('x') as log:
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
        atomic(root/'execution.json',{'pid':p.pid,'command':command,'state':'running'});code=p.wait()
    atomic(root/'execution.json',{'pid':p.pid,'command':command,'state':'exited','exit_code':code})
    if code:raise RuntimeError('Physics-only worker failed; preserve outputs')
    return analyze(root,protocol)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='runs/gaussian-campaign-v01/numerical-capture-backaction-v01')
    p.add_argument('--launch',action='store_true');p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.worker:raise SystemExit(worker(args,json.loads((Path(args.output)/'protocol.json').read_text())))
    print(json.dumps(main(args),indent=2))
