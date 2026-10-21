"""Isolated documented TGS force-integration option, never a silent dataset fix."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import numpy as np
try:
    from campaign_numerical_convergence import atomic,checked_write,sha,command_clock_audit
    from campaign_numerical_force_polynomial import finite_strain
except ImportError:
    from experiments.campaign_numerical_convergence import atomic,checked_write,sha,command_clock_audit
    from experiments.campaign_numerical_force_polynomial import finite_strain


def set_forcing_flag(stage,enabled):
    if type(enabled) is not bool:raise ValueError('Explicit Boolean required')
    scene=stage.GetPrimAtPath('/physicsScene')
    attr=scene.GetAttribute('physxScene:enableExternalForcesEveryIteration')
    if not attr or not attr.Set(enabled) or attr.Get()!=enabled:
        raise ValueError('Requested documented scene flag unavailable or unresolved')
    return {'property':'physxScene:enableExternalForcesEveryIteration',
        'value':bool(attr.Get()),'authored':attr.HasAuthoredValueOpinion(),
        'provider':'Composed USD after explicit authoring; not an independent native getter'}


def factorial_config(protocol,cell):
    config=copy.deepcopy(protocol['factorial_config']);config['iterations']=[cell['iterations']]
    config['coordinate_conditions']=[{'name':'original','translation_m':[0.,0.,0.],'gpu':cell['gpu']}]
    config.update(protocol=protocol['config']['protocol'],qualification=protocol['config']['qualification'])
    return config


def worker(args,protocol):
    import isaac_calibrated_probes_v03 as base
    import campaign_numerical_solver_coordinates as factorial
    if sha(base.__file__)!=protocol['config']['base_worker_sha256'] or sha(factorial.__file__)!=protocol['factorial_wrapper_sha256']:
        raise ValueError('Frozen base/factorial source changed')
    cell=next(c for c in protocol['config']['cells'] if c['id']==args.cell)
    root=Path(args.output)/cell['id'];config=factorial_config(protocol,cell)
    original_environment=base.create_environment;original_capture=base.capture_current_state
    state_root=None
    def environment(stage,case,path):
        nonlocal state_root
        state_root=Path(path);record=set_forcing_flag(stage,cell['enabled'])
        env=original_environment(stage,case,path)
        atomic(state_root/'additional-scene-flag.json',record)
        return env
    def capture(world,*a,**kw):
        attr=world.stage.GetPrimAtPath('/physicsScene').GetAttribute('physxScene:enableExternalForcesEveryIteration')
        before=bool(attr.Get())
        if before!=cell['enabled']:raise ValueError('Scene flag changed after reset')
        # The immediate caller is the hash-pinned factorial capture wrapper;
        # locate the pinned base caller to access its existing rigid tensor view.
        frame=inspect.currentframe().f_back
        try:
            for _ in range(4):
                if frame is None:raise RuntimeError('Base frame unavailable for native mass snapshot')
                if frame.f_code is base.run_worker.__code__:break
                frame=frame.f_back
            else:raise RuntimeError('Unexpected capture stack')
            handles=frame.f_locals['handles'];action=frame.f_locals['action_id']
        finally:del frame
        target=state_root/(action+'-native-rigid-properties.json')
        if not target.exists():
            view=handles._physics_rigid_body_view
            getters=['get_masses','get_inv_masses','get_inertias','get_coms']
            values={name:getattr(view,name)().numpy().copy().tolist() for name in getters}
            atomic(target,{'provider':'Native RigidBodyView tensor API backend getters, copied after reset',
                'prim_paths':list(view.prim_paths),'buffers':values,
                'deformable_nodal_mass_getter_available':False,
                'qualification':'Native rigid properties only; do not infer actual deformable mass from authored density. COM quaternion layout retained as native raw buffer.'})
        sensor,audit=original_capture(world,*a,**kw)
        after=bool(attr.Get())
        if after!=before:raise ValueError('Scene force flag changed during capture')
        audit['composed_external_force_flag_before']=before;audit['composed_external_force_flag_after']=after
        return sensor,audit
    base.create_environment=environment;base.capture_current_state=capture
    return factorial.worker(argparse.Namespace(condition='original',iterations=cell['iterations'],
        dataset=str(root/'original/staged-dataset'),captures=str(root/'original'/f'i{cell["iterations"]}'/'captures')),config)


def analyze(output,protocol,ledger):
    root=Path(output);rows=[];loaded={};hashes={}
    for cell in protocol['config']['cells']:
        status=next(r for r in ledger['cells'] if r['id']==cell['id'])
        private=root/cell['id']/'original'/f'i{cell["iterations"]}/captures/private/t0001'
        row={**cell,'state':status['state'],'exit_code':status['exit_code']}
        if status['state']!='completed_verified':
            path=private/'failure.json';row['failure']=json.loads(path.read_text()) if path.exists() else status.get('error')
            rows.append(row);continue
        metadata=json.loads((private/'actual-solver-state.json').read_text())
        observed=metadata['prims']['scene']['attributes']['physxScene:enableExternalForcesEveryIteration']
        if observed['value']!=cell['enabled'] or not observed['authored']:raise ValueError('Scene flag metadata mismatch')
        loaded[cell['id']]={}
        for action in ['zero_force','evaluation','reset_replay']:
            path=private/action/'trajectory.npz'
            with np.load(path) as raw:data={k:raw[k].copy() for k in raw.files}
            loaded[cell['id']][action]=data;hashes[str(path)]=sha(path)
            audits=json.loads((private/action/'capture-audit.json').read_text())
            if not all(a['paused_physics_state_bitwise_unchanged'] and a['composed_external_force_flag_before']==cell['enabled'] and a['composed_external_force_flag_after']==cell['enabled'] for a in audits):raise ValueError('Incomplete state/flag guards')
            if action=='evaluation':
                with np.load(private/'reference.npz') as raw:rest,tets=raw['rest_points_m'],raw['tetrahedra']
                row['diagonal_coordinate_rms_from_rest_m']=float(np.sqrt(np.mean((data['positions_m'].astype(float)-rest)**2)))
                row['diagonal_strain']=finite_strain(rest,tets,data['positions_m'])
                row['command_clock']=command_clock_audit(data,960,metadata,[.01,.01,.005])
        rows.append(row)
    comparisons=[]
    for a,b in [('false32','true32'),('true32','true64'),('true64','true128')]:
        if a not in loaded or b not in loaded:
            comparisons.append({'cells':[a,b],'available':False,'reason':'Failed predeclared cell retained'});continue
        x,y=loaded[a]['evaluation'],loaded[b]['evaluation']
        np.testing.assert_array_equal(x['force_step_commands_N'],y['force_step_commands_N'])
        dq=x['positions_m'].astype(float)-y['positions_m'].astype(float)
        comparisons.append({'cells':[a,b],'available':True,'coordinate_rmse_m':float(np.sqrt(np.mean(dq**2)))})
    replay={};baseline=Path('runs/gaussian-campaign-v01/numerical-solver-coordinates-v02/original/i32/captures/private/t0001')
    if 'false32' in loaded:
        for action in ['zero_force','evaluation','reset_replay']:
            with np.load(baseline/action/'trajectory.npz') as raw:
                replay[action]={k:bool(np.array_equal(raw[k],loaded['false32'][action][k])) for k in ['positions_m','velocities_m_s','force_step_commands_N']}
        if not all(all(v.values()) for v in replay.values()):raise ValueError('FALSE32 fails required bitwise baseline replay; do not interpret flag intervention')
    result={'cells':rows,'comparisons':comparisons,'false32_baseline_bitwise':replay,
        'input_sha256':hashes,'source_sha256':sha(__file__),'qualification':protocol['config']['qualification']}
    atomic(root/'analysis.json',result);return result


def driver(args,config):
    import campaign_numerical_solver_coordinates as factorial
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    fp=Path(factorial.__file__);fc=fp.with_name('campaign_numerical_solver_coordinates_config.json')
    protocol={'config':config,'config_sha256':sha(args.config),'wrapper_sha256':sha(__file__),
        'factorial_wrapper_sha256':sha(fp),'factorial_config':json.loads(fc.read_text())}
    checked_write(root/'protocol.json',(json.dumps(protocol,indent=2)+'\n').encode())
    for p in [Path(__file__),Path(args.config),fp,fc]:checked_write(root/'provenance'/p.name,p.read_bytes())
    for cell in config['cells']:factorial.prepare(root/cell['id'],factorial_config(protocol,cell))
    if not args.launch:return {'prepared':True,'simulation_launched':False}
    if (root/'execution.json').exists():raise FileExistsError('Preserve all existing flag trials')
    ledger={'driver_pid':os.getpid(),'cells':[],'completed':False};lock=threading.Lock()
    def save():atomic(root/'execution.json',ledger)
    def group(gpu):
        for cell in [c for c in config['cells'] if c['gpu']==gpu]:
            if sha(__file__)!=protocol['wrapper_sha256'] or sha(fp)!=protocol['factorial_wrapper_sha256']:raise ValueError('Source changed')
            command=[protocol['factorial_config']['isaac_python'],'-u',str(Path(__file__)),
                '--worker','--output',str(root),'--cell',cell['id']]
            row={**cell,'state':'running','command':command,'started_unix_s':time.time()}
            with (root/cell['id']/'worker.log').open('x') as log:
                p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
                with lock:row['pid']=p.pid;ledger['cells'].append(row);save()
                code=p.wait()
            with lock:row.update(exit_code=code,state='captured' if code==0 else 'failed',finished_unix_s=time.time());save()
    with ThreadPoolExecutor(max_workers=2) as executor:list(executor.map(group,[1,2]))
    for cell in config['cells']:
        row=next(r for r in ledger['cells'] if r['id']==cell['id'])
        if row['exit_code']==0:
            try:
                sub=factorial_config(protocol,cell);factorial.validate_cell(root/cell['id'],sub,sub['coordinate_conditions'][0],cell['iterations']);row['state']='completed_verified'
            except Exception as e:row.update(state='failed_validation',error=repr(e))
        save()
    ledger['completed']=all(r['state']=='completed_verified' for r in ledger['cells']);save()
    return analyze(root,protocol,ledger)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('campaign_numerical_external_forces_config.json'))
    p.add_argument('--output',default='runs/gaussian-campaign-v01/numerical-external-forces-v01')
    p.add_argument('--worker',action='store_true');p.add_argument('--launch',action='store_true');p.add_argument('--cell')
    args=p.parse_args()
    if args.worker:raise SystemExit(worker(args,json.loads((Path(args.output)/'protocol.json').read_text())))
    print(json.dumps(driver(args,json.loads(args.config.read_text())),indent=2))
