"""Explicit paired attachment experiment; original simulator files stay intact."""
import argparse
import copy
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

try:
    from campaign_numerical_convergence import sha,atomic,checked_write
    from campaign_numerical_validate_rates import adapt_reset_reference
except ImportError:
    from experiments.campaign_numerical_convergence import sha,atomic,checked_write
    from experiments.campaign_numerical_validate_rates import adapt_reset_reference


def replace_patch(spec,config):
    if spec['load_nodes']!=config['original_anchor_nodes']:raise ValueError('Anchor changed')
    h=config['handle_index'];patch=list(config['replacement_patch_sorted'])
    if sorted(spec['load_patches'][h])!=config['original_patch_sorted']:raise ValueError('Original hardware changed')
    if len(patch)!=9 or len(set(patch))!=9 or spec['load_nodes'][h] not in patch:raise ValueError('Invalid replacement attachment')
    if set(patch).intersection(spec['fixture_nodes']) or set(patch).intersection(spec['load_patches'][1-h]):
        raise ValueError('Replacement overlaps a fixture or other handle')
    modified=copy.deepcopy(spec);modified['load_patches'][h]=patch
    return modified


def install_geometry(base,config,variant):
    import isaac_dataset
    original=isaac_dataset.geometry
    def geometry(case):
        if case['episode_id']!='t0000' or case['split']!='train' or case['physics_hz']!=960:
            raise ValueError('Only declared fixed TRAIN hardware audit allowed')
        points,tets,faces,spec=original(case)
        return points,tets,faces,replace_patch(spec,config) if variant=='counterfactual' else spec
    isaac_dataset.geometry=geometry
    return isaac_dataset,original


def worker(args,config):
    import isaac_calibrated_probes_v03 as base
    if sha(base.__file__)!=config['base_worker_sha256']:raise ValueError('Base worker changed')
    original_menu=base.calibrated_menu
    def menu(*a,**kw):
        m=original_menu(*a,**kw);m['candidates']=[{**m['candidates'][0],
            'action_id':'zero_force','kind':'hardware_audit_zero','peak_force_N':[0.,0.,0.]}]
        m['protocol']=config['protocol'];m['hardware_variant']=args.variant
        m['qualification']=config['qualification'];return m
    base.calibrated_menu=menu
    module,original=install_geometry(base,config,args.variant)
    # The immutable original recipe remains intact; this additional wrapper
    # record, staged spec and actual generated spec identify the counterfactual.
    return base.run_worker(argparse.Namespace(dataset=Path(args.dataset),output=Path(args.captures),episode='t0000',gpu=config['gpu']))


def validate_pair_capture(root,config,variant,provenance):
    import isaac_calibrated_probes_v03 as base
    if sha(base.__file__)!=config['base_worker_sha256']:raise ValueError('Base changed')
    module,original=install_geometry(base,config,variant)
    try:
        source=adapt_reset_reference(inspect.getsource(base.validate_case),'zero_force')
        path=Path(provenance)/('adapted_validate_'+variant+'.py');checked_write(path,source.encode())
        ns=dict(vars(base));exec(compile(source,str(path),'exec'),ns)
        result=ns['validate_case'](root,'t0000')
        result['additional_hardware_wrapper_sha256']=sha(__file__)
        result['menu_reset_adapter_sha256']=sha(path)
        return result
    finally:module.geometry=original


def prepare(output,config):
    from gaussian_operator import physical_vector
    dataset=Path(config['dataset']);original=dataset/'episodes/t0000'
    case=json.loads((original/'truth/recipe.json').read_text())
    if case['split']!='train' or case['physics_hz']!=960:raise ValueError('Case role/rate changed')
    spec=json.loads((original/'truth/boundaries.json').read_text());alternative=replace_patch(spec,config)
    with np.load(original/'reference.npz') as raw:points=raw['rest_points_m']
    before,after=physical_vector(case,spec,points),physical_vector(case,alternative,points)
    np.testing.assert_array_equal(before,after)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    lineages={}
    for variant,actual in [('original',spec),('counterfactual',alternative)]:
        staged=output/variant/'staged-dataset';episode=staged/'episodes/t0000'
        for name in ['truth/recipe.json','observations/cameras.npz']:
            checked_write(episode/name,(original/name).read_bytes())
        checked_write(episode/'truth/boundaries.json',(json.dumps(actual,indent=2)+'\n').encode())
        lineage={'variant':variant,'original_recipe_sha256':sha(original/'truth/recipe.json'),
            'original_boundaries_sha256':sha(original/'truth/boundaries.json'),
            'staged_boundaries_sha256':sha(episode/'truth/boundaries.json'),
            'patch_nodes':actual['load_patches'][1],
            'patch_sha256':hashlib.sha256(np.array(actual['load_patches'][1],dtype=np.int64).tobytes()).hexdigest(),
            'full25D_vector':before.tolist(),'full25D_identical_between_variants':True,
            'qualification':config['qualification'],'runtime_geometry_override':variant=='counterfactual'}
        checked_write(staged/'lineage.json',(json.dumps(lineage,indent=2)+'\n').encode());lineages[variant]=lineage
    checked_write(output/'pair-lineage.json',(json.dumps(lineages,indent=2)+'\n').encode())


def analysis(output):
    output=Path(output);data={};hashes={}
    for variant in ['original','counterfactual']:
        private=output/variant/'captures/private/t0000'
        if not json.loads((private/'validation.json').read_text()).get('passed'):raise ValueError('Unverified variant')
        data[variant]={}
        for action in ['zero_force','evaluation']:
            path=private/action/'trajectory.npz'
            with np.load(path) as raw:data[variant][action]={k:raw[k].copy() for k in raw.files}
            hashes[str(path)]=sha(path)
    rows=[]
    for action in ['zero_force','evaluation']:
        a,b=[data[v][action] for v in ['original','counterfactual']]
        for key in ['times_s','force_step_commands_N','force_step_midpoints_s','boundary_target_positions_m']:
            np.testing.assert_array_equal(a[key],b[key])
        delta=b['positions_m'].astype(float)-a['positions_m'].astype(float)
        rows.append({'action_id':action,'coordinate_difference_rms_m':float(np.sqrt(np.mean(delta**2))),
            'maximum_nodal_distance_m':float(np.linalg.norm(delta,axis=-1).max()),
            'velocity_difference_rms_m_s':float(np.sqrt(np.mean((a['velocities_m_s'].astype(float)-b['velocities_m_s'].astype(float))**2)))})
    ua=data['original']['evaluation']['positions_m'].astype(float)-data['original']['zero_force']['positions_m'].astype(float)
    ub=data['counterfactual']['evaluation']['positions_m'].astype(float)-data['counterfactual']['zero_force']['positions_m'].astype(float)
    result={'rows':rows,'zero_referenced_response_difference_rms_m':float(np.sqrt(np.mean((ub-ua)**2))),
        'original_zero_referenced_response_rms_m':float(np.sqrt(np.mean(ua**2))),
        'counterfactual_zero_referenced_response_rms_m':float(np.sqrt(np.mean(ub**2))),
        'input_sha256':hashes,'source_sha256':sha(__file__),
        'qualification':'One controlled TRAIN pair at fixed numerical discretization. Measures this omitted hardware effect; not a global error decomposition, not continuum-accuracy evidence, and never fitting data.'}
    atomic(output/'analysis.json',result);return result


def driver(args,config):
    output=Path(args.output);prepare(output,config)
    base=Path(__file__).with_name('isaac_calibrated_probes_v03.py')
    if sha(base)!=config['base_worker_sha256']:raise ValueError('Base changed')
    protocol={'config':config,'config_sha256':sha(args.config),'wrapper_sha256':sha(__file__),'base_worker_sha256':sha(base)}
    checked_write(output/'protocol.json',(json.dumps(protocol,indent=2)+'\n').encode())
    for p in [Path(__file__),Path(args.config),Path(__file__).with_name('campaign_numerical_validate_rates.py')]:
        checked_write(output/'provenance'/p.name,p.read_bytes())
    if not args.launch:return {'prepared':True,'simulation_launched':False}
    lock=(output/'.driver.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'execution.json').exists():raise FileExistsError('Inspect existing pair; no overwrite')
    ledger={'driver_pid':os.getpid(),'gpu':config['gpu'],'variants':[],'completed':False};atomic(output/'execution.json',ledger)
    for variant in ['original','counterfactual']:
        if sha(__file__)!=protocol['wrapper_sha256']:raise ValueError('Wrapper source changed')
        captures=output/variant/'captures';command=[config['isaac_python'],'-u',str(Path(__file__)),
            '--config',str(args.config),'--worker','--variant',variant,
            '--dataset',str(output/variant/'staged-dataset'),'--captures',str(captures)]
        row={'variant':variant,'state':'running','command':command,'started_unix_s':time.time()};ledger['variants'].append(row)
        with (output/variant/'worker.log').open('x') as log:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
            row['pid']=proc.pid;atomic(output/'execution.json',ledger);code=proc.wait()
        row['exit_code']=code
        if code==0:
            try:
                result=validate_pair_capture(captures,config,variant,output/'provenance')
                atomic(captures/'private/t0000/validation.json',result);row['state']='completed_verified'
            except Exception as e:row.update(state='failed_validation',error=repr(e))
        else:row['state']='failed'
        row['finished_unix_s']=time.time();atomic(output/'execution.json',ledger)
    ledger['completed']=all(r['state']=='completed_verified' for r in ledger['variants']);atomic(output/'execution.json',ledger)
    if ledger['completed']:analysis(output)
    return ledger


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('campaign_numerical_hardware_pair_config.json'))
    p.add_argument('--output',default='runs/gaussian-campaign-v01/numerical-hardware-pair-v01');p.add_argument('--launch',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--variant',choices=['original','counterfactual'])
    p.add_argument('--dataset');p.add_argument('--captures')
    args=p.parse_args();config=json.loads(args.config.read_text())
    if args.worker:raise SystemExit(worker(args,config))
    print(json.dumps(driver(args,config),indent=2))
