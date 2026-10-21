"""Failure-aware post-analysis of the frozen six-cell numerical factorial."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from campaign_numerical_force_polynomial import finite_strain
except ImportError:
    from experiments.campaign_numerical_force_polynomial import finite_strain


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root,output):
    root,output=Path(root),Path(output);ledger=json.loads((root/'execution.json').read_text())
    if len(ledger['cells'])!=6 or any(r['state'] in ['running','captured'] for r in ledger['cells']):
        raise ValueError('All six simulations and serialized CPU validation must finish')
    protocol=json.loads((root/'protocol.json').read_text());config=protocol['config']
    if sha(Path(__file__).with_name('campaign_numerical_solver_coordinates.py'))!=protocol['wrapper_sha256']:
        raise ValueError('Frozen factorial source changed')
    hashes={str(root/n):sha(root/n) for n in ['protocol.json','execution.json']}
    data={};rows=[]
    for cell in ledger['cells']:
        name,count=cell['condition'],cell['iterations'];private=root/name/f'i{count}/captures/private/t0001'
        condition=next(c for c in config['coordinate_conditions'] if c['name']==name)
        shift=np.array(condition['translation_m']);row={k:cell[k] for k in ['condition','iterations','state','exit_code']}
        if cell['state']!='completed_verified':
            failure_path=private/'failure.json'
            row['failure']=json.loads(failure_path.read_text()) if failure_path.exists() else cell.get('error')
            if (private/'failure-state.npz').is_file():
                with np.load(private/'failure-state.npz') as raw:q=raw['positions_m'].copy()
                with np.load(private/'reference.npz') as raw:x,tets=raw['rest_points_m'],raw['tetrahedra']
                spec=json.loads((private/'truth/boundaries.json').read_text());fixed=spec['fixture_nodes']
                row['failure_state']={**finite_strain(x,tets,q),
                    'maximum_fixture_error_m':float(np.linalg.norm(q[fixed]-x[fixed],axis=-1).max()),
                    'maximum_nodal_displacement_m':float(np.linalg.norm(q-x,axis=-1).max())}
                hashes[str(private/'failure-state.npz')]=sha(private/'failure-state.npz')
            rows.append(row);continue
        validation=json.loads((private/'validation.json').read_text())
        if not validation.get('passed'):raise ValueError('Missing independent validation')
        metadata=json.loads((private/'actual-solver-state.json').read_text())
        warmup=json.loads((private/'warmup-state-guards.json').read_text())
        if not all(a['bitwise_unchanged'] for a in warmup):raise ValueError('Failed warmup guard')
        row['warmup_guards_passed']=len(warmup);row['synchronous_guards_passed']=0
        row['composed_solver_sleep_attributes']={n:{k:v for k,v in p['attributes'].items()
            if any(w in k.lower() for w in ['sleep','settling','stabil','solverposition','solvertype'])}
            for n,p in metadata['prims'].items()}
        data[(name,count)]={}
        with np.load(private/'reference.npz') as raw:rest,tets=raw['rest_points_m'],raw['tetrahedra']
        for action in ['zero_force','evaluation','reset_replay']:
            path=private/action/'trajectory.npz'
            with np.load(path) as raw:record={k:raw[k].copy() for k in raw.files}
            audits=json.loads((private/action/'capture-audit.json').read_text())
            if len(audits)!=31 or not all(a['paused_physics_state_bitwise_unchanged'] for a in audits):
                raise ValueError('Incomplete synchronous physical-state checks')
            row['synchronous_guards_passed']+=len(audits)
            record['canonical_q']=record['positions_m'].astype(float)-shift
            data[(name,count)][action]=record;hashes[str(path)]=sha(path)
            if action=='evaluation':
                row['diagonal_coordinate_rms_from_rest_m']=float(np.sqrt(np.mean((record['positions_m'].astype(float)-rest)**2)))
                row['diagonal_strain']=finite_strain(rest,tets,record['positions_m'])
        hashes[str(private/'validation.json')]=sha(private/'validation.json')
        hashes[str(private/'actual-solver-state.json')]=sha(private/'actual-solver-state.json')
        rows.append(row)
    def compare(first,second,action):
        if first not in data or second not in data:
            return {'available':False,'reason':'At least one predeclared cell failed; no replacement or imputation'}
        a,b=data[first][action],data[second][action]
        for k in ['times_s','force_step_commands_N','force_step_midpoints_s']:
            np.testing.assert_array_equal(a[k],b[k])
        q=a['canonical_q']-b['canonical_q'];v=a['velocities_m_s'].astype(float)-b['velocities_m_s'].astype(float)
        return {'available':True,'coordinate_rmse_m':float(np.sqrt(np.mean(q*q))),
            'maximum_nodal_distance_m':float(np.linalg.norm(q,axis=-1).max()),
            'velocity_rmse_m_s':float(np.sqrt(np.mean(v*v)))}
    iterations=[{'condition':name,'iterations':[lo,hi],'action_id':action,
        **compare((name,lo),(name,hi),action)} for name in ['original','recentered']
        for lo,hi in [(32,64),(64,128)] for action in ['zero_force','evaluation']]
    translations=[{'iterations':count,'action_id':action,
        **compare(('original',count),('recentered',count),action)} for count in [32,64,128] for action in ['zero_force','evaluation']]
    replay={}
    old=Path('runs/gaussian-campaign-v01/numerical-timestep-v01/hz960/captures/private/t0001')
    if ('original',32) in data:
        for action in ['zero_force','evaluation','reset_replay']:
            with np.load(old/action/'trajectory.npz') as raw:
                replay[action]={k:bool(np.array_equal(raw[k],data[('original',32)][action][k])) for k in ['positions_m','velocities_m_s','force_step_commands_N']}
    result={'cells':rows,'iteration_pairs':iterations,'translation_pairs':translations,
        'unshifted32_vs_old960_bitwise':replay,'input_sha256':hashes,'source_sha256':sha(__file__),
        'qualification':'Includes every predeclared cell and its failure. One TRAIN layout, no gate/continuum claim. Composed USD attributes are not independent native-backend verification. Observed q/v/rigid/time/count invariance does not expose every hidden solver cache or kinematic target flag.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='runs/gaussian-campaign-v01/numerical-solver-coordinates-v02')
    p.add_argument('--output',default='runs/gaussian-campaign-v01/numerical-solver-coordinates-v02/failure-aware-analysis.json')
    r=run(**vars(p.parse_args()));print(json.dumps({'cells':len(r['cells']),'failed':sum(c['state']!='completed_verified' for c in r['cells'])}))
