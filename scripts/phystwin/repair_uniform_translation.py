"""Re-solve an unqualified analytic load using its exact equilibrium as a start.

This uses the sealed solver and gates unchanged. Original checkpoints and the
failed audit are retained; already-qualified references cannot be overwritten.
"""
import argparse,fcntl,hashlib,json,shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.phystwin.qualified_reference import BASE,ACTIONS,protocol,solve,release,checked_compare,write_json
from scripts.phystwin.qualified_tensor_reference import build
from scripts.phystwin.qualify_fixture import metrics
from scripts.phystwin.qualified_policy import verify_seal


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--levels',type=int,nargs='+',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    seal=json.loads((BASE/'analysis/seal.json').read_text());verify_seal(seal)
    spec=protocol();r=spec['test'][a.index];name=f'test-{a.index}'
    assert r['layout'].endswith('full'),'Only the exact full-face uniform-load equilibrium is eligible'
    assert not (BASE/'fields'/f'{name}-qualified.npz').exists(),'Do not overwrite an accepted reference'
    action='push_'+r['layout'].split('_')[0];directory=BASE/'tensor-refinements'/name
    out=BASE/'initialization-audits'/name;out.mkdir(parents=True,exist_ok=True)
    with open(BASE/'job-locks'/f'truth-{name}.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        audit=BASE/'tensor-qualification'/f'{name}.json'
        if not (out/'original-qualification.json').exists():shutil.copyfile(audit,out/'original-qualification.json')
        rows=[]
        for n in a.levels:
            path=directory/f'n{n}-q3-{action}.npz';backup=out/path.name
            with np.load(path) as z:old={k:z[k] for k in z.files}
            assert 'initialization_kind' not in old,'Checkpoint already repaired; inspect the existing audit'
            untouched={str(q.relative_to(BASE)):digest(q) for q in [directory/f'n{n}-q3-{other}.npz' for other in ACTIONS if other!=action]}
            m=build(n,r,a.device)
            try:
                start=time.time();m.set_action(action);initial=np.broadcast_to(m.load.sum(0)/r['K'],m.rest.shape).copy()
                state=solve(m,action,initial);u=state['displacement_local_m']
                row=dict(u=u,elapsed_seconds=time.time()-start,initialization_kind='exact_uniform_translation',**metrics(m,u,state))
                check=checked_compare(row,row,spec,r,action)
                delta=float(abs(u-old['u']).max())
                assert check['pass'],'The original unchanged absolute/residual gates must pass'
                assert delta<=1e-10,'Initialization must not materially change the physical solution'
                assert not backup.exists(),'Original checkpoint archive must not be overwritten'
                shutil.copyfile(path,backup);temporary=path.with_suffix('.exact.tmp.npz');np.savez_compressed(temporary,**row);temporary.replace(path)
                assert all(digest(BASE/q)==h for q,h in untouched.items())
                rows.append(dict(mesh=n,action=action,maximum_nodal_change_m=delta,unchanged_gate_result=check,
                    original_sha256=digest(backup),replacement_sha256=digest(path),unchanged_other_load_hashes=untouched))
                write_json(out/'repair.json',dict(case=name,scope='Exact analytic equilibrium supplied as initial state to the sealed solver; unchanged force residual and qualification gates. No model, prior, calibration, tolerance or deforming-load change.',rows=rows))
                print('EXACT_INITIALIZATION',name,n,delta,check['analytic_strain_error'],check['analytic_traction_error_Pa'],flush=True)
            finally:release(m)
    verify_seal(seal);print('SEALED_MODEL_UNCHANGED',flush=True)


if __name__=='__main__':main()
