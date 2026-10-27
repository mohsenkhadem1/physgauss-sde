"""Enable the Q2 fallback only after preregistered development checks pass."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.phystwin.qualified_reference import BASE,ROOT,ACTIONS,protocol,checked_compare,write_json
from scripts.phystwin.qualified_tensor_reference import LEVELS


def main():
    spec=protocol();rule=json.loads((BASE/'tensor-candidate-protocol.json').read_text());assert rule['levels']==LEVELS
    audits={str(i):json.loads((BASE/'tensor-qualification'/f'development-{i}.json').read_text()) for i in [1,11]}
    assert all(a['qualified'] for a in audits.values()),'Both development fixtures must qualify'
    p2=json.loads((BASE/'qualification/development-11.json').read_text());assert p2['qualified'] and p2.get('space','P2')=='P2'
    cross=[]
    for action in ACTIONS:
        a=BASE/'refinements/development-11'/f'n{p2["latest_mesh"]}-q3-{action}.npz'
        b=BASE/'tensor-refinements/development-11'/f'n{audits["11"]["latest_mesh"]}-q3-{action}.npz'
        with np.load(a) as az,np.load(b) as bz:cross.append(dict(action=action,**checked_compare(az,bz,spec,spec['development'][11],action)))
    a=BASE/'tensor-refinements/development-1'/f'n{audits["1"]["latest_mesh"]}-q3-initial.npz'
    b=BASE/'tensor-edges-diagnostic/development-1/n20-initial.npz'
    with np.load(a) as az,np.load(b) as bz:finer=checked_compare(az,bz,spec,spec['development'][1],'initial')
    sources=['scripts/phystwin/tensor_fixture.py','scripts/phystwin/consistent_fixture.py','scripts/phystwin/gpu_multigrid.py','scripts/phystwin/qualified_reference.py','scripts/phystwin/qualified_tensor_reference.py','experiments/campaign_static_p2_block.py']
    validation=dict(passed=all(c['pass'] for c in cross) and finer['pass'],candidate_audits=audits,qualified_p2_cross_comparison=cross,
        additional_finer_initial_check=dict(case='development-1',coarse=audits['1']['latest_mesh'],fine=20,**finer),
        source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        scope='Two development fixtures, four loads each, unchanged accuracy gates. Finite numerical qualification, not a mathematical continuum-error bound.')
    path=BASE/'tensor-validation.json';write_json(path,validation)
    assert validation['passed'],'Independent qualified P2 comparison failed; fallback not enabled'
    adoption=dict(validated=True,p2_maximum_mesh=28,half_face_reference='Q2_edge',validation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        selection='Preserve already-qualified P2 references. New half-face support references use edge-refined Q2 directly; full-face supports retain P2 through n=28 with Q2 fallback after spatial or quadrature failure. Identical gates apply. Never substitute for unexpected solver errors. This is geometry-dependent simulator meshing; fixture labels remain hidden from inference and probing policies.',
        timing='Development-only numerical method selection, before any calibration/test response generation or inference.',
        hypothesis_bank='Unchanged P2 bank and development-only surrogate selection; Q2 changes reference discretization, not material/force/boundary laws.')
    write_json(BASE/'tensor-reference-adoption.json',adoption);print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
