"""Private seeded truth generation AFTER explicit protocol/source freeze.

No policy imports this module. Public protocol contains names/cameras and a
prior distribution only. Draws remain independent; no balancing hidden classes,
no resampling to fill empty strata, no snapping continuous parameters to grid.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.physical_refinement_v15 import development_mechanics_queue as worker
from scripts.physical_refinement_v15.study import development_actions
from scripts.physical_refinement_v15.response_actions import response_actions
from scripts.physical_refinement_v15.mechanics_adapter import specification
from scripts.physical_refinement_v15.mechanics_pilot import BASE,REPORT,write,sha
OUT=BASE/'reserved';MANIFEST=OUT/'mechanics-manifest.json'


def guard(protocol_path,protocol_sha,freeze_path,freeze_sha):
    if sha(protocol_path)!=protocol_sha or sha(freeze_path)!=freeze_sha:raise ValueError('Protocol/freeze hash mismatch')
    protocol=json.loads(Path(protocol_path).read_text());freeze=json.loads(Path(freeze_path).read_text())
    if freeze['protocol_sha256']!=protocol_sha:raise ValueError('Freeze binds different protocol')
    sources=freeze['source_sha256']
    required=[Path(__file__),ROOT/'scripts/physical_refinement_v15/response_actions.py',
              ROOT/'scripts/physical_refinement_v15/study.py',ROOT/'scripts/physical_refinement_v15/development_mechanics_queue.py']
    if any(str(p.relative_to(ROOT)) not in sources for p in required):raise ValueError('Required queue dependencies not frozen')
    if any(sha(ROOT/p)!=h for p,h in sources.items()):raise ValueError('Frozen source changed')
    return protocol,freeze


def sample_cases(worlds,distribution,seed):
    if not isinstance(seed,int):raise ValueError('Integer truth seed required')
    if any(set(w)!={'name','initial_camera'} for w in worlds):raise ValueError('Public worlds contain metadata only')
    if len({w['name'] for w in worlds})!=len(worlds):raise ValueError('Unique world names required')
    if any(not w['name'].replace('-','').replace('_','').isalnum() for w in worlds):raise ValueError('Unsafe world name')
    g=np.asarray(distribution['gradient_uniform'],float);s=np.asarray(distribution['common_scale_log_uniform'],float)
    layouts=distribution['layouts'];ratios=distribution['fixture_ratios']
    if g.shape!=(2,) or s.shape!=(2,) or not np.isfinite(np.r_[g,s]).all() or not g[0]<g[1] or not 0<s[0]<s[1]:raise ValueError('Finite ordered sampling ranges required')
    if set(layouts)!={'parallel_lower','crossed'} or len(layouts)!=2 or len(ratios)!=2 or not np.all(np.isfinite(ratios)) or min(ratios)<=0:raise ValueError('Two supported layouts and positive fixture ratios required')
    rng=np.random.default_rng(seed);cases=[]
    for world in worlds:
        gradient=float(rng.uniform(*g));scale=float(np.exp(rng.uniform(*np.log(s))))
        layout=layouts[int(rng.integers(len(layouts)))];ratio=float(ratios[int(rng.integers(len(ratios)))])
        if gradient in np.linspace(-.8,.8,17):raise ValueError('Unexpected exact grid draw; do not snap or silently resample')
        case=dict(E=float(distribution['E_base'])*scale,Kleft=float(distribution['K_base'])*scale*ratio,
                  Kright=float(distribution['K_base'])*scale*ratio,layout=layout,material_gradient=gradient)
        cases.append(dict(**world,case=case,common_scale=scale,fixture_ratio=ratio))
    return cases


def prepare(protocol_path,protocol_sha,freeze_path,freeze_sha):
    protocol,freeze=guard(protocol_path,protocol_sha,freeze_path,freeze_sha)
    worlds=protocol['reserved_worlds']
    if len(worlds)!=12 or sorted(w['initial_camera'] for w in worlds)!=['left']*6+['oblique']*6:raise ValueError('Fixed twelve public worlds/six cameras each required')
    cases=sample_cases(worlds,protocol['truth_distribution'],protocol['truth_seed'])
    private=dict(role='PRIVATE evaluator truth; not policy input',protocol_sha256=protocol_sha,freeze_sha256=freeze_sha,
                 seed=protocol['truth_seed'],draw_order='per world: gradient, logscale, layout, fixture_ratio; independent RNG draws',worlds=cases)
    truth_path=OUT/'truth-specs.json'
    if truth_path.exists() and json.loads(truth_path.read_text())!=private:raise ValueError('Private truth specification changed')
    actions=dict(list(development_actions().items())[:5]);actions.update(response_actions());poses=np.asarray(list(actions.values()))
    if len(actions)!=9:raise ValueError('Exactly initial/four probes/four response poses required')
    jobs=[];specs={}
    # Validate complete mechanics dependency subset before creating any truth artifacts.
    for world in cases:
        for n in (4,6):
            required=specification(world['case'],poses,n,3)['source_sha256']
            if any(freeze['source_sha256'].get(k)!=v for k,v in required.items()):raise ValueError('Mechanics dependency absent/changed in freeze')
    write(truth_path,private)
    for world in cases:
        path=OUT/'specs'/f"{world['name']}.json";value=dict(case=world['case'],poses=poses.tolist(),action_names=list(actions))
        if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Changed mechanics job specification')
        write(path,value);specs[str(path.relative_to(ROOT))]=sha(path)
        for n in (4,6):jobs.append(dict(id=f"{world['name']}-n{n}",name=world['name'],n=n,
            spec=str(path.relative_to(ROOT)),out=str((OUT/'worlds'/world['name']/f'n{n}').relative_to(ROOT)),budget_seconds=600))
    m=dict(scope='Reserved private truth plus per-world mesh companions; all failures retained',
           protocol_sha256=protocol_sha,freeze_sha256=freeze_sha,truth_spec_sha256=sha(truth_path),
           source_sha256={**freeze['source_sha256'],str(Path(protocol_path).resolve().relative_to(ROOT)):protocol_sha,
                          str(Path(freeze_path).resolve().relative_to(ROOT)):freeze_sha},
           spec_sha256=specs,jobs=jobs,action_names=list(actions),gates=worker.GATES,workers=3,devices=['cuda:0','cuda:1','cuda:2'])
    if MANIFEST.exists() and json.loads(MANIFEST.read_text())!=m:raise ValueError('Changed reserved queue manifest')
    write(MANIFEST,m);return m


def compare(world,action_names):
    rows=[]
    for i,action in enumerate(action_names):
        folders=[OUT/'worlds'/world/f'n{n}'/f'pose-{i:03d}' for n in (4,6)]
        if not all((p/'complete.json').exists() for p in folders):rows.append(dict(action=action,passed=False,reason='Missing completed mesh companion'));continue
        for folder in folders:
            record=json.loads((folder/'complete.json').read_text())
            if any(sha(folder/k)!=v for k,v in record['artifact_sha256'].items()):raise ValueError('Changed truth companion artifact')
        arrays=[]
        for folder in folders:
            with np.load(folder/'private-state.npz') as data:field=data['field'].copy()
            with np.load(folder/'surface.npz') as data:surface=data['vertices'].copy()
            with np.load(folder/'reaction-wrench.npz') as data:force=data['forces_N'].copy();torque=data['torques_Nm'].copy()
            arrays.append((field,surface,force,torque))
        a,b=arrays;diff=[x-y for x,y in zip(a,b)];f=np.linalg.norm(diff[2],axis=1);t=np.linalg.norm(diff[3],axis=1)
        metrics=dict(field_rms_m=float(np.sqrt(np.mean(diff[0]**2))),field_max_m=float(abs(diff[0]).max()),
                     surface_rms_m=float(np.sqrt(np.mean(diff[1]**2))),surface_max_m=float(abs(diff[1]).max()),
                     force_difference_N=f.tolist(),torque_difference_Nm=t.tolist())
        finite=all(np.isfinite(x).all() for values in arrays for x in values)
        passed=finite and all(metrics[k]<=worker.GATES[k] for k in ('field_rms_m','field_max_m','surface_rms_m','surface_max_m')) and bool((f<=worker.GATES['force_floor_N']+worker.GATES['force_relative']*np.linalg.norm(b[2],axis=1)).all() and (t<=worker.GATES['torque_floor_Nm']+worker.GATES['torque_relative']*np.linalg.norm(b[3],axis=1)).all())
        rows.append(dict(action=action,passed=bool(passed),**metrics))
    return dict(world=world,mesh_pair=[4,6],rows=rows,passed=all(r['passed'] for r in rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True);a=p.parse_args()
    m=prepare(a.protocol,a.protocol_sha256,a.freeze,a.freeze_sha256);worker.OUT=OUT;worker.MANIFEST=MANIFEST;worker.phase(m['jobs'],m)
    guard(a.protocol,a.protocol_sha256,a.freeze,a.freeze_sha256)
    states=[json.loads((OUT/'jobs'/f"{j['id']}.json").read_text()) for j in m['jobs']]
    checks=[compare(j['name'],m['action_names']) for j in m['jobs'] if j['n']==6]
    write(REPORT/'reserved-mechanics-qualification.json',dict(scope=m['scope'],manifest_sha256=sha(MANIFEST),protocol_sha256=a.protocol_sha256,
        freeze_sha256=a.freeze_sha256,jobs=states,failures=[s['job']['id'] for s in states if s['state']!='complete'],
        comparisons=checks,all_spatial_gates_pass=all(c['passed'] for c in checks),
        limitation='Mechanical mesh checks do not establish rendered sensor or posterior quadrature convergence'))
if __name__=='__main__':main()
