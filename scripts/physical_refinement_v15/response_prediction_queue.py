"""Commit hypothesis-only forecasts for new response poses; no truth reader."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.physical_refinement_v15 import development_mechanics_queue as worker
from scripts.physical_refinement_v15.response_actions import solve_actions
from scripts.physical_refinement_v15.quadrature import geometry_cases
from scripts.physical_refinement_v15.mechanics_adapter import specification
from scripts.physical_refinement_v15.mechanics_pilot import BASE,REPORT,write,sha
OUT=BASE/'response-predictions';MANIFEST=OUT/'manifest.json'


def prepare():
    actions=solve_actions();poses=np.asarray(list(actions.values()));jobs=[];sources={}
    cases=geometry_cases(17)
    for case in cases:
        name=case['name'];physical={k:v for k,v in case.items() if k!='name'}
        path=OUT/'specs'/f'{name}.json'
        payload=dict(case=physical,poses=poses.tolist(),action_names=list(actions))
        if path.exists() and json.loads(path.read_text())!=payload:raise ValueError('Changed response specification')
        write(path,payload)
        for n in ([4,6] if np.isclose(abs(case['material_gradient']),.8) else [4]):
            sources.update(specification(physical,poses,n,3)['source_sha256'])
            jobs.append(dict(id=f'{name}-n{n}',name=name,n=n,spec=str(path.relative_to(ROOT)),
                             out=str((OUT/'hypotheses'/name/f'n{n}').relative_to(ROOT)),budget_seconds=600))
    for path in (Path(__file__),ROOT/'scripts/physical_refinement_v15/response_actions.py',
                 ROOT/'scripts/physical_refinement_v15/quadrature.py',ROOT/'scripts/physical_refinement_v15/study.py',
                 ROOT/'scripts/physical_refinement_v15/development_mechanics_queue.py'):
        sources[str(path.relative_to(ROOT))]=sha(path)
    result=dict(scope='Hypothesis forecasts on response-only actions; never acquisition utility or actual reserved truth',
                cases=cases,jobs=jobs,action_names=list(actions),poses_m_rad=poses.tolist(),
                source_sha256=sources,spec_sha256={j['spec']:sha(ROOT/j['spec']) for j in jobs},
                qualification='n4/n6 at eight gradient/layout/ratio corners; internal-node range assumption',
                gates=worker.GATES,workers=3,devices=['cuda:0','cuda:1','cuda:2'])
    if MANIFEST.exists() and json.loads(MANIFEST.read_text())!=result:raise ValueError('Changed response prediction manifest')
    write(MANIFEST,result);return result


def main():
    m=prepare();worker.OUT=OUT;worker.MANIFEST=MANIFEST
    worker.phase(m['jobs'],m)
    checks=[worker.compare(j['name'],4,6,m) for j in m['jobs'] if j['n']==6]
    states=[json.loads((OUT/'jobs'/f"{j['id']}.json").read_text()) for j in m['jobs']]
    write(REPORT/'response-prediction-qualification.json',dict(scope=m['scope'],manifest_sha256=sha(MANIFEST),
          comparisons=checks,all_spatial_gates_pass=all(c['passed'] for c in checks),
          jobs=states,failures=[s['job']['id'] for s in states if s['state']!='complete']))


if __name__=='__main__':prepare() if '--prepare-only' in sys.argv else main()
