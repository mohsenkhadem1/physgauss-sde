from collections import Counter
from scripts.complex_scenes_v18.held_protocol import draft
from scripts.complex_scenes_v18.held_tasks import build
from scripts.complex_scenes_v18.run_development import ROOT


def test_all_scenario_tasks_exist_and_predictions_never_depend_on_test_references():
    p=draft();worlds=[]
    for family in p['families']:
        for i in range(family['worlds']):worlds.append(dict(**family,world_id=f'{family["family"]}-{i}',refinement=i==0))
    tasks=build(p,dict(worlds=worlds),ROOT/'runs/complex-scenes-v18/not-generated-test',ROOT/'reports/complex-scenes-v18/not-generated-freeze.json')
    counts=Counter(t['role']for t in tasks)
    assert counts=={'reference':72,'reference-refinement':12,'prefix':24,'fit':100,'prediction':200}
    by_id={t['id']:t for t in tasks}
    def ancestors(key):
        result=set()
        for dep in by_id[key]['dependencies']:result.add(dep);result.update(ancestors(dep))
        return result
    for t in tasks:
        if t['role']=='prediction':
            reference=[by_id[k]for k in ancestors(t['id'])if by_id[k]['role']=='reference']
            assert len(reference)==1 and reference[0]['id'].endswith(':calibration')
            assert '--fitdir'in t['argv']and '--query-json'not in t['argv']
