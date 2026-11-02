from scripts.compositional_dynamics_v17 import cohort_appearance as r
from scripts.compositional_dynamics_v17 import cohort_appearance_transport as transport

def test_failed_native_world_keeps310slots_and_shared_images(tmp_path,monkeypatch):
    queries=[]
    for world in range(4):
        for role in ['forecast-40mm','forecast-80mm']:
            queries.append(dict(world_id=f'world-{world}',role=role,methods=['shared_q2']if world==3 else ['native','shared_q2'],failed={'native':'empty graph'}if world==3 else {},times_s=[i*.01 for i in range(31)],cameras=[dict(name=f'c-{i}',role='train'if i<3 else'held')for i in range(5)]))
    def render(q,folder):
        return dict(gaussians=10,initial_rgb_max_difference=0 if len(q['methods'])==2 else None,records=[dict(method=m,frame=i,camera=c['name'],time_s=t,path='synthetic',sha256='synthetic')for m in q['methods']for i,t in enumerate(q['times_s'])for c in q['cameras']])
    monkeypatch.setattr(r,'guard',lambda:dict(queries=queries));monkeypatch.setattr(r,'BASE',tmp_path/'output');monkeypatch.setattr(r,'sha',lambda p:'synthetic');monkeypatch.setattr(transport,'render_query',render)
    r.predict();commit=r.read(r.BASE/'prediction-commit.json')
    assert len(commit['rows'])==2480
    assert sum(x['completed']for x in commit['rows'])==2170
    failed=[x for x in commit['rows']if not x['completed']]
    assert len(failed)==310 and all(x['world_id']=='world-3'and x['method']=='native'for x in failed)
    assert all(x['completed']for x in commit['rows']if x['method']=='shared_q2')
