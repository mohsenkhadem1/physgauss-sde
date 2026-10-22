import json,os
from pathlib import Path
import numpy as np
import pytest
from experiments import run_campaign_c38_graded_family_warp as m


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'source_closure',lambda:['source.py']);(tmp_path/'source.py').write_text('# synthetic\n')
    env=tmp_path/'installed';(env/'warp').mkdir(parents=True);(env/'warp/a.py').write_text('# synthetic API\n');monkeypatch.setattr(m,'ENV_SITE',env)
    def put(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return {'path':name,'sha256':m.sha(p)}
    hvp_sources={}
    for name in ['experiments/campaign_p2_torch_warp_hvp.py','experiments/campaign_p2_warp_bulk_hvp.py']:
        f=tmp_path/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('# synthetic accepted source\n');hvp_sources[name]=m.sha(f)
    old=put('incident.json',{'original_full_case_JSON_recovered':False});selected=[];caches=[]
    for i,case in enumerate(c for c in m.all_cases() if c['id'] in m.REUSED_CASE_IDS):
        r=put(f'selected-{i}.json',{'completed':True});cached=tmp_path/f'cached-{i}.json';cached.write_bytes((tmp_path/r['path']).read_bytes());caches.append({'original_path':r['path'],'cache_path':cached.name,'sha256':r['sha256']});meshes=[]
        for n in (32,40):
            cap=tmp_path/f'capture-{i}-{n}.npz';cap.write_bytes(b'No array reads during preparation'+bytes([i,n]));meshes.append({'cells':[n,n//2,n//2],'capture':{'path':cap.name,'sha256':m.sha(cap)}})
        selected.append({'case':case,'case_result':r,'mesh_records':meshes})
    values={'base_family_protocol':{'resources':{'physical_GPU_UUID':m.GPU_UUID},'gates':m.GATES},'original14_summary':{'completed':True,'cases_completed':14},'original14_terminal':{'completed':True},'coarse_identity':{'passed':True,'rows':[{'case':c,'passed':True} for c in m.all_cases()]},'selected_summary':{'completed':True,'selected_two_case_finest_displacement_indicators_passed':True,'rows':selected,'metadata_lineage':{'incident':old}},'selected_terminal':{'all_owned_PIDs_absent':True,'all_source_pins_unchanged':True},'selected_cache':{'completed_cases_cached':caches},'HVP_protocol':{'source_sha256':hvp_sources,'CPU_affinity_cores':sorted(os.sched_getaffinity(0))[:2],'installed_warp_API_sha256':{'warp/a.py':m.sha(env/'warp/a.py')}},'HVP_result':{'completed':True,'passed':True},'HVP_terminal':{'completed':True}}
    for k,name in m.METADATA_INPUTS.items():put(name,values[k])
    values['HVP_terminal']={'all_source_pins_match':True,'owned_PIDs_absent':{'123':True},'protocol_sha256':m.sha(tmp_path/m.METADATA_INPUTS['HVP_protocol'])}
    put(m.METADATA_INPUTS['HVP_terminal'],values['HVP_terminal'])
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(k,'2')


def test_exact24new_and4reused_configuration_no_exclusions():
    assert len(m.all_cases())==14 and len(m.cases())==12
    assert set(c['id'] for c in m.cases()).isdisjoint(m.REUSED_CASE_IDS)
    assert m.MESHES==((32,16,16),(40,20,20))
    assert m.GATES['conditional_actual_h_spatial_indicator_m']==20e-6


def test_exclusive_destination_only_metadata_preparation_and_full_input_immutability(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()};dest=tmp_path/'new.json';original_open=Path.open
    def guard(path,mode='r',*a,**kw):
        if any(x in mode for x in 'wax+'):assert path==dest and mode=='x'
        return original_open(path,mode,*a,**kw)
    monkeypatch.setattr(Path,'open',guard);monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No arrays during preparation')))
    digest=m.prepare(dest,tmp_path/'new-out');p=m.validate(dest,digest)
    assert p['fine_field_counts']=={'new':24,'reused':4,'total':28}
    assert len(p['reused_fine_fields'])==4 and p['resources']['global_wall_cap_s']==10800 and p['resources']['per_case_wall_cap_s']==900
    assert p['first_full_nonlinear_case_id']==m.cases()[0]['id']
    assert all(path.read_bytes()==value for path,value in before.items())
    with pytest.raises(ValueError):m.prepare(dest,tmp_path/'other')


def test_changed_reused_capture_rejected_before_arrays(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);dest=tmp_path/'new.json';digest=m.prepare(dest,tmp_path/'new-out');(tmp_path/'capture-0-32.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='capture changed'):m.validate(dest,digest)


def test_missing_first_case_result_retains_primary_failure_and_unlaunched_queue(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'guard_resources',lambda *a,**k:None);monkeypatch.setattr(os,'sched_getaffinity',lambda _:{0,1})
    class Child:
        pid=123456
        def __init__(self,*a,**kw):pass
        def wait(self,timeout=None):return 0
        def poll(self):return 0
    monkeypatch.setattr(m.subprocess,'Popen',Child)
    protocol=tmp_path/'protocol.json';protocol.write_text('{}');cases=m.cases()[:2]
    p={'output':'new-run','cases':cases,'all_original_cases':m.all_cases(),'reused_fine_fields':[],'CPU_affinity_cores':[0,1],'taskset':{'path':'/usr/bin/taskset'},'first_full_nonlinear_case_id':cases[0]['id'],'resources':{'global_wall_cap_s':10800,'per_case_wall_cap_s':900,'physical_GPU_UUID':m.GPU_UUID}}
    r=m.run(p,protocol)
    assert not r['completed'] and not r['first_case_gate_passed']
    assert len(r['records'])==1 and r['queued_case_ids']==[cases[1]['id']]
    assert 'technical/physicalfailure' in r['failure'] and 'controller_failure' not in r
