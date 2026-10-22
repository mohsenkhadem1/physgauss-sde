"""All preparation fixtures are temporary: no campaign artifact may be written."""
import json
from pathlib import Path
import numpy as np
import pytest
from experiments import run_campaign_c38_static_graded_refinement_v02 as m


def temporary_prior(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();monkeypatch.setattr(m,'ROOT',root)
    monkeypatch.setattr(m,'source_closure',lambda:['dummy.py'])
    (root/'dummy.py').write_text('# independent synthetic source\n')
    def put(name,obj):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj));return {'path':name,'sha256':m.sha(p)}
    uniform=put('uniform.json',{'completed':True})
    prior_rows=[]
    for i,case in enumerate(m.cases()):
        cap=root/f'capture-{i}.npz';cap.write_bytes(b'not opened during metadata preparation'+bytes([i]));record={'path':cap.name,'sha256':m.sha(cap)}
        obj={'case':case,'completed':True,'physics_passed':True,'meshes':[{'cells':[20,10,10],'seconds':20.},{'cells':[24,12,12],'seconds':40.,'capture':record}],'coarse_CPU_GPU':{'passed':True}}
        r=put(f'case-{i}.json',obj);prior_rows.append({'case':case,'result':r})
    base='runs/operator-dev/c38-static-graded-pilot-v01/'
    put(base+'protocol-v01.json',{'resources':{'CPU_threads':2},'gates':m.GATES,'mapping':m.MAPPING,'existing_uniform_matrix':uniform,'uniform_terminal_proof':uniform,'domain_or_image_cohort_admitted':False})
    put(base+'completed-summary.json',{'completed':True,'spatial_cases_failed':2,'rows':prior_rows})
    put(base+'terminal-proof.json',{'all_owned_PIDs_absent':True,'all_source_pins_unchanged':True})
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(key,'2')
    return root


def test_prepare_writes_only_exclusive_requested_destination_and_preserves_every_input(tmp_path,monkeypatch):
    root=temporary_prior(tmp_path,monkeypatch)
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    destination=root/'new-protocol.json';original_open=Path.open;writes=[]
    def guarded_open(path,mode='r',*args,**kwargs):
        if any(c in mode for c in 'wax+'):
            assert path.resolve()==destination.resolve(), 'Attempted write outside exact requested protocol'
            assert mode=='x', 'Protocol creation must be exclusive'
            writes.append(path)
        return original_open(path,mode,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded_open)
    monkeypatch.setattr(np,'load',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('No capture arrays during preparation')))
    digest=m.prepare(destination,root/'fresh-output');p=m.validate(destination,digest)
    assert writes==[destination] and len(p['reference_mesh24_captures'])==2
    assert p['resources']['global_wall_cap_s']==1800 and p['resources']['per_case_wall_cap_s']==900
    assert all(path.read_bytes()==value for path,value in before.items())
    assert not (root/'fresh-output').exists()
    with pytest.raises(FileExistsError):m.prepare(destination,root/'other-output')
    assert all(path.read_bytes()==value for path,value in before.items())


def test_existing_evidence_cannot_be_protocol_destination(tmp_path,monkeypatch):
    root=temporary_prior(tmp_path,monkeypatch);case=root/'case-0.json';before=case.read_bytes()
    with pytest.raises(FileExistsError):m.prepare(case,root/'fresh')
    assert case.read_bytes()==before


def test_reused_capture_hash_and_probes_checked_before_read(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);points=np.arange(189).reshape(63,3)/1000
    np.savez(tmp_path/'old24.npz',material_probes_m=points,probe_positions_m=points+1)
    capture={'path':'old24.npz','sha256':m.sha(tmp_path/'old24.npz')}
    (tmp_path/'case.json').write_text(json.dumps({'meshes':[{'cells':[24,12,12],'capture':capture}],'coarse_CPU_GPU':{'passed':True}}))
    p={'reference_case_results':{'case':{'path':'case.json','sha256':m.sha(tmp_path/'case.json')}},'reference_mesh24_captures':{'case':capture}}
    _,Q,_=m.reused_reference(p,'case',points);np.testing.assert_array_equal(Q,points+1)
    with pytest.raises(ValueError,match='Original63'):m.reused_reference(p,'case',points+.001)
    (tmp_path/'old24.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed before read'):m.reused_reference(p,'case',points)


def test_fixed_new_resolutions_leave_physics_and_original_threshold_unchanged():
    assert m.MESHES==((32,16,16),(40,20,20)) and len(m.cases())==2
    assert m.GATES['conditional_actual_h_spatial_indicator_m']==20e-6
