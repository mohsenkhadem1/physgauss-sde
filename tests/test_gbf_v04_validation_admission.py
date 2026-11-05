import json,os
from pathlib import Path
import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02 import softvt_validation_inputs as api
from scripts.gaussian_boundary_flow_v02 import softvt_validation_score as scorer
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import predictor_digest
from scripts.gaussian_boundary_flow_v02.export_inputs import write_package


def protocol(root,source):
    stat=source.stat()
    recipe=root/'recipe.json';recipe.write_text('{}')
    d={'schema':1,'status':'frozen','role':'validation','cases':[{'case':f'{s}-task{t}-demo{e}','subset':s,'task':t,'episode':e,'role':'validation'}for s,t,e in api.CASES],
       'recipe_path':str(recipe),'recipe_sha256':api.sha(recipe),'source_sha256':{api.__file__:api.sha(api.__file__),str(recipe):api.sha(recipe)},
       'source_files':{f'{s}/task{t}':{'path':str(source),'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}for s in ['object-soft','spatial-soft']for t in range(3)},
       'access_ledger':str(root/'ledger.jsonl')}
    path=root/'protocol.json';path.write_text(json.dumps(d));return path,api.sha(path),d


def fixture(root):
    h5py=pytest.importorskip('h5py');source=root/'object-soft/libero_object/libero_object_task1/replayed_demos/test.hdf5';source.parent.mkdir(parents=True)
    rest=np.array([[x,y,z]for x in (0.,.04)for y in (0.,.04)for z in (0.,.04)])
    with h5py.File(source,'w')as f:
        g=f.create_group('data/demo_50');g.attrs['asset_name']='toy'
        g.create_dataset('states/deformable_object/toy/nodal_pos_w',data=np.stack([rest+[.001*t,0,0]for t in range(10)]))
        g.create_dataset('obs/eef_pose',data=np.tile(np.r_[rest.mean(0),1.,0,0,0],(10,1)))
        g.create_dataset('obs/gripper_width',data=np.ones(10)*.02);g.create_dataset('obs/gripper_binary',data=-np.ones(10))
        g.create_dataset('obs/gripper_net_force',data=np.ones((10,3))*42)
        g.create_dataset('obs/fem_deformation_rms',data=np.ones(10)*999)
        g.create_dataset('initial_state/articulation/robot/root_pose',data=[[0.,0,0,1,0,0,0]])
    path,digest,d=protocol(root,source);return source,path,digest,d


@pytest.mark.parametrize('episode',[0,49,52,59,60,69,70,99])
def test_unlisted_and_private_roles_rejected_before_any_file_access(tmp_path,episode):
    with pytest.raises(PermissionError):api.load_predictor('object-soft',1,episode,protocol_path=tmp_path/'absent',protocol_sha256='bad',root=tmp_path)


def test_protocol_hash_status_and_role_fail_before_hdf_open(tmp_path,monkeypatch):
    source=tmp_path/'placeholder';source.touch();path,digest,d=protocol(tmp_path,source)
    monkeypatch.setattr(api,'_softvt_path',lambda *a:pytest.fail('Should not even locate HDF5'))
    with pytest.raises(PermissionError,match='hash'):api.admission('object-soft',1,50,protocol_path=path,protocol_sha256='b'*64)
    for key,value in [('status','draft'),('role','confirmation')]:
        changed={**d,key:value};path.write_text(json.dumps(changed))
        with pytest.raises(PermissionError):api.admission('object-soft',1,50,protocol_path=path,protocol_sha256=api.sha(path))


def test_prefix_slicing_and_private_target_mutation_invariance(tmp_path,monkeypatch):
    source,path,digest,d=fixture(tmp_path);import h5py
    reads=[];original=h5py.Dataset.__getitem__
    def logged(dataset,key):reads.append((dataset.name,key));return original(dataset,key)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',logged)
    first=api.load_predictor('object-soft',1,50,protocol_path=path,protocol_sha256=digest,root=tmp_path)
    assert first['data_role']=='prospective_validation'and len(first['observed'])==6
    object_reads=[key for name,key in reads if name.endswith('nodal_pos_w')]
    assert len(object_reads)==1 and object_reads[0].stop==6
    assert not any('force'in name or'deformation'in name for name,key in reads)
    stat=source.stat()
    with h5py.File(source,'a')as f:
        f['data/demo_50/states/deformable_object/toy/nodal_pos_w'][6:]=1000
        f['data/demo_50/obs/gripper_net_force'][:]=-999
        f['data/demo_50/obs/fem_deformation_rms'][:]=-999
    # Controlled sensitivity test: preserve file-stat identity; production never changes source files.
    os.utime(source,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    second=api.load_predictor('object-soft',1,50,protocol_path=path,protocol_sha256=digest,root=tmp_path)
    assert predictor_digest(first)==predictor_digest(second)
    ledger=[json.loads(line)for line in (tmp_path/'ledger.jsonl').read_text().splitlines()]
    assert len(ledger)==4 and all(row['role']=='validation'for row in ledger)
    assert not any('force'in key or'material'in key for key in second)


def test_evaluator_requires_seal_and_package_identity_before_suffix_access(tmp_path,monkeypatch):
    source,path,digest,d=fixture(tmp_path)
    data=api.load_predictor('object-soft',1,50,protocol_path=path,protocol_sha256=digest,root=tmp_path)
    package=tmp_path/'package';write_package(data,package);cohort=tmp_path/'cohort';run=cohort/data['name']/'run';run.mkdir(parents=True)
    from scripts.gaussian_boundary_flow_v02.validation_cohort import STAGES,terminal_gate
    for suite,task,episode in api.CASES:
        case=f'{suite}-task{task}-demo{episode}';directory=cohort/case;directory.mkdir(exist_ok=True)
        (directory/'prediction-status.json').write_text(json.dumps({'case':case,'protocol_sha256':digest,'terminal':True,'stages':{name:{'state':'complete'}for name in STAGES}}))
    (cohort/'evaluation-gate.json').write_text(json.dumps({'protocol_sha256':digest,'all12_terminal_before_any_target_read':True,'cases':terminal_gate(cohort,digest)}))
    kwargs={'protocol_path':path,'protocol_sha256':digest,'cohort_root':cohort,'root':tmp_path}
    with pytest.raises(ValueError,match='not complete'):scorer.score_case(run,**kwargs)
    plan={'case':data['name'],'input_package':str(package),'input_package_manifest_sha256':api.sha(package/'manifest.json'),'input_digest':predictor_digest(data),
          'args':{'nodes':len(data['nodes'])},'prediction_methods':['baseline_raw'],'prefix_frames':[0,5],'point_ids':list(range(len(data['observed_initial'])))}
    (run/'plan.json').write_text(json.dumps(plan))
    np.savez(run/'prediction.npz',baseline_raw=np.broadcast_to(data['nodes'],(10,*data['nodes'].shape)),**{k:data[k]for k in ['nodes','observed_initial','observation_matrix']})
    seal={'forecast_target_access':False,'train_end':6,'plan_sha256':api.sha(run/'plan.json')}
    complete={'completed':True,'held_target_access':False,'fit_end':6,'frames':10,'prediction_sha256':api.sha(run/'prediction.npz')}
    (run/'inference-sealed.json').write_text(json.dumps(seal));(run/'complete.json').write_text(json.dumps(complete))
    result=scorer.score_case(run,**kwargs)
    assert result['data_role']=='prospective_validation'and result['interval']==[6,10]
    assert result['methods']['baseline_raw']['summary']['valid_frames']==4
    import h5py
    monkeypatch.setattr(h5py,'File',lambda *a,**k:pytest.fail('Unverified predictor must not open evaluation arrays'))
    plan['input_digest']='changed';(run/'plan.json').write_text(json.dumps(plan));seal['plan_sha256']=api.sha(run/'plan.json');(run/'inference-sealed.json').write_text(json.dumps(seal))
    with pytest.raises(ValueError,match='predictor identity'):scorer.score_case(run,**kwargs)


def test_recipe_file_binding_and_standalone_baseline(tmp_path,monkeypatch):
    from scripts.gaussian_boundary_flow_v02 import validation_baseline as baseline
    source,path,digest,d=fixture(tmp_path)
    data=api.load_predictor('object-soft',1,50,protocol_path=path,protocol_sha256=digest,root=tmp_path)
    package=tmp_path/'package';manifest=api.write_validation_package(data,package,digest)
    assert manifest['data_role']=='prospective_validation'and manifest['scope'].startswith('Prospective validation')
    monkeypatch.setattr(baseline,'admission',lambda *a,**kw:None)
    out=tmp_path/'baseline';baseline.prepare(package,out,path,digest)
    with np.load(out/'prediction.npz',allow_pickle=False)as f:
        assert f['sealed_prefix_persistence'].shape[0]==10
        assert f['sealed_prefix_velocity'].shape[1]==len(data['observed_initial'])
    assert json.loads((out/'complete.json').read_text())['held_target_access']is False
    Path(d['recipe_path']).write_text('changed')
    with pytest.raises(PermissionError,match='recipe file'):api.admission('object-soft',1,50,protocol_path=path,protocol_sha256=digest,root=tmp_path)
