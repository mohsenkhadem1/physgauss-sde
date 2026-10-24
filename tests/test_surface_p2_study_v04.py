import json,os,subprocess,zipfile
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pytest
from experiments import run_surface_p2_study_v04 as r


def test_prefix_only_reader_two_laws_and_allowlist(tmp_path):
    file=tmp_path/'truth.npz';a=np.arange(240,dtype=float).reshape(10,24);noise=np.arange(24,dtype=float)
    np.savez_compressed(file,matching_linear=a,NH80_stress=a+1000,prefix_noise=noise,probe_noise=np.ones((3,24)))
    np.testing.assert_array_equal(r.selected_prefix_observation(file,'matching_linear',4),a[4]+noise)
    np.testing.assert_array_equal(r.selected_prefix_observation(file,'NH80_stress',4),a[4]+1000+noise)
    with pytest.raises(ValueError):r.selected_prefix_observation(file,'probe_noise',4)


def test_orphan_posterior_fails_before_inference(tmp_path):
    path=tmp_path/'toy-prefix-LOCAL.json';np.savez(path.with_suffix('.npz'),mean=np.zeros(1))
    with pytest.raises(ValueError,match='before inference'):r.one_posterior(path,None,[],r.AmplitudePlan())


def test_full_object_causal_grouped_path_preserves_tenload_target_indices(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'LOCAL',tmp_path);monkeypatch.setattr(r,'environment_guard',lambda *a,**k:None)
    labels=[f'{family}_x{round(c*1000):03d}'for c in(.09,.135,.18,.225,.27)for family in('normal','paired_axial')]
    fields=np.arange(240,dtype=float).reshape(10,24)*1e-5
    def reuse(index,directory,p,digest):
        np.savez(directory/'truth-LOCAL.npz',matching_linear=fields,NH80_stress=fields+.01,prefix_noise=np.zeros(24),probe_noise=np.zeros((3,24)),unit=np.array([.3,.7]),random_realization=np.array([0,2]))
        r.write(directory/'truth-checkpoint-LOCAL.json',dict(labels=labels,all_physical_valid=True,raw_SHA256=r.sha(directory/'truth-LOCAL.npz')))
    monkeypatch.setattr(r,'reuse_truth_fields',reuse);np.savez(tmp_path/'prior-predictive-LOCAL.npz',mean=np.zeros(96))
    observations=[]
    def infer(path,model,data,plan,**kwargs):
        observations.append(data)
        record=dict(qualified=True)
        if path.name.endswith('-prefix-LOCAL.json'):
            np.savez(path.with_suffix('.npz'),group_log_ratio=np.array([-6.]),group_probability=np.ones(1),group_amplitude_mean=np.array([.0002]),group_amplitude_variance=np.array([1e-10]))
            record['raw_SHA256']=r.sha(path.with_suffix('.npz'))
        r.write(path,record);return record
    monkeypatch.setattr(r,'one_posterior',infer)
    def utilities(grouped,predict,actions,targets):
        assert tuple(targets)==r.study.TARGETS
        return [dict(expected_linear_target_MSE_reduction_m2=float(i),Gaussian_observation_information_nats=float(2-i))for i in range(3)]
    monkeypatch.setattr(r,'grouped_utilities',utilities)
    targets=[]
    def score(path,target,unit):
        law='matching_linear'if path.name.startswith('matching_linear')else'NH80_stress'
        assert path.with_name(law+'-choices-seal-LOCAL.json').exists()
        targets.append(target.copy());return dict(resolved=True,posterior_RMS_m=0.,MAP_RMS_m=0.)
    monkeypatch.setattr(r,'posterior_score',score)
    original_load=np.load
    class CheckedArchive:
        def __init__(self,z,path):self.z=z;self.path=path
        def __enter__(self):return self
        def __exit__(self,*args):self.z.close()
        def __getitem__(self,key):
            if key in r.LAWS:assert (self.path.parent/(key+'-choices-seal-LOCAL.json')).exists(),'Futurelawtruth read before its own prefixseal'
            return self.z[key]
    def checked_load(path,*args,**kwargs):
        archive=original_load(path,*args,**kwargs)
        return CheckedArchive(archive,Path(path))if Path(path).name=='truth-LOCAL.npz'else archive
    monkeypatch.setattr(np,'load',checked_load)
    p=dict(device_mode='cpu',numerical_engine='conditional_amplitude_PWL_v02',amplitude_plan=asdict(r.AmplitudePlan()))
    result=r.process_object(0,None,None,p,'a'*64,0,ratio_predict=lambda z,l:None)
    assert result['completed']and result['numerically_resolved']and len(observations)==38 and len(targets)==38
    expected=fields[[labels.index(l)for l in r.study.TARGETS]].ravel()
    for target in targets[:19]:np.testing.assert_array_equal(target,expected)
    for target in targets[19:]:np.testing.assert_array_equal(target,expected+.01)


def test_reused_bytes_and_origin_context_tamper_fail_closed(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();monkeypatch.setattr(r,'ROOT',root)
    source=root/'source.npz';np.savez(source,unit=np.array([.2,.8]));meta=root/'meta.json';r.write(meta,dict(raw_SHA256=r.sha(source)))
    entry=dict(index=0,truth_file='source.npz',truth_SHA256=r.sha(source),metadata_file='meta.json',metadata_SHA256=r.sha(meta))
    out=root/'new';out.mkdir();p=dict(old_data=dict(objects=[entry]));r.reuse_truth_fields(0,out,p,'b'*64)
    assert r.sha(out/'truth-LOCAL.npz')==r.sha(source)
    assert r.sha(out/'truth-checkpoint-LOCAL.json')==r.sha(meta)
    r.reuse_truth_fields(0,out,p,'b'*64)
    with pytest.raises(ValueError,match='provenance context changed'):r.reuse_truth_fields(0,out,p,'c'*64)
    (out/'truth-LOCAL.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='Copied original'):r.reuse_truth_fields(0,out,p,'b'*64)


def test_archive_schema_prefix_and_lean_joint(tmp_path):
    from experiments.campaign_surface_inverse_reference_v01 import ModalStaticReference
    from experiments.campaign_surface_p2_inference_v01 import LabelledModal
    labels=[r.study.PREFIX]+list(r.study.TARGETS)
    modal=ModalStaticReference(np.diag([1.,2.]),np.diag([.2,.3]),np.ones((2,len(labels)))*1e-4,np.eye(2));model=LabelledModal(modal,labels)
    data=[(r.study.PREFIX,np.array([1e-6,2e-6]))];plan=r.AmplitudePlan((128,256),(32,64))
    for filename,retain in(('law-prefix-LOCAL.json',True),('law-action-r0-LOCAL.json',False)):
        path=tmp_path/filename;record=r.one_posterior(path,model,data,plan)
        assert record['qualified']
        with np.load(path.with_suffix('.npz'))as z:
            assert {'mean','SD','unit_mean','unit_covariance','unit_CDF','unit_quantiles','MAP_unit','MAP_prediction','MAP_log_likelihood'}<=set(z.files)
            assert ('unit_grid'in z.files)==retain and ('group_probability'in z.files)==retain
        assert r.one_posterior(path,model,data,plan)==record


def test_actual_generated_CPU_GPU_worker_guards():
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for mode in('cpu','gpu'):
        limits=r.resources_for(mode)
        for slot in range(16):
            childenv=dict(env,CUDA_VISIBLE_DEVICES=''if mode=='cpu'else limits['worker_GPU_UUIDs'][slot])
            x=subprocess.run(r.worker_command(slot,'0'*64,guard_only=True,device_mode=mode),cwd=r.ROOT,env=childenv,capture_output=True,text=True,timeout=20)
            assert x.returncode==0,x.stderr
            assert json.loads(x.stdout)['guard_passed']


def test_execution_failure_preserves_admitted_source_physics(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'LOCAL',tmp_path)
    p=dict(device_mode='cpu',numerical_engine='conditional_amplitude_PWL_v02',resources=dict(workers=16))
    monkeypatch.setattr(r,'validate',lambda digest:p);monkeypatch.setattr(r,'environment_guard',lambda *a,**k:None);monkeypatch.setattr(r,'release_gate',lambda digest:None)
    monkeypatch.setattr(r,'cached_model',lambda *a,**k:(p,None,None));monkeypatch.setattr(r,'ratio_callback',lambda model:None)
    def failed(*args,**kwargs):raise OSError('synthetic interrupted IO')
    monkeypatch.setattr(r,'process_object',failed);r.worker(0,'d'*64)
    record=json.loads((tmp_path/'object-00-LOCAL/terminal-LOCAL.json').read_text())
    assert record['physical_valid']and record['execution_failure']and not record['completed']and not record['numerically_resolved']
