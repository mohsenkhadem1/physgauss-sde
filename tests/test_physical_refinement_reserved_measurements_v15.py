import json
import numpy as np
import pytest
from scripts.physical_refinement_v15 import reserved_measurements as m
from scripts.physical_refinement_v15.measurement_broker import MeasurementBroker

def test_shared_noise_exact_across_modalities_and_repeats():
 C=np.eye(64)*.001;mean=np.arange(64)*.01
 joint=np.zeros((70,70));joint[:64,:64]=C;joint[64:,64:]=np.diag([.01**2]*3+[.001**2]*3)
 kwargs=dict(world_index=0,repeat=0,action='probe_left_twist',camera='left')
 a=m.packet(mean,C,**kwargs);b=m.packet(np.r_[mean,np.ones(6)],joint,**kwargs)
 np.testing.assert_array_equal(a,b[:64]);np.testing.assert_array_equal(a,m.packet(mean,C,**kwargs))
 assert not np.array_equal(a,m.packet(mean,C,**{**kwargs,'repeat':1}))
 with pytest.raises(ValueError):m.packet(mean,np.full((64,64),np.nan),**kwargs)

def test_world_packets_only_observations_no_design_reads(tmp_path,monkeypatch):
 actions=list(m.development_actions())[:5];calls=[]
 def committed(folder,filename,evidence):
  index=int(folder.name.split('-')[-1]);calls.append((index,filename))
  return tmp_path/'wrench.npz',dict(pose_index=index,pose_m_rad=m.development_actions()[actions[index]].tolist())
 monkeypatch.setattr(m,'committed',committed)
 monkeypatch.setattr(m,'render_geometry',lambda *a:({},np.arange(64,dtype=float)))
 np.savez(tmp_path/'wrench.npz',forces_N=np.ones((2,3)),torques_Nm=np.ones((2,3))*2,origins_world_m=np.array([[-.0005,0,0],[.2405,0,0]]))
 bank=dict(actions=actions,cameras=m.banks.CAMERAS,features=np.zeros((1,5,3,64)),wrenches=np.zeros((1,5,6)),covariances=np.tile(np.eye(64)[None,None],(5,3,1,1)))
 output=tmp_path/'packets';result=m.world_packets(dict(name='reserved-00',initial_camera='left'),0,bank,output)
 assert len(result['manifests'])==8
 assert max(i for i,_ in calls)==4
 assert all(i>0 for i,f in calls if f=='reaction-wrench.npz')
 for repeat in range(4):
  manifests={t:output/f'reserved-00/repeat-{repeat}/{t}/manifest.json'for t in ('V','VF')}
  brokers={t:MeasurementBroker(p,m.sha(p),packet_root=output)for t,p in manifests.items()}
  np.testing.assert_array_equal(brokers['V'].request('initial'),brokers['VF'].request('initial'))
  np.testing.assert_array_equal(brokers['V'].request('probe_left_twist'),brokers['VF'].request('probe_left_twist')[:64])
 for p in output.rglob('*.npz'):
  with np.load(p)as data:assert data.files==['observation']
 np.savez(tmp_path/'wrench.npz',forces_N=np.ones((2,3)),torques_Nm=np.ones((2,3)),origins_world_m=np.zeros((2,3)))
 with pytest.raises(ValueError,match='origin mismatch'):
  m.world_packets(dict(name='reserved-00',initial_camera='left'),0,bank,tmp_path/'bad-packets')
 assert not (tmp_path/'bad-packets').exists()
 # No production directory is touched by this synthetic test.

def test_freeze_guard_rejects_source_mutation_before_generation(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'ROOT',tmp_path)
 # Supply the module's own relative identity under the temporary root without loading any real observations.
 fake_module=tmp_path/'reserved_measurements.py';fake_module.write_text('frozen source')
 monkeypatch.setattr(m,'__file__',str(fake_module))
 protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(dict(reserved_worlds=[dict(name=f'reserved-{i:02d}',initial_camera='left')for i in range(12)],measurement_seed=931127,noise_repeats=4)))
 freeze=tmp_path/'freeze.json';freeze.write_text(json.dumps(dict(protocol_sha256=m.sha(protocol),source_sha256={'reserved_measurements.py':m.sha(fake_module)},artifact_sha256={})))
 psha,fsha=m.sha(protocol),m.sha(freeze)
 m.guard(protocol,psha,freeze,fsha)
 fake_module.write_text('changed')
 with pytest.raises(ValueError,match='source changed'):m.guard(protocol,psha,freeze,fsha)


def test_freeze_guard_rejects_nominal_bank_artifact_mutation(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'ROOT',tmp_path)
 source=tmp_path/'reserved_measurements.py';source.write_text('source')
 monkeypatch.setattr(m,'__file__',str(source))
 artifact=tmp_path/'nominal-camera.npz';artifact.write_bytes(b'committed-image')
 protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(dict(reserved_worlds=[dict(name=f'reserved-{i:02d}',initial_camera='left')for i in range(12)],measurement_seed=931127,noise_repeats=4)))
 freeze=tmp_path/'freeze.json';freeze.write_text(json.dumps(dict(protocol_sha256=m.sha(protocol),source_sha256={source.name:m.sha(source)},artifact_sha256={artifact.name:m.sha(artifact)})))
 psha,fsha=m.sha(protocol),m.sha(freeze)
 m.guard(protocol,psha,freeze,fsha)
 artifact.write_bytes(b'changed-image')
 with pytest.raises(ValueError,match='artifact changed'):m.guard(protocol,psha,freeze,fsha)
