import copy,json
import numpy as np
import pytest
from scripts.compositional_neural_mechanics import dataset_runner as r

def plan():
 scene=dict(name='tiny-smoke',geometry=dict(dimensions=[.2,.08,.06],taper=0.,arch=0.),bulk_family='mooney_rivlin',bulk_latent=[0.,0.,-1.],gradient=0.,factors={'heterogeneity':0},interface_latents=[[np.log(6000),.2,0.],[np.log(6000),.2,0.]],centers=[[.5,.25],[.5,.25]],widths=[[.3,.3],[.3,.3]],gravity=[0.,0.,0.],density_kg_m3=125.)
 pose=np.zeros((2,6));pose[1,0]=.0005
 return dict(n=1,order=3,material_query=[[.2,.5,.5],[.8,.3,.4]],continuation=[.25,.5,.75,1.],force_tolerance_N=1e-9,per_scene_timeout_seconds=30.,stage_timeout_seconds=60.,devices=['cpu'],scenes=[dict(role='train',scene=scene,actions=[dict(name='small-stretch',pose=pose.tolist())])],source_sha256=r.source_inventory())

def test_reject_reserved_changed_sources_and_invalid_output(tmp_path):
 p=plan();bad=copy.deepcopy(p);bad['scenes'][0]['role']='reserved'
 with pytest.raises(ValueError,match='forbidden'):r.validate(bad)
 bad=copy.deepcopy(p);bad['devices']=['cuda:3']
 with pytest.raises(ValueError,match='GPU0'):r.validate(bad)
 bad=copy.deepcopy(p);bad['source_sha256']={}
 with pytest.raises(ValueError,match='inventory'):r.validate(bad)
 path=tmp_path/'plan.json';path.write_text(json.dumps(p));out=tmp_path/'already';out.mkdir()
 with pytest.raises(ValueError,match='Unrecognized'):r.prepare(path,out)

def test_subprocess_smoke_verified_resume_and_mutation(tmp_path):
 p=plan();path=tmp_path/'plan.json';path.write_text(json.dumps(p));out=tmp_path/'dataset'
 first=r.run(path,out);assert first['counts']['complete']==1
 folder=out/'scenes/tiny-smoke/small-stretch'
 with np.load(folder/'labels.npz')as data:assert set(data.files)=={'material_query','displacement','wrenches'}and data['wrenches'].shape==(2,6)
 record=json.loads((folder/'complete.json').read_text());assert len(record['continuation'])==4 and record['residual_N']<=p['force_tolerance_N']
 before=r.sha(folder/'labels.npz');second=r.run(path,out)
 assert second['rows'][0]['cached']and r.sha(folder/'labels.npz')==before
 (folder/'labels.npz').write_bytes(b'corrupted')
 with pytest.raises(ValueError,match='artifact changed'):r.run(path,out)

def test_external_timeout_keeps_declared_failed_rows(tmp_path):
 p=plan();p['per_scene_timeout_seconds']=.001;path=tmp_path/'plan.json';path.write_text(json.dumps(p));out=tmp_path/'timeout'
 result=r.run(path,out)
 assert result['counts']['timeout']==1
 assert len(result['rows'][0]['actions'])==1
 assert result['rows'][0]['actions'][0]['state']=='timeout'
 assert len(list((out/'scene-attempts/tiny-smoke').glob('*.json')))==1
