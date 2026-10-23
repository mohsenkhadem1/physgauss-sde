"""Temporary synthetic files only: no campaign observations or noise draws."""
import json
import numpy as np
import pytest
from experiments import run_campaign_c39_side_observations as m
from test_campaign_c38_public_prefix_inference_v02 import design,payload


def write(root,name,value):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return m.io.record(root,p)


def fixture(root,monkeypatch):
 d=write(root,'extension.json',{});monkeypatch.setattr(m.public,'DESIGN_PATH',d['path']);monkeypatch.setattr(m.public,'DESIGN_SHA256',d['sha256'])
 initial=write(root,'initial.json',{});monkeypatch.setattr(m,'INITIAL_SEAL_PATH',initial['path']);monkeypatch.setattr(m,'INITIAL_SEAL_SHA256',initial['sha256'])
 write(root,'source.py',{});monkeypatch.setattr(m,'sources',lambda:['source.py'])
 old=write(root,m.oldobs.DESIGN_PATH,design());monkeypatch.setattr(m.oldobs,'DESIGN_SHA256',old['sha256'])
 ids=[format(i,'032x')for i in range(64)];prefixrows=[];mapping=[];original=[]
 for i,token in enumerate(ids):
  prefix=payload(token);p=root/'oldPUBLIC'/(token+'.npz');p.parent.mkdir(exist_ok=True);m.oldcodec.exclusive_npz(p,m.oldcodec.encode_public(prefix,design=design()));r=m.io.record(root,p)
  prefixrows.append(dict(object_id=token,prefix=dict(path=p.name,sha256=r['sha256'])));mapping.append(dict(object_id=token,private_configuration_index=i//16,repeat_index=i%16));original.append(dict(object_id=token,public_prefix=r,private_future=dict(path='NEVER_OPEN_PRIVATE_TARGET',sha256='f'*64)))
 manifest=write(root,'oldPUBLIC/manifest.json',dict(schema='C38-public-controlled-ideal-prefixes-v01',completed=True,information_role='PUBLIC_ACQUIRED_IDEAL_STATIC_POINTS_B',count=64,physical_configurations=4,noise_repeats_per_configuration=16,independent_population_objects=False,design_sha256=old['sha256'],is_RGBD_perception=False,is_temporal_forecast=False,records=prefixrows))
 configs=[dict(private_configuration_index=i,young_Pa=E,support_category=b)for i,(E,b)in enumerate(m.oldobs.controlled_configurations())]
 identity=write(root,'oldPRIVATE/identity-map.json',dict(mapping_created_before_first_noise=True,configurations=configs,records=mapping))
 original=write(root,'oldPRIVATE/execution.json',dict(completed=True,identity_mapping_persisted_before_noise=True,public_manifest=manifest,records=original,identity_mapping=identity))
 rows=[(2*i+j,c,n)for i,c in enumerate(m.side.side_catalog())if c['material_use']=='WITHHELD_INTERPOLATION_VALIDATION'for j,n in enumerate(m.side.MESHES)]
 a=dict(source_index=np.array([r[0]for r in rows]),case_ids=np.array([r[1]['id']for r in rows]),material_use=np.array(['WITHHELD_INTERPOLATION_VALIDATION']*8),information_role=np.asarray('C39_SIDE_PRIVATE_WITHHELD_POINT_VALIDATION_NOT_FEATURES'),cells=np.array([r[2]for r in rows]),support=np.array([r[1]['support']for r in rows]),young_Pa=np.array([r[1]['young_Pa']for r in rows]),action_id=np.asarray(m.side.SIDE),force_vector_N=np.array([0.,2.,0.]),material_probes_m=m.side.POINTS.copy(),probe_positions_m=np.array([m.side.POINTS+np.array([0.,.001*(i+1),0.])for i in range(8)]))
 np.savez(root/'private-SIDE.npz',**a);archives=dict(PRIVATE_WITHHELD_SIDE=m.io.record(root,root/'private-SIDE.npz'),CANDIDATE_SIDE=write(root,'candidate.placeholder',{}),LINEAR_SIDE=write(root,'linear.placeholder',{}))
 validation=write(root,'validation.json',dict(completed=True,validation=dict(completed=True,finite_checks_passed=True,tolerance_m=1e-6,checks=[dict(passed=True)for _ in range(8)]),archives=archives))
 actions=dict(information=m.side.SIDE,target_risk='middle_down_2N',fixed_down='middle_down_2N',fixed_side=m.side.SIDE,random=m.side.SIDE);choices=[];modelrows={n:[]for n in m.io.MODELS}
 for token in ids:
  model=write(root,'models/'+token+'.json',dict(completed=True));models={name:dict(choices=actions,result=model)for name in m.io.MODELS};r=write(root,'choices/'+token+'.json',dict(completed=True,object_id=token,models=models));choices.append(dict(object_id=token,choices=actions,result=r))
  for name in m.io.MODELS:modelrows[name].append(dict(object_id=token,**models[name]))
 choices=write(root,'choices.json',dict(schema='C39-typed-static-policy-choices-seal-v01',completed=True,count=64,model='nonlinear_table',model_executions=128,extension_design=d,initial_prediction_seal=initial,private_SIDE_outcomes_access=False,acquired_DOWN_observations_or_posteriors_access=False,private_target_or_latent_access=False,all_choices_sealed_before_SIDE_noise=True,forward_point_validation=validation,candidate_side_archive_sha256=archives['CANDIDATE_SIDE']['sha256'],linear_side_archive_sha256=archives['LINEAR_SIDE']['sha256'],action_catalog=[dict(action_id=k,force_world_N=v)for k,v in [('middle_down_2N',[0.,0.,-2.]),(m.side.SIDE,[0.,2.,0.])]],choices=choices,model_choices=modelrows))
 return dict(validation=validation['path'],choices=choices['path'],public_manifest=manifest['path'],original_execution=original['path'])


def test_prepare_is_metadata_only_and_exact_choice_library_required(tmp_path,monkeypatch):
 args=fixture(tmp_path,monkeypatch);monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('arrays before execution')));monkeypatch.setattr(m.public,'side_measurement_error',lambda *a:(_ for _ in ()).throw(AssertionError('noise during prepare')))
 h=m.prepare('protocol.json','out',root=tmp_path,**args);p=m.validate('protocol.json',h,root=tmp_path);assert len(p['inputs']['rows'])==64 and not(tmp_path/'out').exists()
 with pytest.raises(FileExistsError):m.prepare('protocol.json','out',root=tmp_path,**args)
 c=json.loads((tmp_path/'choices.json').read_text());c['linear_side_archive_sha256']='f'*64;write(tmp_path,'badchoices.json',c)
 with pytest.raises(ValueError,match='ExactsameSIDE'):m.prepare('badprotocol.json','badout',root=tmp_path,**dict(args,choices='badchoices.json'))


def test_complete_run_reuses_all_ids_prefixes_groups_and_newSIDE_noise_only(tmp_path,monkeypatch):
 args=fixture(tmp_path,monkeypatch);h=m.prepare('protocol.json','out',root=tmp_path,**args);monkeypatch.setattr(m.sideio,'guard',lambda root:None);calls=[]
 def fake(c,r):calls.append((c,r));return np.full((9,3),.00001*(r+1))
 monkeypatch.setattr(m.public,'side_measurement_error',fake);real=m.io.checked;opened=[]
 def checked(root,r):
  assert 'NEVER_OPEN'not in r['path'];opened.append(r['path']);return real(root,r)
 monkeypatch.setattr(m.io,'checked',checked);state=m.run('protocol.json',h,root=tmp_path)
 assert state['completed']and state['private_configuration_counts']=={str(i):16 for i in range(4)}and calls==[(c,r)for c in range(4)for r in range(16)]
 manifest=json.loads((tmp_path/'out/PUBLIC/manifest.json').read_text());assert len(manifest['records'])==64 and manifest['same_original_identities_and_prefixes']and not manifest['private_labels_or_targets_included']
 for row in manifest['records']:
  with np.load(tmp_path/'out/PUBLIC'/row['acquisition']['path'],allow_pickle=False)as z:
   assert set(z.files)==m.public.PUBLIC_KEYS and str(z['action_id'])==m.side.SIDE and not{'force_Fz_N','young_Pa','support','private_configuration_index','repeat_index'}&set(z.files)
 assert state['mapping_bound_before_first_SIDE_noise']and not state['new_IDs_drawn']and not state['new_DOWN_noise_drawn']
 with pytest.raises(FileExistsError):m.run('protocol.json',h,root=tmp_path)
 assert len(calls)==64


def test_failure_is_preserved_and_cannot_regenerate_partial_noise(tmp_path,monkeypatch):
 args=fixture(tmp_path,monkeypatch);h=m.prepare('protocol.json','out',root=tmp_path,**args);monkeypatch.setattr(m.sideio,'guard',lambda root:None);calls=[]
 def fail(c,r):
  calls.append((c,r))
  if len(calls)==2:raise RuntimeError('manufactured acquisition failure')
  return np.zeros((9,3))
 monkeypatch.setattr(m.public,'side_measurement_error',fail)
 with pytest.raises(RuntimeError):m.run('protocol.json',h,root=tmp_path)
 state=json.loads((tmp_path/'out/PRIVATE/execution.json').read_text());assert not state['completed']and len(state['records'])==1 and state['failure']['type']=='RuntimeError'and state['mapping_bound_before_first_SIDE_noise']
 assert not(tmp_path/'out/PUBLIC/manifest.json').exists()
 with pytest.raises(FileExistsError):m.run('protocol.json',h,root=tmp_path)
 assert len(calls)==2


def test_wrong_typed_catalog_or_original_prefix_binding_rejected_before_noise(tmp_path,monkeypatch):
 args=fixture(tmp_path,monkeypatch);c=json.loads((tmp_path/'choices.json').read_text());c['action_catalog'][1]['force_world_N']=[0.,0.,2.];write(tmp_path,'wrongchoices.json',c)
 with pytest.raises(ValueError,match='worldvectors'):m.prepare('p.json','out',root=tmp_path,**dict(args,choices='wrongchoices.json'))
 original=json.loads((tmp_path/'oldPRIVATE/execution.json').read_text());original['records'][0]['public_prefix']=original['records'][1]['public_prefix'];write(tmp_path,'wrongexecution.json',original)
 with pytest.raises(ValueError,match='prefixbytes'):m.prepare('q.json','out',root=tmp_path,**dict(args,original_execution='wrongexecution.json'))


@pytest.mark.parametrize('tamper',['missing','identity','action','result','missing_model'])
def test_linear_all64_choice_closure_required_before_any_noise(tmp_path,monkeypatch,tamper):
 args=fixture(tmp_path,monkeypatch);c=json.loads((tmp_path/'choices.json').read_text());linear=c['model_choices']['linear_rest_tangent']
 if tamper=='missing':linear.pop()
 elif tamper=='identity':linear[63]['object_id']=linear[62]['object_id']
 elif tamper=='action':linear[63]['choices']['information']='middle_down_2N'
 elif tamper=='result':linear[63]['result']=linear[62]['result']
 else:del c['model_choices']['linear_rest_tangent']
 write(tmp_path,'badlinear.json',c);monkeypatch.setattr(m.public,'side_measurement_error',lambda *a:(_ for _ in ()).throw(AssertionError('noise before BOTH closures')))
 with pytest.raises(ValueError,match='BOTHmodel'):m.prepare('p.json','out',root=tmp_path,**dict(args,choices='badlinear.json'))
 assert not(tmp_path/'p.json').exists()and not(tmp_path/'out').exists()
