"""Manufactured20q63/4linear fields in tempROOT; no actual physics/data."""
from pathlib import Path
import json
import numpy as np
import pytest
from experiments import prepare_campaign_c39_side_points as m


def write(root,name,value):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return m.io.record(root,p)


def fixture(root,monkeypatch,*,bad_withheld=False):
 design=write(root,'design.json',{});catalog=write(root,'catalog.json',{})
 monkeypatch.setattr(m,'DESIGN_PATH',design['path']);monkeypatch.setattr(m,'DESIGN_SHA256',design['sha256']);monkeypatch.setattr(m,'CATALOG_PATH',catalog['path']);monkeypatch.setattr(m,'CATALOG_SHA256',catalog['sha256']);monkeypatch.setattr(m,'sources',lambda:['source.py']);source=write(root,'source.py',{})
 summary=[]
 for index,c0 in enumerate(m.side.side_catalog()):
  c=dict(c0,Fz_N=0.);meshes=[];x=75000/c['young_Pa'];factor=1 if c['support']=='both_ends'else 1.3
  for n in m.side.MESHES:
   q=m.side.POINTS+np.array([.0001*x*x,.001*x*factor,.0002*x*x]);q=q.copy()
   if bad_withheld and c['material_use']=='WITHHELD_INTERPOLATION_VALIDATION':q[11,1]+=.000002
   p=root/'fields'/str(index)/('m'+str(n[0])+'.npz');p.parent.mkdir(parents=True,exist_ok=True);np.savez(p,material_probes_m=m.side.POINTS,probe_positions_m=q,positions_m=np.full((100,3),999.));meshes.append(dict(cells=list(n),capture=m.io.record(root,p),completed=True,physics_passed=True))
  raw=dict(case=c,completed=True,physics_passed=True,passed=True,meshes=meshes);r=write(root,'cases/'+str(index)+'.json',raw);cache=root/'cache'/str(index);cache.parent.mkdir(exist_ok=True);cache.write_bytes((root/r['path']).read_bytes());summary.append(dict(case=c,source_result=r,external_cache=m.io.record(root,cache),completed=True,physics_passed=True,passed=True,meshes=meshes))
 summed=write(root,'summary.json',dict(completed=True,all10_physics_and_q63_passed=True,rows=summary))
 lp=write(root,'linear-proto.json',dict(young_Pa=75000.,source_sha256={'source.py':source['sha256']}));monkeypatch.setattr(m,'LINEAR_PROTOCOL_PATH',lp['path']);monkeypatch.setattr(m,'LINEAR_PROTOCOL_SHA256',lp['sha256'])
 records=[];compact=[]
 for b in m.side.CATEGORIES:
  models=[]
  for n in m.side.MESHES:
   p=root/'linear'/b/('m'+str(n[0])+'.npz');p.parent.mkdir(parents=True,exist_ok=True);np.savez(p,material_probes_m=m.side.POINTS,middle_side_2N=np.full((500,3),999.))
   models.append(dict(cells=list(n),capture=m.io.record(root,p),responses=[dict(action=m.side.SIDE,displacement63_m=np.full((63,3),.001).tolist(),load_metadata=dict(action_id=m.side.SIDE,force_vector_N=[0.,2.,0.]))]))
  result=write(root,'linear/'+b+'/result.json',dict(completed=True,passed=True,models=models));cache=root/'cache'/b;cache.write_bytes((root/result['path']).read_bytes());records.append(dict(support=b,result=result,external_cache=str(cache.relative_to(root)),completed=True,passed=True,exit_code=0));compact.append(dict(LOCAL_full_response_result=result,models=[dict(cells=x['cells'],capture=x['capture'])for x in models]))
 le=write(root,'linear-exec.json',dict(completed=True,terminal=True,all_numeric_passed=True,all_q63_spatial_passed=True,protocol_sha256=lp['sha256'],records=records));lc=write(root,'linear-compact.json',dict(completed=True,all_numeric_passed=True,all_q63_spatial_passed=True,support_summaries=compact))
 return dict(summary=summed['path'],catalog=catalog['path'],linear_protocol=lp['path'],linear_execution=le['path'],linear_compact=lc['path'])


def test_prepare_metadataonly_complete20barrier_and_exclusive_tempROOT(tmp_path,monkeypatch):
 args=fixture(tmp_path,monkeypatch);monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Arraysduringprepare')))
 h=m.prepare('protocol.json','out',root=tmp_path,**args);p=m.validate('protocol.json',h,root=tmp_path);assert len(p['metadata']['nonlinear_rows'])==20 and len(p['metadata']['linear_rows'])==4 and not(tmp_path/'out').exists()
 with pytest.raises(FileExistsError):m.prepare('protocol.json','out',root=tmp_path,**args)


def test_last_linear_capture_tamper_prevents_firstNLpoint_read(tmp_path,monkeypatch):
 args=fixture(tmp_path,monkeypatch);h=m.prepare('protocol.json','out',root=tmp_path,**args);p=m.validate('protocol.json',h,root=tmp_path);last=p['metadata']['linear_rows'][-1]['capture'];(tmp_path/last['path']).write_bytes(b'changedlastfile')
 monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('NoarraybeforeALL24hashes')))
 with pytest.raises(ValueError,match='Changed input'):m.extract_and_validate(tmp_path,p)


def test_only_q63_members_extracted_and_allwithheld_checks_retained(tmp_path,monkeypatch):
 args=fixture(tmp_path,monkeypatch);h=m.prepare('protocol.json','out',root=tmp_path,**args);p=m.validate('protocol.json',h,root=tmp_path);real=np.load;keys=[]
 class Spy:
  def __init__(self,z):self.z=z
  def __enter__(self):return self
  def __exit__(self,*a):self.z.close()
  def __getitem__(self,k):assert k in('material_probes_m','probe_positions_m');keys.append(k);return self.z[k]
 monkeypatch.setattr(np,'load',lambda *a,**k:Spy(real(*a,**k)))
 candidate,private,linear,check=m.extract_and_validate(tmp_path,p)
 assert candidate['probe_positions_m'].shape==(12,63,3)and private['probe_positions_m'].shape==(8,63,3)and len(linear)==4 and len(keys)==44
 assert check['finite_checks_passed']and len(check['checks'])==8 and check['maximum_withheld_point_error_m']<1e-15 and not check['continuous_material_certificate']


def test_fixed1um_failure_is_not_refit_or_dropped(tmp_path,monkeypatch):
 args=fixture(tmp_path,monkeypatch,bad_withheld=True);h=m.prepare('protocol.json','out',root=tmp_path,**args);p=m.validate('protocol.json',h,root=tmp_path);a,b,linear,check=m.extract_and_validate(tmp_path,p)
 assert not check['finite_checks_passed']and len(check['checks'])==8 and check['maximum_withheld_point_error_m']>1e-6
 assert len(b['source_index'])==8 and len(a['source_index'])==12
