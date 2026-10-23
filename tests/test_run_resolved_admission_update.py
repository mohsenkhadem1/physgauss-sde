import copy,json
import numpy as np
import pytest
from experiments import run_resolved_admission_update as m


def prediction():
 a={k:np.array([1.,2.])for k in m.FIELDS};a['material_quantile_details']=[dict(unit_bracket=dict(lower=.1,upper=.2))];a['completed']=True;a['canonical_quantile_details']=[dict(CDF_left='metadata not scored')]
 return dict(methods=dict(adaptive=a))


def test_exact_metric_input_projection_includes_quantile_arrays_and_material_brackets():
 a=prediction();b=copy.deepcopy(a);b['methods']['adaptive']['canonical_quantile_details']=[dict(CDF_left='refined arithmetic metadata')]
 assert all(m.metric_input_identity(a,b).values())
 b['methods']['adaptive']['material_quantile_details'][0]['unit_bracket']['upper']=.3
 with pytest.raises(ValueError,match='Changed adaptive'):m.metric_input_identity(a,b)
 b=copy.deepcopy(a);b['methods']['adaptive']['canonical_position_interval95_numerical_enclosure_m'][0]=np.nextafter(1.,2.)
 with pytest.raises(ValueError,match='Changed adaptive'):m.metric_input_identity(a,b)


def test_update_changes_only_two_admissions_and_preserves_old_rows_and_denominators():
 rows=[dict(object_id=str(i),admitted=i>=2,primary_prediction_available=True,numerical_failures=[dict(method='GL128')]if i<2 else[],private_configuration_index=i//16,repeat_index=i%16,retained_metric=i*.01)for i in range(64)]
 old=copy.deepcopy(rows);new=m.update_rows(rows,{'0','1'})
 assert rows==old and len(new)==64 and all(r['admitted']for r in new)
 for a,b in zip(old,new):
  if a['object_id']not in ('0','1'):assert a==b
  else:assert a['retained_metric']==b['retained_metric']and b['previous_numerical_failures']==a['numerical_failures']
 with pytest.raises(ValueError,match='All64'):m.update_rows(rows[:-1],{'0','1'})


def test_missing_256th_prediction_blocks_all_old_private_metric_content(tmp_path,monkeypatch):
 def put(n,v):
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return m.io.record(tmp_path,p)
 cp=put('current-protocol.json',{});put('current/execution.json',dict(completed=True,technical_failure=False,all_owned_workers_reaped=True,live=[],queued_job_ids=[],pid=9999998,records=[]));put('current/prediction-seal.json',dict(model_executions=256,all_numerical_controls_passed=True,model_records=[{}]*255))
 monkeypatch.setattr(m.repair,'validate',lambda*a,**k:dict(output='current'))
 p=put('p.json',dict(schema='RESOLVED-admission-only-update-v01',source_sha256={},runtime=m.io.runtime(),resources=m.RESOURCES,truth_or_observation_archive_access=False,current_protocol=cp,old_summary={'path':'PRIVATE-OLD-summary.json','sha256':'a'*64},old_metrics={'path':'PRIVATE-OLD-rows.json','sha256':'b'*64}))
 monkeypatch.setattr(m,'sources',lambda:[]);read=m.io.read_json;access=[]
 def spy(root,ref):
  access.append(ref['path'])
  if 'PRIVATE-OLD'in ref['path']:raise AssertionError('Private old metric content before complete prediction barrier')
  return read(root,ref)
 monkeypatch.setattr(m.io,'read_json',spy);monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('No arrays before fullcohort')))
 with pytest.raises(ValueError,match='ALL256'):m.run('p.json',p['sha256'],root=tmp_path)
 assert not any('PRIVATE-OLD'in x for x in access)
