import json
from pathlib import Path
import pytest
from experiments import campaign_c38_nonlinear_table_catalog as c


def test_exact40_material_force_counterfactuals_and_existing14_identity():
 rows=c.catalog();old={x['id']:x for x in c.original_cases()}
 assert len(rows)==40 and len({r['id'] for r in rows})==40
 assert sum(r['material_use']=='TABLE_NODE' for r in rows)==24
 assert sum(r['material_use']=='WITHHELD_INTERPOLATION_VALIDATION' for r in rows)==16
 assert sum(r['reused_original14_case'] for r in rows)==14
 for r in rows:
  assert 'evaluation_role' not in r and 'observation' not in r
  if r['reused_original14_case']:assert all(r[k]==old[r['id']][k] for k in old[r['id']])
 assert {r['Fz_N'] for r in rows}=={-2.,0.,1.,2.}
 assert sorted({r['log_material_fraction'] for r in rows})==[0.,.25,.5,.75,1.]
 for E in {r['young_Pa'] for r in rows}:assert len([r for r in rows if r['young_Pa']==E])==8


def test_three_disjoint_unique_future_gpu_partitions_without_resource_lease():
 phases=c.phase_plan(range(8));ids=[i for p in phases for i in p['case_ids']]
 assert len(ids)==len(set(ids))==26 and [p['new_case_count'] for p in phases]==[9,9,8]
 assert len(set(p['output'] for p in phases))==3 and len(set(p['gpu_uuid'] for p in phases))==3
 assert len({k for p in phases for k in p['CPU_affinity_cores']})==6
 assert sum(p['new_resolution_count'] for p in phases)==52
 assert all('NOT_GPU_LEASE' in p['status'] for p in phases)
 with pytest.raises(ValueError):c.phase_plan([0,1,2,3,4])


def test_draft_cannot_cross_physics_write_boundary(tmp_path,monkeypatch):
 monkeypatch.setattr(c,'ROOT',tmp_path);f=tmp_path/'draft.json';f.write_text(json.dumps(dict(schema='C38-static-nonlinear-table-CATALOG-DRAFT-v0.1',status='DESIGN_SOURCE_ONLY_NOT_PHYSICAL_PROTOCOL_NOT_LAUNCH_AUTHORIZATION')))
 with pytest.raises(ValueError,match='catalogDRAFT'):c.dispatch_worker('draft.json',c.immutable.sha(f),'anything')
 assert set(tmp_path.iterdir())=={f}
 for bad in ['/absolute','../escape','a/../b','a//b']:
  with pytest.raises(ValueError):c.repo_path(bad)


def test_reuse_requires_exact_fields_external_bytes_and_linear_provenance(tmp_path,monkeypatch):
 monkeypatch.setattr(c,'ROOT',tmp_path);case=next(x for x in c.catalog() if x['reused_original14_case']);meshes=[]
 for n,ny in [(32,16),(40,20)]:
  field=tmp_path/('mesh'+str(n)+'.npz');field.write_bytes(b'syntheticmetadataonly')
  meshes.append(dict(cells=[n,ny,ny],completed=True,physics_passed=True,all_independent_original_Torch_linear_checks_passed=True,capture=dict(path=field.name,sha256=c.immutable.sha(field))))
 result=tmp_path/'result.json';cache=tmp_path/'cached.json'
 def commit(data):
  result.write_text(json.dumps(data));cache.write_bytes(result.read_bytes());return dict(path=result.name,sha256=c.immutable.sha(result)),dict(path=cache.name,sha256=c.immutable.sha(cache))
 data=dict(case=case,completed=True,passed=True,physics_passed=True,meshes=meshes,conditional_spatial_indicators_m=[1e-6]);a,b=commit(data)
 assert len(c.check_completed_reuse(case,a,b)['fields'])==2
 data['meshes'][0].pop('all_independent_original_Torch_linear_checks_passed');a,b=commit(data)
 with pytest.raises(ValueError,match='certifiedoldTorch'):c.check_completed_reuse(case,a,b)
 assert c.check_completed_reuse(case,a,b,[a])['original_Torch_backend_reuse']
 data['conditional_spatial_indicators_m']=[21e-6];a,b=commit(data)
 with pytest.raises(ValueError,match='20um'):c.check_completed_reuse(case,a,b,[a])
 cache.write_bytes(b'changed')
 with pytest.raises(ValueError,match='Changed hashed'):c.check_completed_reuse(case,a,b,[a])
