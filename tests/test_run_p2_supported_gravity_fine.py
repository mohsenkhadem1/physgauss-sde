import numpy as np
import pytest
from experiments.campaign_c38_static_family import build_geometry
from experiments.run_p2_supported_gravity_fine import probes,probe_map,query_field,conditional_indicator


def test_fixed_physical_probe_map_reproduces_affine_positions_and_velocities():
 m=build_geometry((4,2,2),100000.,'both_ends',Fz=0.,rule='keast24');ids,w=probe_map(m);A=np.array([[1.1,.2,0],[0,.9,.1],[.1,0,1.]]);t=np.array([.01,-.02,.03]);q=m.rest@A.T+t
 result=query_field(np.stack([q,2*q]),ids,w);expected=probes()@A.T+t
 np.testing.assert_allclose(result[0],expected,atol=2e-15,rtol=0);np.testing.assert_allclose(result[1],2*expected,atol=2e-15,rtol=0)
 np.testing.assert_allclose(w.sum(1),1,atol=1e-15,rtol=0);assert result.shape==(2,63,3)


def test_conditional_indicator_uses_actual_ratio_not_fixed_4():
 assert abs(conditional_indicator(2e-6,.016,.0133333333333333)-1e-5)<1e-18
 for args in [(-1,.02,.01),(1,.01,.02),(1,.01,.01),(float('nan'),.02,.01)]:
  with pytest.raises(ValueError):conditional_indicator(*args)


def test_joint_refinement_requires_distinct_actual_partitions_and_hashed_evidence(tmp_path):
 from experiments.run_p2_supported_gravity_fine import compare,sha
 rows=[];times=np.linspace(0,.1,25)
 for n,h in [(12000,.016),(20736,.0133333333333333)]:
  for label in ['coarse','refined']:
   f=tmp_path/(str(n)+'-'+label+'.npz')
   np.savez(f,times_s=times,positions_m=np.zeros((25,3,3)),velocities_m_s=np.zeros((25,3,3)),accepted_times_s=times,material_probes_m=probes(),probe_positions_m=np.zeros((25,63,3)),probe_velocities_m_s=np.zeros((25,63,3)))
   rows.append(dict(id=str(n)+'tets-'+label,capture=dict(path=str(f),sha256=sha(f)),case=dict(expected_hmax_m=h)))
 temporal,spatial=compare({},rows)
 assert all(not r['passed'] and r['agreement_only_if_partitions_identical'] for r in temporal)
 assert all(r['passed'] for r in spatial)
 for row in rows:
  if row['id'].endswith('refined'):
   f=row['capture']['path']
   with np.load(f) as a:data={k:a[k] for k in a.files}
   data['accepted_times_s']=np.linspace(0,.1,49);np.savez(f,**data);row['capture']['sha256']=sha(f)
 assert all(r['passed'] for r in compare({},rows)[0])
 rows[0]['capture']['sha256']='0'*64
 with pytest.raises(ValueError,match='trajectory evidence'):compare({},rows)
