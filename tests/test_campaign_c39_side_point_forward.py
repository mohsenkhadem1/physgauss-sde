"""Manufactured typed forward fields only; no archived fields or observations."""
import copy
import numpy as np
import pytest
from experiments import campaign_c39_side_point_forward as m


def coefficients(b):
 a=np.zeros((3,63,3));factor=1 if b==m.CATEGORIES[0]else 1.3;a[1,:,1]=.001*factor;a[2,:,2]=.0002*factor;return a


def archive():
 rows=[(2*i+j,c,n)for i,c in enumerate(m.side_catalog())if c['material_use']=='TABLE_NODE'for j,n in enumerate(m.MESHES)];q=[]
 for index,c,n in rows:
  a=coefficients(c['support']);x=75000/c['young_Pa'];q.append(m.POINTS+a[0]+x*(a[1]+x*a[2]))
 return dict(source_index=np.array([r[0]for r in rows]),case_ids=np.array([r[1]['id']for r in rows]),material_use=np.array(['TABLE_NODE']*12),information_role=np.asarray('C39_SIDE_CANDIDATE_TABLE_NODE_NOT_ADMITTED'),cells=np.array([r[2]for r in rows]),support=np.array([r[1]['support']for r in rows]),young_Pa=np.array([r[1]['young_Pa']for r in rows]),action_id=np.asarray(m.SIDE),force_vector_N=np.array([0.,2.,0.]),material_probes_m=m.POINTS.copy(),probe_positions_m=np.array(q))


class Initial:
 categories=m.CATEGORIES;young_nodes=m.YOUNG.copy()
 def observation_coefficients(self,b,F,ids):
  a=np.zeros((3,len(ids),3));a[1]=F*.0002;return a.reshape(3,-1)


def test_nonaffine_SIDE_interpolation_and_exact_old_model_reuse():
 a=archive();side=m.SidePointForward(a);initial=Initial();composite=m.CompositePointForward(initial,side);assert composite.initial_candidate is initial
 E=110000.;ids=np.array([11,32,53]);x=75000/E
 np.testing.assert_allclose(composite.predict_displacement(m.CATEGORIES[1],m.SIDE,E,ids),coefficients(m.CATEGORIES[1])[:,ids][0]+x*(coefficients(m.CATEGORIES[1])[:,ids][1]+x*coefficients(m.CATEGORIES[1])[:,ids][2]),atol=1e-16)
 for action,F in [('gravity',0.),('middle_down_2N',-2.),('canonical_up_1N',1.)]:np.testing.assert_array_equal(composite.observation_coefficients(m.CATEGORIES[0],action,ids),initial.observation_coefficients(m.CATEGORIES[0],F,ids))
 before=composite.predict_displacement(m.CATEGORIES[0],m.SIDE,E,ids);a['probe_positions_m'][:]=0;np.testing.assert_array_equal(before,composite.predict_displacement(m.CATEGORIES[0],m.SIDE,E,ids))


def test_typed_direction_roles_withheld_promotion_and_point_scope_rejected():
 for key,value in [('action_id',np.asarray('middle_down_2N')),('force_vector_N',np.array([0.,0.,2.])),('information_role',np.asarray('PRIVATE_WITHHELD'))]:
  a=archive();a[key]=value
  with pytest.raises(ValueError):m.SidePointForward(a)
 a=archive();a['source_index'][0]=4
 with pytest.raises(ValueError,match='withheld'):m.SidePointForward(a)
 a=archive();a['material_probes_m'][0,0]+=1e-6
 with pytest.raises(ValueError):m.SidePointForward(a)
 c=m.CompositePointForward(Initial(),m.SidePointForward(archive()))
 for action in (0.,'left_down_2N','middle_up_2N'):
  with pytest.raises(ValueError):c.observation_coefficients(m.CATEGORIES[0],action,[11])
 with pytest.raises(ValueError):c.observation_coefficients(m.CATEGORIES[0],m.SIDE,[63])


def test_linearSIDE_is_displacement_and_Eref75k_exact_scaling():
 rows=[dict(support=b,cells=n,young_Pa=75000.,action_id=m.SIDE,force_vector_N=[0.,2.,0.],material_probes_m=m.POINTS.copy(),displacement63_m=np.full((63,3),.001*(1 if b==m.CATEGORIES[0]else 2)))for b in m.CATEGORIES for n in m.MESHES]
 c=m.CompositePointForward(Initial(),m.LinearSidePointForward(rows));np.testing.assert_allclose(c.predict_displacement(m.CATEGORIES[1],m.SIDE,150000.,[11]),.001,atol=1e-17)
 bad=copy.deepcopy(rows);bad[0]['young_Pa']=100000.
 with pytest.raises(ValueError):m.LinearSidePointForward(bad)
 with pytest.raises(ValueError):m.LinearSidePointForward(rows[:-1])


def test_side_catalog_matches_frozen_mechanics_physicalkeys_exactly():
 from experiments.run_campaign_c39_side_table import catalog
 a=catalog();b=m.side_catalog();assert len(a)==len(b)==10
 for x,y in zip(a,b):assert all(x[k]==v for k,v in y.items())
 assert b[4]['young_Pa']==float(np.sqrt(75000.*150000.))
