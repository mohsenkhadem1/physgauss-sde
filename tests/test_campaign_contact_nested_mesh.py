import copy
import numpy as np
import pytest
from experiments import campaign_contact_nested_mesh as m


def test_actual_factor_two_partitions_and_P2_polynomials_and_signed_mass():
 coarse,fine=m.geometry((1,1,1)),m.geometry((2,2,2));P,proof=m.nested_transfer(coarse,fine)
 assert proof['exact_dyadic_material_partition']and proof['coarse_face_area_partition_exact']
 assert m.polynomial_checks(coarse,fine,P)['passed']
 assert m.mass_check(coarse,fine,P)['passed']
 assert (P.data<0).any()


def test_wrong_actual_connectivity_rejected_despite_identical_box():
 coarse,fine=m.geometry((1,1,1)),m.geometry((2,2,2));bad=copy.deepcopy(fine)
 bad.tets[0,0]=bad.tets[-1,-1]
 with pytest.raises(ValueError):m.nested_transfer(coarse,bad)


def test_contact_virtual_work_and_deliberately_wrong_measure_mapping_rejected():
 coarse,fine=m.geometry((1,1,1)),m.geometry((2,2,2));P,_=m.nested_transfer(coarse,fine)
 q=coarse.rest.copy();q[:,2]-=.051;q[:,2]+=.003*(q[:,0]/.24-.5)
 a=m.contact(q,coarse);b=m.contact(P@q,fine)
 assert m.compare(a,b,P)['passed']
 wrong=m.contact(P@q,fine,area_scale=2.)
 assert not m.compare(a,wrong,P)['passed']
 wrong=dict(b,gradient=b['gradient'].copy());wrong['gradient'][0,2]+=.001
 assert not m.compare(a,wrong,P)['passed']


def test_strict_adaptive_contact_agrees_and_open_state_is_exact_zero():
 coarse,fine=m.geometry((1,1,1)),m.geometry((2,2,2));P,_=m.nested_transfer(coarse,fine)
 q=coarse.rest.copy();q[:,2]-=.05001
 a=m.contact(q,coarse);reference=m.contact(q,coarse,adaptive=True)
 assert m.compare(a,reference,np.eye(len(coarse.rest)))['passed']
 open_state=m.contact(coarse.rest,coarse)
 assert open_state['energy']==0 and not open_state['gradient'].any()


def test_actual_96_to_768_partition_geometry_and_sparse_mass_without_fields():
 coarse,fine=m.geometry((4,2,2)),m.geometry((8,4,4));P,proof=m.nested_transfer(coarse,fine)
 assert (proof['coarse_tets'],proof['fine_tets'])==(96,768)
 assert m.polynomial_checks(coarse,fine,P)['passed']and m.mass_check(coarse,fine,P)['passed']


def test_failed_adaptive_status_is_not_waived_or_replaced_by_partial_sums():
 coarse=m.geometry((1,1,1));q=coarse.rest.copy();q[:,2]-=.051
 with pytest.raises(RuntimeError,match='Strict adaptive E/g reference failed'):
  m.contact(q,coarse,adaptive=True)
