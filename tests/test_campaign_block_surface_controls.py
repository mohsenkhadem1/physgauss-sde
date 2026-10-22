import numpy as np
from experiments.campaign_static_p2_block import QuadraticBlock
from experiments.campaign_block_surface_controls import PATTERNS,SUPPORTS,surface_control,wrench,support_groups,actuator_clock

def test_consistent_surface_resultants_and_rest_moments_P1_P2():
 for cells in [(4,2,2),(8,4,4)]:
  m=QuadraticBlock(cells,gravity=0.)
  for basis in ['P1','P2']:
   X=m.rest if basis=='P2' else m.rest[:m.tets.max()+1]
   for name in PATTERNS:
    f,meta=surface_control(X,m.tets,m.faces,name,m.elements if basis=='P2' else None)
    F,M=wrench(X,f);np.testing.assert_allclose(F,meta['rest_resultant_N'],atol=1e-14);np.testing.assert_allclose(M,meta['rest_moment_Nm'],atol=1e-14)
    assert meta['no_material_or_BC_input']
    if name=='axial_pull':np.testing.assert_allclose(F,[1,0,0],atol=1e-14)
    if name=='bending_couple':np.testing.assert_allclose(M,[0,.02,0],atol=1e-14)
    if name=='torsion_couple':np.testing.assert_allclose(M,[.02,0,0],atol=1e-14)
    if name=='top_middle_down':
     np.testing.assert_allclose(F,[0,0,-.12],atol=1e-14);np.testing.assert_allclose(M,0,atol=1e-14)
     assert meta['reference_patch_area_m2']==.0096

def test_actual_moment_and_known_clock_are_not_latent_dependent():
 m=QuadraticBlock(gravity=0.);f,meta=surface_control(m.rest,m.tets,m.faces,'transverse_end',m.elements)
 q=m.rest@np.diag([1.1,1,1]);_,moment=wrench(q,f)
 assert abs(moment[1]-meta['rest_moment_Nm'][1])>1e-4
 expected={'left_face':25,'right_face':25,'both_ends':50,'bottom_face':45,'separated_bottom_patches':30}
 for name in SUPPORTS:
  mask,active=support_groups(m.rest,name);assert mask.shape==(2,len(m.rest)) and mask.sum()==expected[name]
  again,_=surface_control(m.rest,m.tets,m.faces,'torsion_couple',m.elements);np.testing.assert_array_equal(again,surface_control(m.rest,m.tets,m.faces,'torsion_couple',m.elements)[0])
 for name in ['ramp_hold','bipolar_pulse']:
  t,c=actuator_clock(name);assert len(t)==361 and len(c)==360;np.testing.assert_array_equal(t,np.arange(361)/120)
