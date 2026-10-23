import numpy as np
import torch
import pytest
from experiments.run_campaign_c39_side_table import catalog,phase_cases,build_model
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.campaign_p2_torch import TorchP2


def test_typed_catalog_and_disjoint_fixed_material_uses():
 from experiments.campaign_c38_nonlinear_table_catalog import catalog as original_catalog
 rows=catalog();original={(r['material_id'],r['support']):r['young_Pa'] for r in original_catalog()}
 assert all(r['young_Pa']==original[(r['material_id'],r['support'])] for r in rows)
 assert len(rows)==10 and len({r['id'] for r in rows})==10
 assert [len(phase_cases(i)) for i in range(3)]==[4,3,3]
 assert sum(r['material_use']=='TABLE_NODE' for r in rows)==6
 assert sum(r['material_use']=='WITHHELD_INTERPOLATION_VALIDATION' for r in rows)==4
 assert all(r['action_id']=='middle_side_2N' and r['Fz_N']==0 and r['force_vector_N']==[0.,2.,0.] for r in rows)
 with pytest.raises(ValueError):build_model((4,2,2),{**rows[0],'action_id':'gravity'},rule='keast24')


def test_new_load_inputs_exact_across_quadrature_material_support_and_finite_u():
 a=build_model((4,2,2),catalog()[0],rule='keast24');b=build_model((4,2,2),catalog()[-1],rule='exact80')
 for field in ['rest','tets','elements','faces','load']:np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
 np.testing.assert_allclose(a.public_force_components['surface_N'].sum(0),[0.,2.,0.],rtol=0,atol=1e-12)
 np.testing.assert_allclose(a.force_metadata['rest_surface_moment_Nm'],[-.08,0.,0.],rtol=0,atol=1e-12)
 # Manufactured compatible deformation; no nonlinear equilibrium or label used.
 u=2e-5*np.random.default_rng(391).normal(size=a.rest.shape);u[a.mask]=0
 E,g,_,J=StableP2Block.elastic(a,a.rest+u,False);op=TorchP2(a,'cpu');c=op.evaluate(torch.tensor(u,dtype=torch.float64))
 np.testing.assert_allclose(float(c['energy']),E,rtol=2e-12,atol=1e-14)
 np.testing.assert_allclose(c['gradient'].numpy()-a.load,g-a.load,rtol=2e-11,atol=1e-11)
 assert min(J)>.1


def test_worker_all_referenced_global_names_resolve_before_GPU_startup():
 import builtins,inspect,symtable
 from experiments import run_campaign_c39_side_table as m
 table=symtable.symtable(inspect.getsource(m.worker),'worker-startup','exec')
 def check(t):
  for symbol in t.get_symbols():
   if symbol.is_referenced() and symbol.is_global():
    assert symbol.get_name() in vars(m) or hasattr(builtins,symbol.get_name()),symbol.get_name()
  for child in t.get_children():check(child)
 check(table)
 assert callable(m.validate_cpu)
