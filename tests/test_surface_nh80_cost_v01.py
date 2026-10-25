import builtins,inspect,symtable
import numpy as np
import pytest
from experiments import run_surface_nh80_cost_v01 as m


def test_only_prospective_public_controls_accepted_no_parameter_redraw_or_grid():
    np.testing.assert_array_equal(m.validate_parameters(m.PARAMETERS),np.array([[2000,5],[4000,np.sqrt(150)],[8000,30]]))
    for bad in [np.array(m.PARAMETERS)[::-1],[[2000,5],[4000,12],[8000,30]],np.array(m.PARAMETERS)[:2]]:
        with pytest.raises(ValueError):m.validate_parameters(bad)


def test_field_parity_is_maximum_Euclidean_distance_not_coordinate_or_mean_error():
    zero=np.zeros((100,3));same=zero.copy();same[-1]=[1e-12,1e-12,0];assert m.field_parity(zero,same)['passed']
    same[-1]=[1.5e-12]*3;assert not m.field_parity(zero,same)['passed']
    with pytest.raises(ValueError):m.field_parity(zero,np.ones((2,3)))
    with pytest.raises(ValueError):m.field_parity(zero,np.full_like(zero,np.nan))


def test_all_worker_controller_global_references_resolve_before_PDE_calls():
    for function in [m.worker,m.run,m.prepare,m.validate,m.guard,m.cleanup]:
        table=symtable.symtable(inspect.getsource(function),'NH80-cost','exec')
        def check(t):
            for symbol in t.get_symbols():
                if symbol.is_referenced() and symbol.is_global():assert symbol.get_name() in vars(m) or hasattr(builtins,symbol.get_name()),symbol.get_name()
            for child in t.get_children():check(child)
        check(table)


def test_resource_guard_rejects_GPU_visibility_and_wrong_affinity_before_capacity_probe(monkeypatch):
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:monkeypatch.setenv(key,'1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setattr(m.os,'sched_getaffinity',lambda _: {17})
    with pytest.raises(ValueError,match='no-GPU'):m.guard('worker')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setattr(m.os,'sched_getaffinity',lambda _: {16,17})
    with pytest.raises(ValueError,match='CPU'):m.guard('worker')
