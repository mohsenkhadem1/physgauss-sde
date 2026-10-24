"""Source-only control guards and serialization; no physical observations."""
import os
import numpy as np
import pytest
from experiments import benchmark_surface_positive_cdf_v01 as control


def test_explicit_CPU17_noGPU_owned_command():
    command=control.command('example',guard_only=True)
    assert command[:5]==['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    assert command[5:8]==['taskset','-c','17']
    assert command[-3:]==['guard','--digest','example']


def test_array_archive_cost_excludes_nested_diagnostic_records(tmp_path):
    final=dict(unit_grid=np.ones((3,2)),quadrature_probability=np.array([.2,.3,.5]),
               grouped={'probability':np.ones(2)},log_evidence=12.)
    path=tmp_path/'test.npz';cost=control.save_arrays(final,path)
    assert cost['array_bytes']==72 and cost['compressed_bytes']==path.stat().st_size
    with np.load(path)as z:
        assert set(z.files)=={'unit_grid','quadrature_probability'}
        np.testing.assert_array_equal(z['unit_grid'],final['unit_grid'])
    with pytest.raises(ValueError):control.save_arrays({},tmp_path/'empty.npz')
