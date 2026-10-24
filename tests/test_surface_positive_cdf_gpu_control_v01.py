import pytest
from experiments import benchmark_surface_positive_cdf_gpu_v01 as control


def test_only_three_explicit_UUIDs_with_distinct_cores_and_fixed_CUBLAS():
    assert len(set(control.UUIDS))==3
    for slot in range(3):
        c=control.command(slot,'example',guard_only=True)
        assert c[1]=='CUDA_VISIBLE_DEVICES='+control.UUIDS[slot]
        assert c[2]=='CUBLAS_WORKSPACE_CONFIG=:4096:8'
        assert c[6:9]==['taskset','-c',str(control.CORES[slot])]
        assert c[-5:]==['guard','--slot',str(slot),'--digest','example']
    with pytest.raises(ValueError):control.command(3,'example')
