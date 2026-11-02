import json
import numpy as np
import pytest
from scripts.compositional_neural_mechanics.evaluation_protocol import held_commands,construct_reserved_records,paired_object_bootstrap,geometry_reference_cells,required_sources
from scripts.compositional_neural_mechanics.scenes import TRAIN_GEOMETRIES,poses,split_cells
from pathlib import Path


def test_reserved_construction_requires_freeze_without_sampling(tmp_path):
    plan=tmp_path/'unfrozen.json';plan.write_text(json.dumps({'status':'draft'}))
    with pytest.raises(ValueError,match='freeze'):construct_reserved_records(plan,tmp_path)


def test_commands_not_training_commands_and_paired_object_reduction():
    assert all(cell in split_cells()['train'] for cell in geometry_reference_cells())
    sources=required_sources(Path(__file__).resolve().parents[1])
    assert 'experiments/campaign_static_p2_block.py' in sources
    assert 'scripts/phystwin/consistent_fixture.py' in sources
    for geometry in TRAIN_GEOMETRIES:
        held=held_commands(geometry)
        assert len(held)==4
        assert all(not np.array_equal(a,b) for a in held.values() for b in poses(geometry).values())
    result=paired_object_bootstrap([1,2,3],[2,3,4],resamples=20)
    assert result['mean_error_reduction']==1
    assert result['percentile95_interval']==[1,1]
    assert result['objects']==3 and result['wins']==3
