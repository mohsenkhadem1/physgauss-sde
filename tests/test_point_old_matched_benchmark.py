import json,tempfile
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from unittest.mock import patch
import numpy as np
import pytest
import torch
from experiments import run_point_old_matched_benchmark as b
from experiments.censored_range_observation import CensoredRanges


def test_matched_resampling_changes_only_declared_seed_without_mutating_legacy_library():
    @dataclass(frozen=True)
    class Library:
        predictive_particles:int=2048
        counts_per_mask:tuple=(256,1024)
        references:tuple=tuple(SimpleNamespace(seed=s) for s in [20261112,20261113,20261114])
        resampling_seed:int=20261115
    original=Library();matched=b.matched_libraries(original)
    assert matched.resampling_seed==2026120700 and original.resampling_seed==20261115
    assert matched.references is original.references


def test_missing_reserved_release_stops_before_capture_cache_or_array_access():
    with patch.object(b,'validate_protocol',return_value=({},None,None,{})), \
            patch.object(b,'authorize_reserved',side_effect=ValueError('No explicit reserved release')), \
            patch.object(b.legacy,'capture_pairs',side_effect=AssertionError('Capture opened')), \
            patch.object(b,'load_support_basis_cache',side_effect=AssertionError('Forward loaded')):
        with pytest.raises(ValueError,match='reserved release'):b.predict({}, {})


def test_prediction_object_uses_only_acquired_frames_noise_seeds_and_freezes_all5states():
    rest=np.array([[0.,0,0],[.1,0,0],[0,.1,0],[0,0,.1]])
    pairs={label:dict(trajectory={'path':'synthetic-'+label,'sha256':'unused'}) for label in 'AB'}
    empirical=dict(geometry_ids=np.zeros(2048,dtype=np.int64),physical_coordinates=np.tile([5000.,625.,.35,.6],(2048,1)),reference_indices=np.arange(2048),weights=np.full(2048,1/2048))
    calls=[]
    def actual(path,frames,X,faces,queries,sensor,seed,device):
        calls.append((frames,seed))
        if frames not in [[10,20],[24]]:raise AssertionError('Unacquired frame supplied')
        n=72 if frames==[10,20] else 36
        return CensoredRanges(np.ones(n,dtype=bool),np.zeros(n))
    class Packed:
        def predict(self,clock,ids,theta,device='cpu'):
            return {'positions_m':torch.tensor(np.broadcast_to(rest,(len(ids),61,4,3)).copy())}
    p=dict(device='cpu',source_sha256={},program_manifests={program:b.make_exciting_commands(program).public_manifest() for program in b.PROGRAM_IDS})
    context=(Packed(),dict(rest_points_m=rest,surface_faces=[[0,1,2]]),None,[],[{}]*4)
    posterior=dict(empirical_posterior=empirical,summaries=[],stability={})
    prediction=CensoredRanges(np.ones((2048,144),dtype=bool),np.zeros((2048,144)))
    with tempfile.TemporaryDirectory(prefix='old-matched-test-',dir=b.ROOT/'runs/operator-dev') as temp:
        destination=Path(temp)/'object16'
        with patch.object(b.legacy,'_actual_observation',side_effect=actual), \
                patch.object(b.legacy,'_load_positions_subset',side_effect=AssertionError('Future target read')), \
                patch.object(b,'condition_reference',return_value=posterior), \
                patch.object(b,'predict_ranges',return_value=prediction), \
                patch.object(b,'update_after_view',return_value=(np.full(2048,1/2048),{'effective_sample_size':2048})):
            record=b.predict_object(dict(index=16,object_id='point_prior_016'),pairs,context,p,{'path':'protocol','sha256':'h'},{'path':'release','sha256':'h'},destination)
        frozen=json.loads((destination/'states-frozen.json').read_text())
        assert not frozen['future_A_or_B_truth_opened'] and not frozen['material_or_support_truth_opened'] and not frozen['primary_result_values_opened']
        assert frozen['state_names']==list(b.STATE_NAMES) and record['sha256']==b.sha(destination/'states-frozen.json')
        with np.load(destination/'predictive-states.npz') as saved:
            assert all(saved[n+'__A__mean_displacement_m'].shape==(61,4,3) for n in b.STATE_NAMES)
    assert calls==[([10,20],2026120216),*[([24],2026120500+4*16+j) for j in range(4)]]


def test_compare_blocks_primary_result_access_until_all32_baseline_states_frozen():
    with patch.object(b,'verify_all_states',side_effect=ValueError('All32 baseline states not frozen')), \
            patch.object(b,'checked',side_effect=AssertionError('Primary result or future target access')):
        with pytest.raises(ValueError,match='All32'):b.compare({}, {}, {})


def test_incomplete_global_freeze_rejected_before_any_predictive_array_load():
    with tempfile.TemporaryDirectory(prefix='old-freeze-test-',dir=b.ROOT/'runs/operator-dev') as temp:
        path=Path(temp)/'incomplete.json'
        path.write_text(json.dumps(dict(schema='old-model-new32-all-states-frozen-v0.1',completed=True,protocol={},object_indices=list(range(16,47)))))
        rec=b.record(path)
        with patch.object(b,'validate_protocol',return_value=({'source_sha256':{}},None,None,None)), \
                patch.object(b.np,'load',side_effect=AssertionError('Array opened before complete freeze')):
            with pytest.raises(ValueError,match='All32 baseline states'):b.verify_all_states({},rec)
