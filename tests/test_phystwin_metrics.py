"""Independent checks for the adapter's official metric conventions."""
import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('phystwin_runner', Path(__file__).resolve().parents[1]/'scripts/phystwin/run_physics.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_split_and_l1_chamfer():
    target = np.zeros((4,1,3))
    prediction = target.copy()
    prediction[1,0] = [1,2,2]
    prediction[2:,0] = [2,0,0]
    data = {'object_points': target, 'surface_points': np.empty((0,3)),
            'object_visibilities': np.ones((4,1),bool)}
    result = module.metrics(prediction, data, {'train':[0,2],'test':[2,4]},target)
    assert result['train']['manual_track_mean_m'] == 3
    assert result['train']['chamfer_l1_mean_m'] == 5
    assert result['test']['manual_track_mean_m'] == 2
    assert result['test']['track_frames'] == 2


def test_missing_manual_tracks_are_not_zero_error():
    points = np.zeros((4,1,3))
    manual = points.copy()
    manual[2] = np.nan
    pred = points.copy();pred[3] = 1
    data = {'object_points':points,'surface_points':np.empty((0,3)),
            'object_visibilities':np.ones((4,1),bool)}
    r = module.metrics(pred,data,{'train':[0,2],'test':[2,4]},manual)
    assert r['test']['track_frames'] == 1
    assert np.isclose(r['test']['manual_track_mean_m'],np.sqrt(3))


def test_rigid_diagnostic_removes_rotation_but_not_stretch():
    from scripts.phystwin.audit_results import rigid_residual
    points = np.array([[0.,0.,0.],[1,0,0],[0,1,0],[0,0,1]])
    rotation = np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    moved = points @ rotation + np.array([3.,2.,1.])
    assert rigid_residual(points,moved) < 1e-25
    assert rigid_residual(points,2*moved) > 1
