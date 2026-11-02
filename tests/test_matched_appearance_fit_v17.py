import json,pickle
from pathlib import Path
import pytest
from scripts.compositional_dynamics_v17 import matched_appearance_fit as m

def test_exact_training_catalog_rejects_added_held_image(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'BASE',tmp_path/'raw');monkeypatch.setattr(m,'REPORT',tmp_path/'report');monkeypatch.setattr(m,'PLAN',tmp_path/'report/plan.json');monkeypatch.setattr(m,'DESIGN',tmp_path/'design.json')
    m.REPORT.mkdir();m.DESIGN.write_text('{}');proj=m.REPORT/'appearance-camera-projection-smoke.json';proj.write_text('{}')
    folder=m.BASE/'train-input';folder.mkdir(parents=True)
    for i in range(3):
        (folder/f'{i}.png').write_bytes(b'train');(folder/f'mask_{i}.png').write_bytes(b'mask')
    with (folder/'camera_meta.pkl').open('wb')as f:pickle.dump(dict(c2ws=[0]*3,intrinsics=[0]*3),f)
    m.write(m.PLAN,dict(design_sha256=m.sha(m.DESIGN),projection_sha256=m.sha(proj),source_sha256={}))
    m.write(m.BASE/'input-manifest.json',dict(plan_sha256=m.sha(m.PLAN),files={}))
    m.guard();(folder/'held.png').write_bytes(b'forbidden')
    with pytest.raises(ValueError,match='Exactly three'):m.guard()
