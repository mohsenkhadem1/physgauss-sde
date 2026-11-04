import json
import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02 import retrieval_contact_campaign as module


@pytest.mark.parametrize('failed',[False,True])
def test_identical_policy_inputs_and_failed_neighbor_retention(tmp_path,monkeypatch,failed):
    source=tmp_path/'source';source.mkdir();package=tmp_path/'package';package.mkdir();(package/'manifest.json').write_text('{}')
    for name in ['plan.json','inference-sealed.json','complete.json','prediction.npz']:(source/name).write_text('{}')
    nodes=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    data={'nodes':nodes,'observed_initial':nodes.copy(),'observation_matrix':np.eye(4)}
    worlds=np.broadcast_to(nodes,(8,5,4,3)).copy()
    if failed:worlds[7]=np.nan
    arrays={**data,'nearest_raw':worlds[0].copy(),'retrieval_worlds':worlds}
    plan={'input_package':str(package),'input_package_manifest_sha256':module.sha(package/'manifest.json'),
          'input_digest':'stable','retrieval':{},'prefix_frames':[0,2],'point_ids':[0,1,2]}
    monkeypatch.setattr(module,'load_sealed',lambda *a:(plan,{'train_end':3},{'fit_end':3,'frames':5},arrays))
    monkeypatch.setattr(module,'load_package',lambda *a:data)
    monkeypatch.setattr(module,'predictor_digest',lambda *a:'stable')
    calls=[]
    def policy(payload,samples,**kwargs):
        assert payload is data and kwargs=={'prefix_frames':[0,2],'point_ids':[0,1,2]}
        calls.append(len(samples));return samples+2,{'policy':'same'}
    monkeypatch.setattr(module,'predict',policy)
    output=tmp_path/'result';module.prepare(source,output)
    with np.load(output/'prediction.npz',allow_pickle=False)as f:
        assert np.allclose(f['nearest_contact'],nodes+2)
        assert np.isnan(f['top8_contact']).all()if failed else np.allclose(f['top8_contact'],nodes+2)
        assert not any(name.endswith('_samples')for name in f.files)
    assert calls==([1]if failed else[1,8])
    assert json.loads((output/'complete.json').read_text())['no_new_sample_filtering']
