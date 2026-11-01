import copy
import numpy as np
import pytest
from scripts.physical_refinement_v15.belief import Query
from scripts.physical_refinement_v15.reserved_forecasts import validate_packet_manifest


def test_reject_unexpected_outcomes_and_cross_modality():
    queries=[Query(name,'rgbd',np.zeros((2,d)),np.ones(d)) for name,d in [('initial',64),('probe',70)]]
    meta=dict(world='reserved-00',repeat=0,track='VF',packets={'initial':dict(dimension=64),'probe':dict(dimension=70)})
    validate_packet_manifest(meta,'reserved-00',0,'VF',queries)
    for modified in (dict(meta,track='V'),dict(meta,world='reserved-01')):
        with pytest.raises(ValueError):validate_packet_manifest(modified,'reserved-00',0,'VF',queries)
    changed=copy.deepcopy(meta);changed['packets']['target_response']=dict(dimension=70)
    with pytest.raises(ValueError):validate_packet_manifest(changed,'reserved-00',0,'VF',queries)
    changed=copy.deepcopy(meta);changed['packets']['probe']['dimension']=64
    with pytest.raises(ValueError):validate_packet_manifest(changed,'reserved-00',0,'VF',queries)
