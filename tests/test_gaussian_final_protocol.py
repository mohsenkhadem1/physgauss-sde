import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from gaussian_final_protocol import seal,load_protocol,require_model,collect_test_cases


def fixture(tmp_path):
    design=tmp_path/'design.json';design.write_text('{"posterior_samples":128}')
    root=tmp_path/'dataset';root.mkdir()
    (root/'plan.json').write_text(json.dumps(dict(dataset='synthetic',episodes=[dict(episode_id='z0',split='test',group_id='g0',recipe_sha256='recipe') ])))
    inv=tmp_path/'inv.pt';inv.write_bytes(b'frozen inverse')
    fwd=tmp_path/'fwd.pt';fwd.write_bytes(b'frozen forward')
    output=tmp_path/'seal.json'
    seal(design,[root],[dict(name='model',inverse=str(inv),forward=str(fwd))],output)
    return design,root,inv,fwd,output


def test_seal_requires_exact_checkpoint_and_design_identity(tmp_path):
    design,root,inv,fwd,output=fixture(tmp_path)
    protocol=load_protocol(output);require_model(protocol,inv,fwd)
    with pytest.raises(FileExistsError):seal(design,[root],[],output)
    inv.write_bytes(b'changed inverse')
    with pytest.raises(ValueError,match='Checkpoint pair'):require_model(protocol,inv,fwd)
    design.write_text('{}')
    with pytest.raises(ValueError,match='design changed'):load_protocol(output)


def test_seal_and_public_plan_tampering_fail_before_episode_access(tmp_path):
    design,root,inv,fwd,output=fixture(tmp_path)
    protocol=load_protocol(output)
    # No episodes exist: changed public plan must be rejected first.
    (root/'plan.json').write_text('{}')
    with pytest.raises(ValueError,match='public plan changed'):collect_test_cases([root],protocol)
    value=json.loads(output.read_text());value['models'][0]['name']='changed'
    output.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='seal changed'):load_protocol(output)
