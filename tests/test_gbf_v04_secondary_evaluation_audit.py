import json
from pathlib import Path
import pytest
from scripts.gaussian_boundary_flow_v02 import secondary_evaluation as module


@pytest.fixture
def setup(tmp_path,monkeypatch):
    root=tmp_path/'cohort';root.mkdir();protocol=tmp_path/'protocol.json';protocol.write_text('{}');digest=module.sha(protocol)
    rows=[{'case':f'case{i}','status_sha256':'status','stages':{}}for i in range(12)]
    for row in rows:(root/row['case']).mkdir()
    expected={'protocol_sha256':digest,'all12_terminal_before_any_target_read':True,'cases':rows};(root/'evaluation-gate.json').write_text(json.dumps(expected))
    monkeypatch.setattr(module,'terminal_gate',lambda *a:rows)
    imports=[{'case':r['case'],'state':'failed','imported':False}for r in rows]
    source=tmp_path/'source';source.mkdir();destination=root/'case0/rigid_contact';destination.mkdir()
    for folder in (source,destination):(folder/'payload').write_text('immutable')
    row={'case':'case0','state':'complete','imported':True,'source':str(source),'destination':str(destination),
         'secondary_addendum_sha256':'addendum','files_sha256':module.file_manifest(source)}
    imports[0]=row;(root/'case0/rigid_contact-import.json').write_text(json.dumps(row))
    plan={'case':'case0','secondary_addendum_sha256':'addendum','validation_protocol_sha256':digest}
    monkeypatch.setattr(module,'load_sealed',lambda *a:(plan,{},{},{}))
    report=tmp_path/'imports.json';report.write_text(json.dumps({'cases':imports,'secondary':True,'target_access':False}))
    calls=[]
    monkeypatch.setattr(module,'launch',lambda *a:(calls.append(a[0]) or 0))
    monkeypatch.setattr(module,'verified_evaluation',lambda *a:True)
    return root,protocol,digest,report,plan,calls


def test_live_later_orphan_blocks_every_earlier_write_or_launch(setup):
    root,protocol,digest,report,plan,calls=setup
    path=root/'case11/physics-evaluation-status.json';path.write_text(json.dumps({'state':'running','coordinator':module.process_identity(),'child':module.process_identity(),'command':['live']}))
    before={str(p):p.read_bytes()for p in root.rglob('*.json')}
    with pytest.raises(RuntimeError,match='Live or uncertain'):module.evaluate(root,protocol,digest,report)
    assert not calls and before=={str(p):p.read_bytes()for p in root.rglob('*.json')}


@pytest.mark.parametrize('change',['source','destination','plan','gate','cohort'])
def test_all_provenance_guards_precede_any_target_process(setup,monkeypatch,change):
    root,protocol,digest,report,plan,calls=setup
    document=json.loads(report.read_text())
    if change=='source':Path(document['cases'][0]['source'],'payload').write_text('changed')
    elif change=='destination':(root/'case0/rigid_contact/payload').write_text('changed')
    elif change=='plan':plan['validation_protocol_sha256']='changed'
    elif change=='gate':(root/'evaluation-gate.json').unlink()
    else:document['cases'].pop();report.write_text(json.dumps(document))
    with pytest.raises((ValueError,PermissionError)):module.evaluate(root,protocol,digest,report)
    assert calls==[]
    assert not list(root.glob('*/rigid_contact-evaluation-status.json'))


def test_failed_imports_are_preserved_and_no_retry(setup):
    root,protocol,digest,report,plan,calls=setup
    module.evaluate(root,protocol,digest,report)
    assert len(calls)==1
    assert json.loads((root/'case11/rigid_contact-evaluation-status.json').read_text())['state']=='blocked'
    module.evaluate(root,protocol,digest,report)
    assert len(calls)==1


def test_successful_exit_without_score_is_failure(setup,monkeypatch):
    root,protocol,digest,report,plan,calls=setup
    monkeypatch.setattr(module,'verified_evaluation',lambda *a:False)
    module.evaluate(root,protocol,digest,report)
    status=json.loads((root/'case0/rigid_contact-evaluation-status.json').read_text())
    assert status['state']=='failed' and 'verified score'in status['error']
    module.evaluate(root,protocol,digest,report);assert len(calls)==1


@pytest.mark.parametrize('verified',[False,True])
def test_dead_evaluator_recovers_only_verified_score_without_retry(setup,monkeypatch,verified):
    root,protocol,digest,report,plan,calls=setup
    old=module.process_identity();old['start_ticks']='0'
    path=root/'case0/rigid_contact-evaluation-status.json'
    path.write_text(json.dumps({'state':'running','coordinator':old,'child':old,'command':['finished'],
                                'cohort_gate_sha256':module.sha(root/'evaluation-gate.json'),'import_report_sha256':module.sha(report)}))
    monkeypatch.setattr(module,'verified_evaluation',lambda *a:verified)
    module.evaluate(root,protocol,digest,report)
    status=json.loads(path.read_text())
    assert status['state']==('complete'if verified else'interrupted')
    assert status['recovered_verified_score']is verified and not calls
