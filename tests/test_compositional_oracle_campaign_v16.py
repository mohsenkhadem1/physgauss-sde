"""Synthetic bookkeeping checks; never construct reserved physical scenes."""
import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from scripts.compositional_neural_mechanics import oracle_campaign as campaign


def test_reference_nan_and_artifact_mutation_are_not_silent_success(tmp_path,monkeypatch):
    q=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    with pytest.raises(ValueError,match='Nonfinite'):
        campaign.validate_value(dict(displacement=q*np.nan,wrenches=np.zeros((2,6))),q)
    out=tmp_path/'out';out.mkdir();report=tmp_path/'reports';report.mkdir()
    monkeypatch.setattr(campaign,'ROOT',tmp_path);monkeypatch.setattr(campaign,'OUT',out);monkeypatch.setattr(campaign,'REPORT',report)
    records=[];bad_artifact=None
    for i in range(12):
        cohort='reserved_composition' if i<6 else 'reserved_geometry';name=f'synthetic-{i}'
        record=dict(role=cohort,scene=dict(name=name,geometry=dict(dimensions=[1.,1.,1.],taper=0.,arch=0.)),actions=[dict(name=f'command-{j}') for j in range(4)])
        records.append(record);folder=out/name;folder.mkdir();rows=[]
        for action in record['actions']:
            for method in ('reference',*campaign.KINDS):
                path=folder/(action['name']+'-'+method+'.npz')
                error=0 if method=='reference' else 1 if method=='shared_laws' else 2
                np.savez(path,material_query=q,displacement=np.full_like(q,error*.001),wrenches=np.full((2,6),error*.01))
                rows.append(dict(scene_id=name,cohort=cohort,action=action['name'],method=method,state='complete',artifact=str(path.relative_to(tmp_path)),sha256=campaign.sha(path),scores={'deliberately_wrong_metric':12345}))
                if i==0 and method=='custom_direct':bad_artifact=path
        (folder/'complete.json').write_text(json.dumps(dict(rows=rows)))
    plan=dict(material_query=q.tolist(),scope='Synthetic bookkeeping only',statistics='Object means')
    monkeypatch.setattr(campaign,'runtime_records',lambda:(plan,records));monkeypatch.setattr(campaign,'guard',lambda:plan)
    monkeypatch.setattr(campaign,'FREEZE',tmp_path/'freeze.json');campaign.FREEZE.write_text('{}')
    campaign.summarize(records)
    results=json.loads((report/'oracle-reserved-forward-results.json').read_text())
    assert results['failed_predictions']==0
    assert all(x['complete'] and x['scored_commands']==24 for x in results['summaries'])
    assert all(x['objects']==6 and x['wins']==6 for x in results['paired_object_comparisons'])
    assert all('deliberately_wrong_metric' not in r.get('scores',{}) for r in results['rows'] if r['method']!='reference')
    bad_artifact.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):campaign.summarize(records)


def test_inventory_includes_legacy_solver_and_module_aliases():
    names=campaign.inventory()
    assert 'experiments/campaign_static_p2_block.py' in names
    assert 'scripts/compositional_neural_mechanics/forward_evaluation.py' in names
