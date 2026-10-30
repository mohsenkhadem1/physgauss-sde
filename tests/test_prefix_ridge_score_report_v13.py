import json
import pytest
from scripts.prefix_ridge_switch_v13 import score,report

def test_terminal_gate_rejects_incomplete_before_any_evaluator(tmp_path,monkeypatch):
 monkeypatch.setattr(score,'BASE',tmp_path);monkeypatch.setattr(score,'REPORT',tmp_path)
 cases=[dict(name=str(i),cohort='dev',spec={}) for i in range(148)]
 monkeypatch.setattr(score,'guard',lambda:({'cases':cases},100.))
 (tmp_path/'method-freeze.json').write_text('{}');(tmp_path/'protocol.json').write_text('{}')
 manifest=dict(protocol_sha256=score.sha(tmp_path/'protocol.json'),method_freeze_sha256=score.sha(tmp_path/'method-freeze.json'),cases=[])
 (tmp_path/'attempts.json').write_text(json.dumps(manifest))
 with pytest.raises(RuntimeError,match='All148'):score.terminal_gate()

def test_complete_pairs_exclude_failures_not_rows():
 def metric(x,failed=0):return {'summary':dict(coordinate_rmse_mm=x,total_frames=10,valid_frames=10-failed,failed_frames=failed)}
 rows=[dict(name='a',attempt_state='complete',switched=True,results={score.METHOD:metric(1),'model_selection_selected':metric(2)}),dict(name='b',attempt_state='failed',switched=False,results={score.METHOD:metric(3,2),'model_selection_selected':metric(4)})]
 s=report.summarize(rows,'SoftVTBench')
 assert s['expected']==2 and s['methods'][score.METHOD]['complete']==1
 assert s['methods'][score.METHOD]['complete_only_mean_mm']==1
 assert s['methods'][score.METHOD]['finite_case_mean_mm']==2
 assert s['paired_comparisons']['model_selection_selected']['matched']==1
 assert s['paired_comparisons']['model_selection_selected']['mean_gain_mm']==1
 assert not s['methods']['official_phystwin']['applicable']

def test_dataset_metrics_not_substituted():
 r=dict(name='a',attempt_state='complete',switched=False,results={score.METHOD:{'summary':dict(coordinate_rmse_mm=1,total_frames=10,valid_frames=10,failed_frames=0)}})
 assert report.summarize([r],'PhysTwin')['methods'][score.METHOD]['complete']==0
