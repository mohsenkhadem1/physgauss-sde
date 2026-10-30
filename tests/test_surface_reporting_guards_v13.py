import hashlib,json
from pathlib import Path
import pytest
from scripts.coupled_contact_v13 import build_surface_report as report

def evidence(tmp_path,monkeypatch):
    directory=tmp_path/'reports/coupled-surface-v13';directory.mkdir(parents=True);monkeypatch.setattr(report,'ROOT',tmp_path);monkeypatch.setattr(report,'REPORT',directory)
    def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return hashlib.sha256(path.read_bytes()).hexdigest()
    prior={}
    for i in range(405):
        path=tmp_path/f'old/{i}.json';prior[str(path.relative_to(tmp_path))]=save(path,{})
    save(directory/'prior-preservation.json',{'sha256':prior})
    files=['protocol.json','method-freeze.json','physics-development-qualification.json','summary.json','development-results.json','reserved-results.json','ablation-results.json','supplementary-protocol.json']
    hashes={f:save(directory/f,{}) for f in files}
    for name in ('independent-audit.json','ablation-audit.json'):save(directory/name,dict(passed=True,errors=[],pending=[],prior_preserved_files=405,audited_report_sha256=hashes))
    protocol=dict(video_cases=[['o','edge_drop']],stress_forecast_scenarios=['edge_drop']);base=tmp_path/'runs/coupled-surface-v13';video=directory/'o-edge_drop.mp4'
    row=dict(path=str(video.relative_to(tmp_path)),sha256=save(video,{}),frames=61,simulation_fps=100,playback_fps=25,forecast_cohort='stress',development_scope_qualified=False,reference_display_label='Development unqualified')
    for file,key in [(base/'reference/o/edge_drop/trajectory.npz','reference_sha256'),(base/'evaluation/o/edge_drop/predictions.npz','prediction_sha256'),(base/'reference/o/edge_drop/complete.json','primary_status_sha256')]:row[key]=save(file,{})
    save(directory/'video-provenance.json',[row]);return protocol,directory,save,row

def test_pdf_guard_requires_current_successful_audits(tmp_path,monkeypatch):
    p,d,save,row=evidence(tmp_path,monkeypatch);assert 'video-provenance.json' in report.validate_evidence(p)
    save(d/'summary.json',{'changed':True})
    with pytest.raises(RuntimeError,match='stale'):report.validate_evidence(p)

def test_pdf_guard_rejects_unlabeled_stress_video(tmp_path,monkeypatch):
    p,d,save,row=evidence(tmp_path,monkeypatch);row['reference_display_label']='Reference checks passed';save(d/'video-provenance.json',[row])
    with pytest.raises(RuntimeError,match='unqualified label'):report.validate_evidence(p)
