"""Actual frozen v04 surface-marker case000 figures; no outcome selection.

Reuse rendering/layout helpers only. Inference is reconstructed from v04 sources
and observations; v01 inference globals are never modified. The full-node FEM
response cache is reused after exact parameter-grid and interpolation checks.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import torch
from . import surface,surface_campaign,visualize
from .mechanics import LinearGaussianBody

ROOT=surface.ROOT;BASE=ROOT/'runs/joint-field-surface-v04';REPORT=ROOT/'reports/joint-field-surface-v04';OUT=ROOT/'output/joint-field-surface-v04'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pdf',action='store_true');ap.add_argument('--skip-video',action='store_true');args=ap.parse_args();torch.set_num_threads(1);OUT.mkdir(exist_ok=True,parents=True)
    cohort={};sources=[];checks={}
    for shape in ['block','step']:
        bankpath=BASE/f'{shape}.npz';data=dict(np.load(bankpath));maps=surface.maps(data);truthpath=BASE/shape/'matched-truth.npz';truth=dict(np.load(truthpath));casepath=REPORT/shape/'matched/case-000.json';case=json.loads(casepath.read_text())
        post={'initial':surface_campaign.initial(maps,truth['y'][0])};chosen={}
        for policy in ['fixed','active']:
            actions=[r['action'] for r in case['rows'] if r['policy']==policy and r['step']>0]
            assert len(actions)==4;p=post['initial']
            for a in actions:p=surface_campaign.action_update(p,maps,a,truth['measurements'][0,a])
            post[policy]=p;chosen[policy]=[maps['actions'][a]['name'] for a in actions]
        oldbank=ROOT/f'runs/joint-field-identification-v01/{shape}.npz';old=dict(np.load(oldbank));assert np.array_equal(data['theta'],old['theta']), 'Full-node cache cannot be reused after parameter-grid change'
        cache=ROOT/f'runs/joint-field-identification-v01/{shape}/visual-nodal.npz';nodal=dict(np.load(cache));body=LinearGaussianBody(shape,tuple(data['cells']));N=body.interpolation(data['points']).numpy()
        unit_error=float(np.abs(np.einsum('pn,hncl->hpcl',N,nodal['unit'],optimize=True)-data['unit']).max())
        probe_error=float(np.abs(np.einsum('pn,hncl->hpcl',N,nodal['probe'],optimize=True)-data['probe']).max());assert max(unit_error,probe_error)<1e-12
        checks[shape]=dict(parameter_grid_identical=True,unit_point_interpolation_max_error_m_per_N=unit_error,probe_point_interpolation_max_error_m_per_N=probe_error)
        cohort[shape]=dict(data=data,truth=truth,post=post,chosen=chosen,nodal=nodal,pred=visualize.predictions(nodal,truth,post));sources.extend([bankpath,truthpath,casepath,cache])
    # Output-root substitutions only; helpers receive all measured data explicitly.
    # No mechanics, likelihood, posterior or benchmark globals are patched.
    visualize.OUT=OUT;visualize.BASE=BASE
    original_save=visualize.save
    def publication_save(fig,path,pdf=False):
        from matplotlib.text import Text
        minimum=14. if fig.get_figwidth()>12 else 12.5
        for ax in fig.axes:ax.tick_params(labelsize=minimum)
        for text in fig.findobj(Text):
            label=text.get_text()
            if label.startswith(('Reference is a known','Intervals are marginal','Case 000 ·')):
                text.set_visible(False);continue
            label=label.replace('Volume-weighted error:','RMS:').replace('Fixed measurements','After fixed').replace('Adaptive measurements','After adaptive')
            text.set_text(label)
            if label in ['Block','Step']:text.set_position((.5,.05))
            if label:text.set_fontsize(max(text.get_fontsize(),minimum))
        original_save(fig,path,pdf)
    visualize.save=publication_save
    metrics=visualize.response_figure(cohort,'',args.pdf);visualize.material_figure(cohort,'',args.pdf);visualize.force_figure(cohort,'',args.pdf)
    if not args.skip_video:
        for shape,c in cohort.items():visualize.video(shape,c,'')
    manifest=dict(version='joint-field-surface-v04',selection='Fixed case000 of each shape; held calibrated probe8 atx+.02m, force-basis time4, no outcome-based selection.',actual_geometry_scale=True,known_surface_marker_correspondences=True,visibility_audit='reports/joint-field-surface-v04/visibility-summary.json',changed_load_surface_error_rms_mm=metrics,action_choices={s:c['chosen'] for s,c in cohort.items()},full_node_cache_checks=checks,sources={str(p.relative_to(ROOT)):sha(p) for p in sources},source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),'scripts/joint_field_identification/visualize.py':sha(Path(__file__).with_name('visualize.py'))},force_profile_definition='Normalized longitudinal 1D Gaussian load-basis proxy (N/m), not exact integrated surface traction.',scope='Actual FEM posterior predictions from surface-visible v04 observations. Images show geometry/material/error, not photorealistic reconstruction. Replay uses8quasistatic states with camera orbit, not a dynamic forecast. Fixed and active outcomes both retained.')
    (OUT/'figure-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
