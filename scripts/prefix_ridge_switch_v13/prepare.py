"""Prepare read-only parent input hashes and candidate freeze; never authorize execution."""
from pathlib import Path
import sys,datetime
from scripts.prefix_ridge_switch_v13.run import ROOT,BASE,REPORT,PARENT,read,write,sha

def prepare():
    p=read(REPORT/'protocol-pending.json');paths=set()
    for c in p['cases']:
        name,cohort,s=c['name'],c['cohort'],c['spec'];old=PARENT/'model-selection-evaluation'/cohort/name
        paths.update(old/n for n in ('prediction-metadata.json','predictions.npz','metrics.json'))
        paths.add(PARENT/'base/validation'/name/'prediction.npz')
        paths.add(PARENT/'model-selection-base/validation'/name/'prediction.npz')
        paths.add(PARENT/'model-selection-base/full'/name/'prediction.npz')
        paths.add(PARENT/'feasibility-evaluation'/cohort/name/'predictions.npz')
        if cohort=='regression':
            tag='primary' if 'case' in s else 'physical-release';paths.add(ROOT/'runs/external-benchmark-v11/fits'/tag/name/'prediction.npz')
        else:paths.add(PARENT/'base/full'/name/'prediction.npz')
        if 'case' in s:
            folder=ROOT/'runs/third_party/PhysTwin/data/different_types'/s['case'];paths.update(folder/n for n in ('final_data.pkl','split.json','gt_track_3d.pkl'))
        else:
            family='object' if s['subset']=='object-soft' else 'spatial';folder=ROOT/'runs/external-benchmark-v11/raw/SoftVTBench'/s['subset']/f'libero_{family}'/f'libero_{family}_task{s["task"]}'/'replayed_demos';paths.add(next(folder.glob('*.hdf5')))
    # Pin fit-boundary/completion metadata accompanying every cached anchor.
    for path in list(paths):
        if path.name=='prediction.npz':
            paths.update(path.parent/name for name in ('complete.json','fit.json') if (path.parent/name).exists())
    paths.update((PARENT/'scene-assets').glob('*.usd'))
    paths.update(ROOT/f'runs/third_party/SoftVTBench/configs/{f}_soft/libero_{f}.json' for f in ('object','spatial'))
    paths.update(ROOT/name for name in ('reports/contact-benchmark-v12/model-selection-execution-plan.json','reports/contact-benchmark-v12/model-selection-freeze.json','reports/contact-benchmark-v12/model-selection-protocol.json'))
    sources=set()
    for folder in ('scripts/prefix_ridge_switch_v13','scripts/contact_benchmark_v12','scripts/external_benchmark'):
        sources.update((ROOT/folder).glob('*.py'))
    sources.update((ROOT/'tests').glob('test_prefix_ridge*.py'))
    sources.update(ROOT/name for name in read(ROOT/'reports/contact-benchmark-v12/model-selection-freeze.json')['source_sha256'])
    write(REPORT/'input-manifest.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_pending_sha256':sha(REPORT/'protocol-pending.json'),'input_sha256':{str(f.relative_to(ROOT)):sha(f) for f in sorted(paths)},'scope':'Byte hashes only; no evaluator arrays/metrics parsed to design selector.'})
    write(REPORT/'source-candidate.json',{'source_sha256':{str(f.relative_to(ROOT)):sha(f) for f in sorted(sources)},'input_manifest_sha256':sha(REPORT/'input-manifest.json'),'protocol_pending_sha256':sha(REPORT/'protocol-pending.json'),'not_execution_authorization':True})
if __name__=='__main__':prepare()
