"""Combine continued prefixes with unchanged original arms, then audit/publish."""
import argparse
import json
import os
from pathlib import Path
import shutil
import urllib.request
from download_study_artifact import github_token,NoRedirect,download


def readiness(config):
    token=github_token();opener=urllib.request.build_opener(NoRedirect());records=[]
    for case,run in config['continuations'].items():
        req=urllib.request.Request(f'https://api.github.com/repos/mohsenkhadem1/physgauss-sde/actions/runs/{run}',
            headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'physgauss-resumption-report'})
        with opener.open(req,timeout=40) as response:meta=json.load(response)
        if meta['path']!='.github/workflows/pacnerf-exact-resume.yml' or meta['event']!='workflow_dispatch':raise ValueError('Unexpected continuation workflow')
        records.append({'case_index':int(case),'run_id':run,'status':meta['status'],'conclusion':meta['conclusion'],'url':meta['html_url']})
    return {'ready':all(r['status']=='completed' for r in records),'runs':records}


def publish(config,output,name,push):
    import torch
    from collect_pacnerf_actions import collect
    from pacnerf_comparison import write
    from verify_pacnerf_continuation import verify
    from inspect_pacnerf_attempts import find
    from finalize_pacnerf_exact import finalize
    from publish_pacnerf_discrepancy import snapshot_push
    import hashlib
    torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    status=readiness(config)
    if not status['ready']:raise ValueError('Continuations still running; report not finalized')
    root=Path('runs/resumption-publication');root.mkdir(parents=True,exist_ok=True)
    before=root/'original';collect(config['original_run'],before,chains=True,once=True)
    inputs=root/'inputs';download(config['original_run'],'pacnerf-inputs',inputs)
    merged=root/'merged'
    if merged.exists():raise ValueError('Choose a new report attempt directory')
    merged.mkdir()
    original_status=json.loads((before/'workflow_status.json').read_text())
    jobs=[j for j in original_status['jobs'] if j['name'] not in [f'compare ({i})' for i in config['continuations']]]
    proofs=[]
    for index in range(16):
        source=before
        if str(index) in config['continuations']:
            after=root/f'continued-{index}';collect(config['continuations'][str(index)],after,chains=True,once=True)
            new_status=json.loads((after/'workflow_status.json').read_text())
            jobs.extend(j for j in new_status['jobs'] if j['name']==f'compare ({index})')
            from pacnerf_comparison import cases
            s,v,n,f=cases()[index];case=f'{s}__{v}views__{n}__{f}'
            old_path=find(before/f'chains-{index}',case,'chains.pt');new_path=find(after/f'chains-{index}',case,'chains.pt')
            if old_path is None or new_path is None:raise ValueError('Continuation lost its saved prefix')
            old=torch.load(old_path,weights_only=True);new=torch.load(new_path,weights_only=True)
            if old['white'].shape==new['white'].shape:
                # A job interrupted after final draws may need only diagnostics/scoring.
                for key in ['white','log_ratio','energy_errors','rng_state']:
                    if not torch.equal(old[key],new[key]):raise ValueError('Zero-extension prefix changed')
                if old_path.parent.joinpath('reference.pt').read_bytes()!=new_path.parent.joinpath('reference.pt').read_bytes():raise ValueError('Reference changed')
                if old['step_size']!=new['step_size'] or old['counts']!=new['counts']:raise ValueError('Kernel/counters changed')
                old_settings=json.loads(find(before/f'results-{index}',case,'settings.json').read_text())
                new_settings=json.loads(find(after/f'results-{index}',case,'settings.json').read_text())
                if old_settings!=new_settings:raise ValueError('Target settings changed')
                proofs.append({'case':case,'additional_draws_per_chain':0,'original_prefix_bit_identical':True,
                    'all_pass':True,'qualification':'Original final draw prefix preserved; continuation completes diagnostics/scoring only'})
            else:proofs.append(verify(before,after,inputs,index))
            source=after
        for kind in ['results','chains']:
            origin=source/f'{kind}-{index}'
            if origin.exists():shutil.copytree(origin,merged/f'{kind}-{index}')
    original_status.update({'jobs':jobs,'continuation_round':status})
    write(merged/'workflow_status.json',original_status)
    result=finalize(merged,inputs,output,push=False)
    write(Path(output)/'continuation-integrity.json',{'all_pass':all(p['all_pass'] for p in proofs),'cases':proofs,'runs':status['runs']})
    # Replace stale original-report URLs in this new snapshot only.
    doc=Path(output)/'RESULTS.md';document=doc.read_text(encoding='utf-8').replace('examples/pacnerf-exact/','examples/'+name+'/')
    document+='\nThis snapshot combines the original nine completed arms with seven explicitly resumed runs. See continuation-integrity.json for source runs, exact prefix and RNG checks. The original report and archive remain unchanged.\n'
    doc.write_text(document,encoding='utf-8',newline='\n')
    publication=json.loads((Path(output)/'publication.json').read_text())
    for relative in ['RESULTS.md','continuation-integrity.json']:
        publication['files_sha256_lf'][relative]=hashlib.sha256((Path(output)/relative).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    write(Path(output)/'publication.json',publication)
    if push:snapshot_push(output,name,document)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/pacnerf_resumptions_round1.json')
    p.add_argument('--check-only',action='store_true');p.add_argument('--output');p.add_argument('--name');p.add_argument('--push',action='store_true')
    a=p.parse_args();config=json.loads(Path(a.config).read_text())
    if a.check_only:
        result=readiness(config);print(json.dumps(result))
        if os.environ.get('GITHUB_OUTPUT'):
            with open(os.environ['GITHUB_OUTPUT'],'a',encoding='utf-8') as stream:stream.write('ready='+str(result['ready']).lower()+'\n')
    else:publish(config,a.output,a.name,a.push)
