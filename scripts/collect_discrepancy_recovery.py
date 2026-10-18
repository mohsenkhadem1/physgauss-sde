"""Collect original and one bounded recovery, preserving artifact identities."""
import argparse
import json
from pathlib import Path
import shutil
import urllib.request
from collect_pacnerf_actions import collect
from download_study_artifact import download,github_token,NoRedirect


def collect_latest(run,output):
    output=Path(output)
    if output.exists():raise ValueError('Choose a fresh combined collector directory')
    initial=output.parent/'base-collected';collect(run,initial,chains=True,once=True)
    request=urllib.request.Request(f'https://api.github.com/repos/mohsenkhadem1/physgauss-sde/actions/runs/{run}/artifacts?per_page=100',
        headers={'Authorization':'Bearer '+github_token(),'Accept':'application/vnd.github+json','User-Agent':'physgauss-discrepancy-collector'})
    with urllib.request.build_opener(NoRedirect()).open(request,timeout=40) as response:artifacts=json.load(response)['artifacts']
    names={a['name'] for a in artifacts if not a['expired']}
    output.mkdir(parents=True)
    sources=[]
    for index in range(16):
        recovered=all(f'pacnerf-recovered-{kind}-{index}' in names for kind in ['results','chains'])
        for kind in ['results','chains']:
            if recovered:
                folder=output/f'{kind}-{index}'
                download(run,f'pacnerf-recovered-{kind}-{index}',folder)
            elif (initial/f'{kind}-{index}').exists():
                shutil.copytree(initial/f'{kind}-{index}',output/f'{kind}-{index}')
        sources.append({'case_index':index,'recovery_artifacts_selected':recovered})
    shutil.copyfile(initial/'workflow_status.json',output/'workflow_status.json')
    (output/'recovery-selection.json').write_text(json.dumps(sources,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();collect_latest(a.run,a.output)
