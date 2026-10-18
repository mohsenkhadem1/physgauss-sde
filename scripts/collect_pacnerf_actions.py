"""Collect completed owned-repository benchmark artifacts during an Actions run.

Read-only monitoring. Existing Git credentials remain in memory and are sent
only to api.github.com; artifact storage downloads use unsigned redirects.
Does not start, cancel, modify, or rerun any workflow.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request
from download_study_artifact import download,NoRedirect

REPOSITORY='mohsenkhadem1/physgauss-sde'


def collect(run_id, output, chains=False, once=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'workflow_status.json').exists():
        previous_run=json.loads((output/'workflow_status.json').read_text())['run_id']
        if previous_run!=run_id:raise ValueError('Output belongs to another run; choose a fresh folder')
    credential=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
        text=True,capture_output=True,check=True,
        env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never'))
    fields=dict(x.split('=',1) for x in credential.stdout.splitlines() if '=' in x)
    token=fields.get('password')
    if not token:raise ValueError('Existing GitHub authentication unavailable')
    def api(path):
        request=urllib.request.Request('https://api.github.com/repos/'+REPOSITORY+'/'+path,
            headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                     'User-Agent':'physgauss-benchmark-collector'})
        # These JSON endpoints return no cross-host redirects.
        with urllib.request.build_opener(NoRedirect()).open(request,timeout=40) as response:return json.load(response)
    previous=None
    while True:
        run=api(f'actions/runs/{run_id}')
        jobs=api(f'actions/runs/{run_id}/jobs?per_page=100')['jobs']
        state={'run_id':run_id,'source_commit':run['head_sha'],'status':run['status'],
            'conclusion':run['conclusion'],'checked_utc':datetime.now(timezone.utc).isoformat(),
            'jobs':[{'name':j['name'],'id':j['id'],'status':j['status'],'conclusion':j['conclusion'],
                'started_at':j['started_at'],'completed_at':j['completed_at'],
                'compute_steps':[{key:s.get(key) for key in ['name','status','conclusion','started_at','completed_at']}
                    for s in j.get('steps',[]) if s['name'] in [
                        'Fit and sample the original image target','Continue saved chains and RNG without new warmup',
                        'Fit and sample the exact-integral image target','Continue the exact target and saved RNG without new warmup']]}
                for j in jobs]}
        (output/'workflow_status.json').write_text(json.dumps(state,indent=2)+'\n')
        signature=[(j['name'],j['status'],j['conclusion']) for j in jobs]
        if signature!=previous:
            print(json.dumps({'status':run['status'],'completed':[(j['name'],j['conclusion']) for j in jobs if j['status']=='completed'],
                'running':sum(j['status']=='in_progress' for j in jobs)}),flush=True)
            previous=signature
        for job in jobs:
            match=re.fullmatch(r'compare \((\d+)\)',job['name'])
            if not match or job['status']!='completed':continue
            index=int(match[1])
            for kind in (['results','chains'] if chains else ['results']):
                folder=output/f'{kind}-{index}'
                if (folder/'artifact_download.json').exists():continue
                # Failed gates may still provide useful settings/progress.
                artifacts=api(f'actions/runs/{run_id}/artifacts?per_page=100')['artifacts']
                name=f'pacnerf-{kind}-{index}'
                if any(a['name']==name and not a['expired'] for a in artifacts):
                    download(run_id,name,folder)
        if run['status']=='completed':
            artifacts=api(f'actions/runs/{run_id}/artifacts?per_page=100')['artifacts']
            if any(a['name']=='complete-pacnerf-comparison' and not a['expired'] for a in artifacts):
                download(run_id,'complete-pacnerf-comparison',output/'ci-publication')
            return state
        if once:return state
        time.sleep(45)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=int)
    p.add_argument('--output',required=True);p.add_argument('--chains',action='store_true');p.add_argument('--once',action='store_true')
    a=p.parse_args();collect(a.run,a.output,a.chains,a.once)
