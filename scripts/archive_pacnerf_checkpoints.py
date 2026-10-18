"""Preserve verified owned-repository Actions archives in an experimental release.

Never replaces/deletes an asset. Existing assets must match GitHub's SHA256.
Credentials stay in memory and are sent only to the two official GitHub API
origins. Redirects are disabled on authenticated requests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from download_study_artifact import NoRedirect,github_token

REPO='mohsenkhadem1/physgauss-sde'
STUDIES={
    'exact':{
        'run':39100239747,'commit':'219665b0776d9a1b5d092176331e21f724499bca',
        'tag':'pacnerf-exact-2026-10-18',
        'name':'Corrected PAC-NeRF raw checkpoints (experimental)',
        'status':'Corrected exact-integral study. Inference may still be incomplete; asset integrity does not establish convergence or calibrated uncertainty.',
        'document':'PACNERF_EXACT_RESULTS.md'},
    'historical':{
        'run':39073611351,'commit':'b2d081a7a55ecf1027921f5d841c043e23713a9d',
        'tag':'pacnerf-historical-quadrature-2026-10-18',
        'name':'SUPERSEDED PAC-NeRF quadrature checkpoints (historical)',
        'status':'SUPERSEDED: a confirmed high-stiffness covariance/load integration defect invalidates this target as corrected-model evidence. These assets preserve the original experiment and failures for audit only.',
        'document':'PACNERF_RESULTS.md'}}


def archive(raw,inputs,plan_only=True,study='exact',initialize_only=False,existing_only=False):
    profile=STUDIES[study];commit=profile['commit'];tag=profile['tag']
    raw,inputs=Path(raw),Path(inputs)
    metadata=[inputs/'artifact_download.json']+sorted(raw.glob('*/artifact_download.json'))
    sources=[]
    for file in metadata:
        record=json.loads(file.read_text());name=record['artifact_name']
        if name!='pacnerf-inputs' and not name.startswith(('pacnerf-results-','pacnerf-chains-')):continue
        if record['repository']!=REPO or record['workflow_run_id']!=profile['run']:raise ValueError('Unexpected study provenance')
        path=file.parent.parent/f"{name}-{record['artifact_id']}.zip"
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=record['archive_sha256'] or path.stat().st_size!=record['archive_size_bytes']:
            raise ValueError('Local archive differs from verified Actions artifact')
        sources.append((path,digest,record))
    if not sources:raise ValueError('No verified archives to preserve')
    if plan_only:
        plan={'state':'local plan only; no release created or asset uploaded',
            'repository':REPO,'tag':tag,'source_commit':commit,'prerelease':True,
            'study':study,'numerical_status':profile['status'],
            'assets':[{'name':p.name,'sha256':digest,'size_bytes':p.stat().st_size,
                'original_artifact_id':origin['artifact_id']} for p,digest,origin in sources]}
        (raw/'checkpoint-archive-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        print(json.dumps({'plan_only':True,'assets':len(sources),
            'total_bytes':sum(p.stat().st_size for p,_,_ in sources)}),flush=True)
        return
    token=github_token()
    opener=urllib.request.build_opener(NoRedirect())
    def request(url,method='GET',data=None,content_type='application/json',size=None):
        parsed=urllib.parse.urlsplit(url)
        if parsed.scheme!='https' or parsed.hostname not in ['api.github.com','uploads.github.com']:
            raise ValueError('Unexpected authenticated API origin')
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
            'User-Agent':'physgauss-checkpoint-archive'}
        if data is not None:headers['Content-Type']=content_type
        if size is not None:headers['Content-Length']=str(size)
        req=urllib.request.Request(url,method=method,data=data,headers=headers)
        with opener.open(req,timeout=180) as response:return json.load(response)
    base=f'https://api.github.com/repos/{REPO}/releases'
    try:release=request(base+'/tags/'+tag)
    except urllib.error.HTTPError as error:
        if error.code!=404:raise
        if existing_only:raise ValueError('Initialize the approved source-pinned prerelease with owner credentials before CI uploads') from None
        body=("**"+profile['status']+"**\n\n"
            "This is not a stable software release. The tag pins this archive's numerical worker; "
            "the main branch contains portable replay tools and the latest reports. "
            "Failed convergence, unfinished prefixes and references are deliberately retained.\n\n"
            "Assets are the original SHA256-verified Actions ZIPs: frozen prepared inputs, "
            "compact case results, and full chains/references. Available cases are added "
            "incrementally. Input images and particles derive from the authors' official "
            "PAC-NeRF data: https://github.com/xuan-li/PAC-NeRF .\n\n"
            "Protocol and status: https://github.com/"+REPO+"/blob/main/docs/"+profile['document']+"\n"
            "Original workflow: https://github.com/"+REPO+"/actions/runs/"+str(profile['run']))
        release=request(base,'POST',json.dumps({'tag_name':tag,'target_commitish':commit,
            'name':profile['name'],
            'body':body,'draft':False,'prerelease':True,'make_latest':'false'}).encode())
    if release['target_commitish']!=commit or not release['prerelease'] or release['draft']:
        raise ValueError('Existing release does not match this experimental archive')
    if initialize_only:
        print(json.dumps({'release_url':release['html_url'],'initialized_only':True,'assets_uploaded':0}),flush=True)
        return
    assets=request(release['assets_url']+'?per_page=100')
    known={a['name']:a for a in assets};preserved=[]
    upload=release['upload_url'].split('{',1)[0]
    for path,digest,origin in sources:
        asset=known.get(path.name)
        if asset is None:
            with path.open('rb') as stream:
                asset=request(upload+'?'+urllib.parse.urlencode({'name':path.name}),
                    'POST',stream,'application/zip',path.stat().st_size)
        if asset.get('digest')!='sha256:'+digest or asset['size']!=path.stat().st_size or asset['state']!='uploaded':
            raise ValueError('GitHub release asset digest/state differs; no asset replaced')
        preserved.append({'name':path.name,'asset_id':asset['id'],'sha256':digest,
            'size_bytes':asset['size'],'download_url':asset['browser_download_url'],
            'workflow_run_id':origin['workflow_run_id'],'original_artifact_id':origin['artifact_id']})
        manifest={'repository':REPO,'tag':tag,'source_commit':commit,'release_url':release['html_url'],
            'prerelease':True,'study':study,'numerical_status':profile['status'],'assets':preserved,
            'qualification':'experimental archive; asset integrity does not imply posterior convergence or physical correctness'}
        temp=raw/'checkpoint-archive.tmp';temp.write_text(json.dumps(manifest,indent=2)+'\n')
        temp.replace(raw/'checkpoint-archive.json')
        print(json.dumps({'preserved':path.name,'sha256':digest}),flush=True)
    print(json.dumps({'release_url':release['html_url'],'assets':len(preserved)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--inputs',required=True)
    p.add_argument('--study',choices=sorted(STUDIES),default='exact')
    p.add_argument('--initialize-release-only',action='store_true',help='Owner bootstrap of the approved source-pinned prerelease; upload no assets')
    p.add_argument('--require-existing-release',action='store_true',help='CI must not create source tags/releases')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--plan-only',action='store_true',help='Default: verify local ZIPs and write a plan without authentication or network access')
    modes.add_argument('--publish-public-archive',action='store_true',help='Requires explicit human approval: publish prepared inputs and raw chains as public release assets')
    a=p.parse_args()
    if a.initialize_release_only and not a.publish_public_archive:p.error('Release initialization also requires --publish-public-archive')
    archive(a.raw,a.inputs,not a.publish_public_archive,a.study,a.initialize_release_only,a.require_existing_release)
