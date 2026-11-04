"""Pinned public Deform360 cohort; raw recordings stay local."""
import concurrent.futures, hashlib, json, time, urllib.request
from pathlib import Path
from scripts.real_transfer_v19.phystwin_run import BASE, ROOT, write

OBJECTS=['043-dog','046-sponge','047-rectangle-sponge','048-butter-sponge','049-ball','051-cube']
REPO='brownu/deform360'

def read(url):
    with urllib.request.urlopen(url,timeout=90) as r:return json.load(r)

def listing(revision,path):
    url=f'https://huggingface.co/api/datasets/{REPO}/tree/{revision}/{path}?recursive=true&limit=1000'
    result=[]
    while url:
        with urllib.request.urlopen(url,timeout=90) as r:
            result.extend(json.load(r));links=r.headers.get('Link','')
        url=None
        for link in links.split(','):
            if 'rel="next"' in link:url=link.split('<')[1].split('>')[0]
    return result

def acquire():
    report=ROOT/'reports/real-transfer-v19';planpath=report/'deform360-download-plan.json'
    if planpath.exists():plan=json.loads(planpath.read_text())
    else:
        revision=read(f'https://huggingface.co/api/datasets/{REPO}')['sha'];files=[]
        for obj in OBJECTS:
            files.extend(x for x in listing(revision,'raw/'+obj) if x['type']=='file' and not x['path'].endswith(('.wav','.flac')))
        plan=dict(repository=REPO,revision=revision,objects=OBJECTS,files=files,
                  bytes=sum(x['size'] for x in files),selection='Six named volumetric objects selected before downloading/scoring; all released episodes and cameras; no outcome selection.')
        write(planpath,plan)
    root=BASE/'Deform360';done=[];errors=[];start=time.time()
    def download(item):
        path=root/item['path'];path.parent.mkdir(parents=True,exist_ok=True)
        expected=item.get('lfs',{}).get('oid')
        if path.exists() and path.stat().st_size==item['size']:
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if expected is None or digest==expected:return dict(path=item['path'],sha256=digest,bytes=item['size'])
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{plan["revision"]}/{item["path"]}'
        for attempt in range(3):
            try:
                tmp=path.with_suffix(path.suffix+'.partial');h=hashlib.sha256();size=0
                with urllib.request.urlopen(url,timeout=90) as source,tmp.open('wb') as target:
                    while True:
                        chunk=source.read(1024*1024)
                        if not chunk:break
                        h.update(chunk);target.write(chunk);size+=len(chunk)
                if size!=item['size'] or (expected and h.hexdigest()!=expected):raise ValueError('Download integrity mismatch')
                tmp.replace(path);return dict(path=item['path'],sha256=h.hexdigest(),bytes=size)
            except Exception:
                if attempt==2:raise
    print('Download plan',len(plan['files']),plan['bytes'],'bytes',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(download,item):item for item in plan['files']}
        for future in concurrent.futures.as_completed(futures):
            try:done.append(future.result())
            except Exception as error:errors.append(dict(path=futures[future]['path'],error=str(error)))
            if (len(done)+len(errors))%50==0:
                status=dict(completed=len(done),total=len(plan['files']),errors=errors,elapsed_s=time.time()-start,bytes=sum(x['bytes'] for x in done))
                write(root/'download-status.json',status);print(status['completed'],status['total'],flush=True)
    write(root/'download-manifest.json',dict(repository=REPO,revision=plan['revision'],records=done,errors=errors))
    write(root/'download-status.json',dict(completed=len(done),total=len(plan['files']),errors=errors,finished=True,elapsed_s=time.time()-start,bytes=sum(x['bytes'] for x in done)))
    return not errors

if __name__=='__main__':raise SystemExit(0 if acquire() else 1)
