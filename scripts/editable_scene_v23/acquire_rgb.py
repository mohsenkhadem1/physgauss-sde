"""Resume pinned full RGB acquisition, with bounded workers and CRC validation."""
import concurrent.futures,json,time,zlib,hashlib,threading
from pathlib import Path
from remotezip import RemoteZip
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/editable-scene-v23'

def main():
    plan=json.loads((BASE/'rgb-members.json').read_text());groups={}
    for item in plan['members']:groups.setdefault(item['member'].split('/')[2],[]).append(item)
    dest=BASE/'phystwin-full-rgb';dest.mkdir(exist_ok=True);lock=threading.Lock();status={};deadline=time.monotonic()+7200
    def run(case,items):
        records=[]
        with RemoteZip(plan['archive'],timeout=30)as archive:
            for item in items:
                if time.monotonic()>deadline:raise TimeoutError('Two-hour acquisition bound; resume to continue')
                path=dest/item['member'];path.parent.mkdir(parents=True,exist_ok=True)
                for attempt in range(3):
                    try:
                        if path.exists():payload=path.read_bytes()
                        else:
                            cached=ROOT/'runs/real-transfer-v20/render-assets'/item['member']
                            payload=cached.read_bytes() if cached.exists() else archive.read(item['member'])
                        if len(payload)!=item['size'] or zlib.crc32(payload)!=item['crc32']:raise ValueError('Archive integrity mismatch')
                        if not path.exists():
                            temporary=path.with_suffix('.partial');temporary.write_bytes(payload);temporary.replace(path)
                        break
                    except Exception:
                        if attempt==2:raise
                        time.sleep(2**attempt)
                records.append(dict(member=item['member'],size=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
                if len(records)%25==0:
                    with lock:
                        status[case]=dict(files=len(records),total=len(items),complete=False);(BASE/'acquisition-status.json').write_text(json.dumps(status,indent=2)+'\n')
        (dest/f'{case}-manifest.json').write_text(json.dumps(dict(revision=plan['revision'],records=records),indent=2)+'\n')
        with lock:
            status[case]=dict(files=len(records),total=len(items),complete=True);(BASE/'acquisition-status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(case,len(records),'verified',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:
        jobs=[pool.submit(run,c,v)for c,v in groups.items()]
        for f in jobs:f.result()

if __name__=='__main__':main()
