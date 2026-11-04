"""Acquire matching processed tracks/calibration, preserving label provenance."""
import json,zlib,hashlib
from pathlib import Path
from remotezip import RemoteZip
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/editable-scene-v23'

def main():
    inventory=json.loads((ROOT/'reports/editable-scene-v23/data-inventory.json').read_text());dest=BASE/'phystwin-full-rgb';records=[];missing=[]
    with RemoteZip(inventory['archive'],timeout=30)as z:
        names=set(z.namelist())
        for case in inventory['cases']:
            for leaf in ('final_data.pkl','calibrate.pkl','metadata.json','split.json','gt_track_3d.pkl'):
                name=f'data/different_types/{case}/{leaf}'
                if name not in names:missing.append(name);continue
                path=dest/name;path.parent.mkdir(parents=True,exist_ok=True);info=z.getinfo(name)
                if path.exists():payload=path.read_bytes()
                else:
                    cached=ROOT/'runs/third_party/PhysTwin'/name
                    payload=cached.read_bytes()if cached.exists()else z.read(name)
                if len(payload)!=info.file_size or zlib.crc32(payload)!=info.CRC:raise ValueError('Archive mismatch '+name)
                if not path.exists():
                    temporary=path.with_suffix(path.suffix+'.partial');temporary.write_bytes(payload);temporary.replace(path)
                records.append(dict(member=name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
            print(case,'labels verified',flush=True)
    report=dict(revision=inventory['revision'],complete=True,records=records,missing=missing,scope='Author processed visual tracks and calibration; tracks are estimated observations, not material/force truth. No train/test allocation yet.')
    (BASE/'label-manifest.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
