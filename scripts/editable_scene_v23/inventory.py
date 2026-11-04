"""Inventory pinned PhysTwin release before allocating object-disjoint splits."""
from pathlib import Path
from collections import defaultdict
import json
from remotezip import RemoteZip
ROOT=Path(__file__).resolve().parents[2]

def main():
    revision=json.loads((ROOT/'runs/real-transfer-v20/render-assets/snapshot.json').read_text())['revision']
    url=f'https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/{revision}/data.zip'
    cases=defaultdict(lambda:dict(rgb_files=0,rgb_bytes=0,depth_files=0,processed=False))
    selected=[]
    with RemoteZip(url)as archive:
        for item in archive.infolist():
            p=item.filename.split('/')
            if len(p)<4 or p[:2]!=['data','different_types']:continue
            row=cases[p[2]]
            if '/color/' in item.filename and item.filename.endswith('.png'):
                row['rgb_files']+=1;row['rgb_bytes']+=item.file_size
                selected.append(dict(member=item.filename,size=item.file_size,crc32=item.CRC))
            if '/depth/' in item.filename:row['depth_files']+=1
            if p[-1]=='final_data.pkl':row['processed']=True
    old=set(p.name for p in (ROOT/'runs/third_party/PhysTwin/data/different_types').iterdir()if p.is_dir())
    for case,row in cases.items():row['historically_exposed']=case in old
    report=dict(revision=revision,archive=url,cases=dict(cases),rgb_total_bytes=sum(r['rgb_bytes']for r in cases.values()),
        scope='Inventory only. Existing eight cases are development; object identity grouping required before sealed train/validation/test split. No allocation by outcome.',
        paused_datasets=['Deform360'],access_pending=['PokeFlex'])
    (ROOT/'reports/editable-scene-v23/data-inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'runs/editable-scene-v23/rgb-members.json').write_text(json.dumps(dict(revision=revision,archive=url,members=selected),indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
