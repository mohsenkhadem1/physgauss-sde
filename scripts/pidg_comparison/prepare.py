"""Fetch released PIDG RGB data and freeze a frame-aligned development protocol.
Raw data remain local. Test RGB is never loaded by the training data directory.
"""
import argparse, concurrent.futures, hashlib, json, time, urllib.request
from pathlib import Path
REPO='HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting'
REV='46b2352'
BASE=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}'
def get(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): return
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url,timeout=120) as r: data=r.read()
            tmp=path.with_suffix(path.suffix+'.part');tmp.write_bytes(data);tmp.replace(path);return
        except Exception:
            if attempt==4: raise
            time.sleep(2**attempt)
def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',default='balls_reaction');p.add_argument('--root',default='runs/pidg-comparison-v01/data');a=p.parse_args()
    root=Path(a.root)/a.scene;raw=root/'released';raw.mkdir(parents=True,exist_ok=True)
    for name in ['transforms_all.json','transforms_train.json','transforms_test.json','points3d.ply']:
        get(f'{BASE}/{a.scene}/{name}',raw/name)
    original={k:json.loads((raw/f'transforms_{k}.json').read_text()) for k in ('all','train','test')}
    for document in original.values():
        for f in document['frames']: f['file_path']=f['file_path'].replace('\\','/'); f['file_path']=f['file_path'][:-4] if f['file_path'].lower().endswith('.png') else f['file_path']
    frames={f['file_path']:f for k in original for f in original[k]['frames']}
    jobs=[(f'{BASE}/{a.scene}/{name}.png',raw/(name+'.png')) for name in sorted(frames)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:list(ex.map(lambda x:get(*x),jobs))
    train=original['train']['frames'];test=original['test']['frames']
    trainids={f['file_path'] for f in train};testids={f['file_path'] for f in test};assert not trainids&testids
    ids=sorted(frames,key=lambda x:int(Path(x).name));n0=int(Path(ids[0]).name);n1=int(Path(ids[-1]).name)
    def aligned(f): return dict(f,time=(int(Path(f['file_path']).name)-n0)/(n1-n0))
    validation=[aligned(f) for i,f in enumerate(train) if i%10==5]
    fitting=[aligned(f) for i,f in enumerate(train) if i%10!=5]
    test=[aligned(f) for f in test]
    # This directory exposes only fitting and validation images; no final test images.
    for folder,groups in [('fit',{'train':fitting,'test':validation}),('evaluation',{'train':fitting,'test':test})]:
        out=root/folder;out.mkdir(exist_ok=True)
        for split,fs in groups.items():
            (out/f'transforms_{split}.json').write_text(json.dumps({'camera_angle_x':original['train']['camera_angle_x'],'frames':fs},indent=2))
            for f in fs:
                dest=out/(f['file_path']+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():dest.symlink_to((raw/(f['file_path']+'.png')).resolve())
        # The released point cloud has unknown reconstruction provenance. Do not expose it;
        # use the same seeded random initial cloud for both methods in the runner.
    manifest={'scene':a.scene,'dataset':REPO,'revision':REV,'status':'data_ready','train_frames':len(fitting),'validation_frames':len(validation),'test_frames':len(test),'train_ids':[f['file_path'] for f in fitting],'validation_ids':[f['file_path'] for f in validation],'test_ids':[f['file_path'] for f in test],'time_convention':'(numeric frame ID - minimum ID)/(maximum ID - minimum ID), shared across all splits','original_time_samples':{k:[{'file_path':f['file_path'],'time':f.get('time')} for f in original[k]['frames'][:3]] for k in original},'initialization':'shared seed random points; released points3d.ply excluded because observation provenance unknown','selection':'validation only; final test remains separate','raw_sha256':{str(p.relative_to(raw)):hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.rglob('*') if p.is_file()}}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:manifest[k] for k in ['scene','status','train_frames','validation_frames','test_frames']}),flush=True)
if __name__=='__main__':main()
