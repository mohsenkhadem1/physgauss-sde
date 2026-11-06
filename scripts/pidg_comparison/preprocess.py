"""Training-only TV-L1 flow between actual retained neighbors; released depth.
The adapter never computes flow through validation or final-test images.
"""
import argparse,concurrent.futures,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.pidg_comparison.prepare import get,BASE
ROOT=Path('runs/pidg-comparison-v01/data/balls_reaction')
def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',default='runs/pidg-comparison-v01/data');parser.add_argument('--scene',default='balls_reaction');args=parser.parse_args();ROOT=Path(args.data_root)/args.scene
    cv2.setNumThreads(1);raw=ROOT/'released';fit=ROOT/'fit';out=ROOT/'fit320';out.mkdir(exist_ok=True)
    docs={s:json.loads((fit/f'transforms_{s}.json').read_text()) for s in ['train','test']}
    for s,d in docs.items():(out/f'transforms_{s}.json').write_text(json.dumps(d,indent=2))
    for d in docs.values():
        for f in d['frames']:
            path=f['file_path'];im=Image.open(raw/(path+'.png')).convert('RGBA').resize((320,180),Image.Resampling.LANCZOS)
            dst=out/(path+'.png');dst.parent.mkdir(exist_ok=True);im.save(dst)
            mask=out/'resized_mask'/(Path(path).name+'.png');mask.parent.mkdir(exist_ok=True);im.getchannel('A').save(mask)
    train=docs['train']['frames'];jobs=[]
    for f in train:
        name=Path(f['file_path']).name+'.npy';jobs.append((f'{BASE}/{args.scene}/depth_distill/{name}',raw/'depth_distill'/name))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(lambda x:get(*x),jobs))
    depthout=out/'depth_distill';depthout.mkdir(exist_ok=True)
    for f in train:
        name=Path(f['file_path']).name+'.npy';d=np.load(raw/'depth_distill'/name);d=cv2.resize(d,(320,180),interpolation=cv2.INTER_LINEAR);np.save(depthout/name,d)
    flowout=out/'flows_flo';flowout.mkdir(exist_ok=True)
    def job(pair):
        a,b=pair;ia=cv2.imread(str(out/(a['file_path']+'.png')),0);ib=cv2.imread(str(out/(b['file_path']+'.png')),0)
        for src,dst,direction,name in [(ia,ib,'fwd',Path(a['file_path']).name),(ib,ia,'bwd',Path(b['file_path']).name)]:
            path=flowout/f'flow_{direction}_{name}.flo'
            if not path.exists():cv2.writeOpticalFlow(str(path),cv2.optflow.DualTVL1OpticalFlow_create().calc(src,dst,None))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(job,zip(train[:-1],train[1:])))
    manifest={'flow':'OpenCV Dual TV-L1, actual adjacent retained training frames only','depth':'Released per-image Distill-Any-Depth; no temporal fitting in this adapter','dimensions':[320,180],'train_count':len(train),'validation_count':len(docs['test']['frames']),'test_images_loaded':False,'files':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}}
    (out/'preprocessing.json').write_text(json.dumps(manifest,indent=2));print('ready',len(manifest['files']),flush=True)
if __name__=='__main__':main()
