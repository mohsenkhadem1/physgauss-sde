"""Frozen visual features plus separate annotation records, no target-derived crops."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,pickle,time
import numpy as np,torch,cv2
from torchvision.models import resnet18,ResNet18_Weights
from .protocol import ROOT,DATA,BASE,REPORT,sha,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);a=p.parse_args();torch.set_num_threads(2)
    plan=json.loads((REPORT/'protocol.json').read_text());cases=sum(plan['cases'].values(),[])[a.shard::3]
    model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda().eval();enc=torch.nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3)
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    for case in cases:
        out=BASE/'prepared'/case;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        directory=DATA/case;raw=pickle.load(open(directory/'final_data.pkl','rb'));metadata=json.load(open(directory/'metadata.json'));C=np.asarray(pickle.load(open(directory/'calibrate.pkl','rb')));K=np.asarray(metadata['intrinsics'])
        tracks=np.asarray(raw['object_points']);T=len(tracks);image=cv2.imread(str(directory/'color/0/0.png'));height,width=image.shape[:2]
        points=(tracks[0]-C[0,:3,3])@C[0,:3,:3];screen=points@K[0].T;uv=screen[:,:2]/screen[:,2,None]
        visible=raw['object_visibilities'][0].astype(bool)&np.isfinite(tracks[0]).all(1)&(points[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
        front={}
        for idx in np.argsort(points[:,2]):
            if visible[idx]:front.setdefault(tuple((uv[idx]/8).astype(int)),int(idx))
        ids=np.array(list(front.values()),int);assert len(ids)>=20
        valid=np.asarray(raw['object_visibilities'][:,ids],bool)&np.isfinite(tracks[:,ids]).all(2)
        np.savez_compressed(out/'labels.npz',tracks=tracks[:,ids].astype('float32'),valid=valid,query_uv=(uv[ids]/[width,height]).astype('float32'),track_ids=ids)
        split=json.load(open(directory/'split.json'));write(out/'geometry.json',dict(camera_to_world=C.tolist(),intrinsics=K.tolist(),WH=[width,height],frames=T,train_end=split['train'][1],queries=len(ids)))
        features=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype='float16',shape=(T,2,256,12,12))
        for camera in (0,1):
            for start in range(0,T,12):
                imgs=[]
                for t in range(start,min(start+12,T)):
                    img=cv2.imread(str(directory/f'color/{camera}/{t}.png'))
                    if img is None:raise FileNotFoundError(f'{case} camera{camera} frame{t}')
                    imgs.append(cv2.cvtColor(cv2.resize(img,(384,384)),cv2.COLOR_BGR2RGB))
                batch=torch.tensor(np.stack(imgs),device='cuda').permute(0,3,1,2).float()/255
                with torch.no_grad():feat=torch.nn.functional.adaptive_avg_pool2d(enc((batch-mean)/std),(12,12))
                features[start:start+len(imgs),camera]=feat.cpu().numpy().astype('float16')
        features.flush();write(out/'complete.json',dict(case=case,frames=T,queries=len(ids),input_cameras=[0,1],feature_sha256=sha(out/'features.npy'),label_sha256=sha(out/'labels.npz'),geometry_sha256=sha(out/'geometry.json'),raw_tracks_sha256=sha(directory/'final_data.pkl')));print(case,T,len(ids),flush=True)

if __name__=='__main__':main()
