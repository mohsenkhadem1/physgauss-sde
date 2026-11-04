"""Inputs and supervision are separated before model invocation."""
import json
import numpy as np,torch
from .protocol import BASE

class Sequence:
    def __init__(self,case):
        self.case=case;folder=BASE/'prepared'/case;self.meta=json.load(open(folder/'geometry.json'))
        self.C=np.array(self.meta['camera_to_world'],np.float32);self.K=np.array(self.meta['intrinsics'],np.float32)
        self.features=np.load(folder/'features.npy',mmap_mode='r')
        with np.load(folder/'labels.npz')as f:self.labels={k:f[k].copy()for k in f.files}
    def sample(self,t,ids):
        uv=self.labels['query_uv'][ids];pixels=np.c_[uv*np.array(self.meta['WH']),np.ones(len(ids))]
        rays=pixels@np.linalg.inv(self.K[0]).T
        features=np.concatenate([self.features[0],self.features[t]],axis=0).astype('float32')
        xy=np.stack(np.meshgrid((np.arange(12)+.5)/12,(np.arange(12)+.5)/12,indexing='xy'),-1).reshape(-1,2)
        geometry=[]
        for time,camera in ((0,0),(0,1),(t,0),(t,1)):
            wh=np.array(self.meta['WH']);kin=np.array([self.K[camera,0,0]/wh[0],self.K[camera,1,1]/wh[1],self.K[camera,0,2]/wh[0],self.K[camera,1,2]/wh[1]])
            geometry.append(np.c_[xy,np.full(144,time/30),np.broadcast_to(self.C[camera,:3].ravel(),(144,12)),np.broadcast_to(kin,(144,4))])
        inputs=dict(features=features,token_geometry=np.concatenate(geometry).astype('float32'),uv=uv,rays=rays.astype('float32'),camera0=self.C[0].copy(),time_s=np.float32(t/30))
        target=dict(position=self.labels['tracks'][t,ids],source=self.labels['tracks'][0,ids])
        return inputs,target
    def eligible(self,t):return np.flatnonzero(self.labels['valid'][t])

def batch(examples,device):
    inputs={k:torch.as_tensor(np.stack([x[0][k]for x in examples]),device=device)for k in examples[0][0]}
    targets={k:torch.as_tensor(np.stack([x[1][k]for x in examples]),device=device)for k in examples[0][1]}
    return inputs,targets
