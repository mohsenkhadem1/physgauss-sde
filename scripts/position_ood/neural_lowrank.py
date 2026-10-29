"""Strong position-only baseline with a learned low-rank spatial decoder.

The spatial basis is estimated from training objects only. At canonical query
points it is stored explicitly; the same quadratic interpolation as the query
mesh extends it to arbitrary positions. No physical labels are neural inputs.
"""
from pathlib import Path
import argparse,sys,time
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,within_deadline
from scripts.phystwin.sequential_reference import write

class LowRankPredictor(nn.Module):
    def __init__(self,rank=32):
        super().__init__();self.network=nn.Sequential(nn.Linear(255,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,rank))
        self.register_buffer('basis',torch.zeros(rank,243));self.register_buffer('mean',torch.zeros(243));self.register_buffer('pose_scale',torch.tensor([.006]*3+[.15]*3,dtype=torch.float32))
    def forward(self,observations,poses):
        x=torch.cat([observations.reshape(-1,243)/.01,(poses/self.pose_scale).reshape(-1,12)],-1);coeff=self.network(x)
        return (self.mean+coeff@self.basis).reshape(-1,81,3)

def train(tag='n2',mode='all',count=512,epochs=20000,rank=32):
    paths=[BASE/f'neural-bank/{tag}/{i}.npz' for i in range(count)]
    if not all(p.exists() for p in paths):raise RuntimeError('Complete declared bank required')
    rows=[np.load(p) for p in paths];obs=np.array([r['observations'] for r in rows]);poses=np.array([r['poses'] for r in rows]);target=np.array([r['fields'] for r in rows]);actions=[0,1,2,3,10,11] if mode=='id' else list(range(12));trainids=np.arange(count)[np.arange(count)%5!=0];validids=np.arange(count)[np.arange(count)%5==0]
    matrix=target[trainids][:,actions].reshape(-1,243);mean=matrix.mean(0);_,singular,V=np.linalg.svd(matrix-mean,full_matrices=False);basis=V[:rank]*np.maximum(singular[:rank,None]/np.sqrt(len(matrix)),1e-6);reconstructed=mean+((matrix-mean)@V[:rank].T)@V[:rank];basis_error=1000*np.sqrt(np.mean((matrix-reconstructed)**2))
    torch.manual_seed(291);model=LowRankPredictor(rank).cuda();model.mean.copy_(torch.tensor(mean,dtype=torch.float32,device='cuda'));model.basis.copy_(torch.tensor(basis,dtype=torch.float32,device='cuda'));optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-5);rng=np.random.default_rng(291)
    obs=torch.tensor(obs,dtype=torch.float32,device='cuda');poses=torch.tensor(poses,dtype=torch.float32,device='cuda');target=torch.tensor(target,dtype=torch.float32,device='cuda');best=np.inf;history=[];dest=BASE/f'neural/lowrank-{mode}-{tag}';dest.mkdir(parents=True,exist_ok=True);start=time.time()
    for epoch in range(epochs):
        within_deadline();idx=rng.choice(trainids,256);act=rng.choice(actions,256);pred=model(obs[idx],poses[idx,act]);loss=((pred-target[idx,act])/.01).square().mean();optimizer.zero_grad();loss.backward();optimizer.step();scheduler.step()
        if epoch%100==0 or epoch==epochs-1:
            with torch.no_grad():
                vi=np.repeat(validids,len(actions));va=np.tile(actions,len(validids));value=float((model(obs[vi],poses[vi,va])-target[vi,va]).square().mean())
            if value<best:best=value;torch.save(model.state_dict(),dest/'best.pt')
            history.append(dict(epoch=epoch,train=float(loss),validation_rmse_mm=1000*np.sqrt(value)));print('LOWRANK_TRAIN',tag,mode,epoch,1000*np.sqrt(value),flush=True)
    write(dest/'summary.json',dict(state='complete',tag=tag,mode=mode,rank=rank,count=count,epochs=epochs,training_actions=actions,training_basis_reconstruction_rmse_mm=float(basis_error),best_validation_rmse_mm=1000*np.sqrt(best),history=history,elapsed_seconds=time.time()-start,scope='Training-bank-only spatial basis and position/pose neural map; no physical labels. Validation objects are disjoint from basis estimation and network fitting.'))

def predict(split,i,tag='extrapolated-n3',mode='all'):
    within_deadline()
    if 'test' in split.split('_'):
        from scripts.position_ood.seal import verify
        verify()
    from scripts.position_ood.observations import make
    obs=np.load(make(split,i));weights=torch.load(BASE/f'neural/lowrank-{mode}-{tag}/best.pt',map_location='cuda',weights_only=True);model=LowRankPredictor(weights['basis'].shape[0]).cuda();model.load_state_dict(weights)
    with torch.no_grad():
        x=torch.tensor(obs['measurements'][:3],dtype=torch.float32,device='cuda')[None].expand(len(obs['poses']),-1,-1,-1);p=torch.tensor(obs['poses'],dtype=torch.float32,device='cuda');pred=model(x,p).cpu().numpy()
    path=BASE/f'neural-predictions/{split}-{i}-lowrank-{mode}-{tag}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,predicted=pred,poses=obs['poses']);return path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',default='n2');p.add_argument('--mode',choices=['id','all'],default='all');p.add_argument('--epochs',type=int,default=20000);a=p.parse_args();train(a.tag,a.mode,epochs=a.epochs)
