"""Position-query neural baseline with a disjoint simulated training bank."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,within_deadline
from scripts.position_ood.runtime import dense_model
from scripts.position_ood.fixture import controls
from scripts.position_ood.pilot import ACTIONS
from scripts.phystwin.sequential_fixture import points
from scripts.phystwin.v06_camera import render,track
from scripts.phystwin.sequential_reference import write


def bank(count=512,mesh=2,worker=0,workers=1,forward='dense'):
    tag=f'n{mesh}' if forward=='dense' else f'{forward}-n{mesh}'
    if forward=='extrapolated':
        from scripts.position_ood.extrapolated import ExtrapolatedPositionFEM
        assert mesh==3
        model=ExtrapolatedPositionFEM()
    else:model=dense_model(mesh)
    ids=np.argmin(np.linalg.norm(model.geo.rest[:,None]-points()[None],axis=-1),axis=0);visible=np.flatnonzero(np.isclose(points()[:,1],-.04))
    for i in range(worker,count,workers):
        within_deadline();path=BASE/f'neural-bank/{tag}/{i}.npz'
        if path.exists():continue
        rng=np.random.default_rng(291000+i);E=np.exp(rng.uniform(np.log(8000),np.log(40000)));K=np.exp(rng.uniform(np.log(1500),np.log(15000),2));g=np.r_[rng.uniform(-.3,.3,2),-1.];g*=9.81/np.linalg.norm(g);A=rng.uniform(-.15,.15);h=rng.uniform(-.5,.5);theta=model.tensor([*np.log([E,*K]),*g,A,h]);layout=i%3
        poses=np.array([controls(a,b) for a,b in ACTIONS]+[controls('translate_y',rng.uniform(-.006,.006)),controls('translate_z',rng.uniform(-.006,.006))]);fields=[]
        try:
            with torch.no_grad():
                for pose in poses:fields.append(model.predict(theta,model.tensor(pose),layout)[ids].cpu().numpy())
        except (ValueError,RuntimeError) as e:
            write(path.with_suffix('.failure.json'),dict(error=repr(e),index=i));continue
        ys=[]
        for u in fields[:3]:
            rgb,depth,c=render(points()+u,0,visible);ys.append(track(rgb,depth,c,visible,rng)-points()[visible])
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,fields=fields,poses=poses,observations=ys,index=i);tmp.replace(path);print('NEURAL_BANK',i,flush=True)

class Predictor(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(243,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU());self.decoder=nn.Sequential(nn.Linear(79,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,3))
        self.register_buffer('query',torch.tensor((points()-[.12,0,0])/[.12,.04,.04],dtype=torch.float32));self.register_buffer('pose_scale',torch.tensor([.006]*3+[.15]*3,dtype=torch.float32))
    def forward(self,observations,poses):
        feature=self.encoder(observations.reshape(-1,243)/.01);action=(poses/self.pose_scale).reshape(-1,12);context=torch.cat([feature,action],-1)[:,None,:].expand(-1,81,-1);query=self.query[None].expand(len(context),-1,-1)
        return .01*self.decoder(torch.cat([context,query],-1))

def train(mesh=2,epochs=1000,mode='id',count=512,forward='dense'):
    tag=f'n{mesh}' if forward=='dense' else f'{forward}-n{mesh}'
    within_deadline();paths=[BASE/f'neural-bank/{tag}/{i}.npz' for i in range(count)]
    missing=[str(p) for p in paths if not p.exists()]
    if missing:raise RuntimeError(f'Bank incomplete: {len(missing)} missing; do not silently select successful simulations')
    rows=[np.load(p) for p in paths];obs=np.array([r['observations'] for r in rows]);poses=np.array([r['poses'] for r in rows]);target=np.array([r['fields'] for r in rows]);actions=[0,1,2,3,10,11] if mode=='id' else list(range(12))
    # Validation objects belong only to the training bank, never the study test.
    trainids=np.arange(count)[np.arange(count)%5!=0];validids=np.arange(count)[np.arange(count)%5==0];torch.manual_seed(291);model=Predictor().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-5);rng=np.random.default_rng(291)
    obs=torch.tensor(obs,dtype=torch.float32,device='cuda');poses=torch.tensor(poses,dtype=torch.float32,device='cuda');target=torch.tensor(target,dtype=torch.float32,device='cuda');best=np.inf;history=[];dest=BASE/f'neural/{mode}-{tag}';dest.mkdir(parents=True,exist_ok=True);start=time.time()
    for epoch in range(epochs):
        within_deadline();model.train();idx=rng.choice(trainids,256);act=rng.choice(actions,256);pred=model(obs[idx],poses[idx,act]);loss=((pred-target[idx,act])/.01).square().mean();optimizer.zero_grad();loss.backward();optimizer.step();scheduler.step()
        if epoch%20==0 or epoch==epochs-1:
            model.eval()
            with torch.no_grad():
                vi=np.repeat(validids,len(actions));va=np.tile(actions,len(validids));value=float((model(obs[vi],poses[vi,va])-target[vi,va]).square().mean())
            if value<best:best=value;torch.save(model.state_dict(),dest/'best.pt')
            history.append(dict(epoch=epoch,train=float(loss),validation_rmse_mm=1000*np.sqrt(value)));print('NEURAL_TRAIN',mode,epoch,1000*np.sqrt(value),flush=True)
    write(dest/'summary.json',dict(state='complete',mode=mode,mesh=mesh,forward=forward,training_actions=actions,count=count,epochs=epochs,seed=291,best_validation_rmse_mm=1000*np.sqrt(best),elapsed_seconds=time.time()-start,history=history,scope='Disjoint synthetic training objects; position observations and tool poses only. id model never trains on x translation, rotations, opposed motion, diagonal motion, or doubled amplitude. all model sees those families on separate training objects and is reported separately.'))

def predict(split,i,mode='id',mesh=2,forward='dense'):
    tag=f'n{mesh}' if forward=='dense' else f'{forward}-n{mesh}'
    within_deadline()
    if 'test' in split.split('_'):
        from scripts.position_ood.seal import verify
        verify()
    from scripts.position_ood.observations import make
    obs=np.load(make(split,i));model=Predictor().cuda();model.load_state_dict(torch.load(BASE/f'neural/{mode}-{tag}/best.pt',map_location='cuda',weights_only=True));model.eval()
    with torch.no_grad():
        x=torch.tensor(obs['measurements'][:3],dtype=torch.float32,device='cuda')[None].expand(len(obs['poses']),-1,-1,-1);p=torch.tensor(obs['poses'],dtype=torch.float32,device='cuda');pred=model(x,p).cpu().numpy()
    suffix=mode if forward=='dense' and mesh==2 else f'{mode}-{tag}'
    path=BASE/f'neural-predictions/{split}-{i}-{suffix}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,predicted=pred,poses=obs['poses']);return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['bank','train'],default='bank');p.add_argument('--count',type=int,default=512);p.add_argument('--mesh',type=int,default=2);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--epochs',type=int,default=2000);p.add_argument('--mode',choices=['id','all'],default='id');p.add_argument('--forward',choices=['dense','extrapolated'],default='dense');a=p.parse_args()
    if a.task=='bank':bank(a.count,a.mesh,a.worker,a.workers,a.forward)
    else:train(a.mesh,a.epochs,a.mode,a.count,a.forward)
