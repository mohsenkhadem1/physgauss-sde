import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from .protocol import BASE,REPORT,ROOT,sha,write
from .data import Sequence,batch
from .model import SceneQueryModel

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['visual','image_blind'],required=True);a=p.parse_args();torch.set_num_threads(2)
    plan=json.load(open(REPORT/'protocol.json'));torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed'])
    out=BASE/a.method;out.mkdir(exist_ok=False);seqs=[Sequence(c)for c in plan['cases']['train']]
    model=SceneQueryModel(a.method=='image_blind').cuda();opt=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=.01)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,plan['steps'],eta_min=1e-5)
    sources={str(p.relative_to(ROOT)):sha(p)for p in (Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('data.py'))}
    write(out/'plan.json',dict(method=a.method,protocol_sha256=sha(REPORT/'protocol.json'),sources=sources,train_cases=plan['cases']['train'],parameters=sum(p.numel()for p in model.parameters()),input_manifests={s.case:sha(BASE/'prepared'/s.case/'complete.json')for s in seqs}))
    started=time.monotonic();history=[]
    for step in range(plan['steps']):
        examples=[]
        while len(examples)<plan['batch']:
            s=seqs[rng.integers(len(seqs))];t=int(rng.integers(s.meta['train_end']));valid=s.eligible(t)
            if len(valid)<8:continue
            ids=rng.choice(valid,plan['queries'],replace=len(valid)<plan['queries']);examples.append(s.sample(t,ids))
        x,y=batch(examples,'cuda');pred,source=model(**x)
        loss=torch.nn.functional.smooth_l1_loss(pred,y['position'],beta=.02)+.3*torch.nn.functional.smooth_l1_loss(source,y['source'],beta=.02)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();schedule.step()
        if step%100==0:
            row=dict(step=step,loss=float(loss),tracking_mm=float(torch.linalg.vector_norm(pred-y['position'],dim=-1).mean()*1000),elapsed_s=time.monotonic()-started);history.append(row);write(out/'progress.json',row);print(row,flush=True)
    checkpoint=out/'model.pt';torch.save(dict(state_dict=model.state_dict(),method=a.method),checkpoint)
    assert all(sha(ROOT/p)==h for p,h in sources.items())
    write(out/'complete.json',dict(completed=True,steps=plan['steps'],elapsed_s=time.monotonic()-started,checkpoint_sha256=sha(checkpoint),history=history,source_sha256=sources));print('COMPLETE',a.method,flush=True)

if __name__=='__main__':main()
