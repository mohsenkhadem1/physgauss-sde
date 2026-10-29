"""Reuse fitted native PhysTwin weights for transfer and timestep audits."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.phystwin import Trainer,configure_simulation,schedule,SHIFT,cfg,logger,wp,torch,np
from scripts.position_ood.observations import make
from scripts.position_ood.study import BASE,within_deadline
from scripts.phystwin.sequential_fixture import points
from scripts.phystwin.sequential_reference import write

def run(split,i,tag='y3-dt1',refinement=1):
    within_deadline()
    if 'test' in split.split('_'):
        from scripts.position_ood.seal import verify
        verify()
    original='development' if split=='transfer_development' else ('test_id' if split=='transfer_test' else split)
    source=BASE/f'phystwin/{original}-{i}/{tag}';summary=json.loads((source/'summary.json').read_text());configure_simulation(refinement,summary['initial_Y']);np.random.seed(42);torch.manual_seed(42)
    destination=BASE/f'phystwin/{split}-{i}/{tag}';destination.mkdir(parents=True,exist_ok=True);logger.set_log_file(path=str(destination),name=f'predict-dt{refinement}');trainer=Trainer(str(source/'train.pkl'),str(destination),train_frame=1+3*summary["frames"])
    weights=torch.load(source/'best.pt',map_location=cfg.device,weights_only=True)
    with torch.no_grad():wp.to_torch(trainer.simulator.wp_spring_Y).copy_(weights)
    obs=np.load(make(split,i));data,starts,ends=schedule(obs['poses'],frames=summary['frames']);pred,drift,_,_=trainer.rollout(data,starts,ends)
    path=destination/('predictions.npz' if split.startswith('transfer_') and refinement==1 else f'predictions-audit-dt{refinement}.npz');np.savez_compressed(path,predicted=pred-SHIFT-points(),drift_m=drift,poses=obs['poses'])
    write(path.with_suffix('.json'),dict(source=str(source.relative_to(BASE)),refinement=refinement,dt=cfg.dt,scope='Fixed previously fitted spring parameters, new positional queries or numerical timestep audit. No new response labels used.'));return path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--object',type=int,default=1);p.add_argument('--tag',default='y3-dt1');p.add_argument('--refinement',type=int,default=2);a=p.parse_args();run(a.split,a.object,a.tag,a.refinement)
