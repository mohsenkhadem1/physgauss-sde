"""Full native simulator step API feasibility, not inverse benchmark qualification."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
AUTHOR=ROOT/'runs/compositional-dynamics-v17/PhysTwin-author'
sys.path.insert(0,str(AUTHOR))
import numpy as np
import torch
from qqtt.utils import cfg
cfg.device='cuda:0';cfg.data_type='synthetic';cfg.use_graph=True
from qqtt.model.diff_simulator.spring_mass_warp import SpringMassSystemWarp
import warp as wp

def main():
    start=time.monotonic();device='cuda:0'
    x=torch.tensor([[-.01,0,.1],[.01,0,.1]],device=device)
    simulator=SpringMassSystemWarp(x,torch.tensor([[0,1]],dtype=torch.int32,device=device),torch.tensor([.02],device=device),torch.tensor([.001,.002],device=device),dt=.0001,num_substeps=10,spring_Y=1000.,collide_elas=0.,collide_fric=0.,dashpot_damping=0.,drag_damping=0.,num_object_points=2,gt_object_points=torch.stack([x,x]),disable_backward=True)
    simulator.set_init_state(simulator.wp_init_vertices,simulator.wp_init_velocities,pure_inference=True)
    simulator.step();wp.synchronize()
    result=simulator.wp_states[-1].wp_x.numpy();velocity=simulator.wp_states[-1].wp_v.numpy()
    expected_v=np.array([0,0,-9.8*.001]);assert np.isfinite(result).all() and np.isfinite(velocity).all()
    error=float(np.max(np.abs(velocity-expected_v)))
    assert error<1e-7 and np.max(abs(result[1]-result[0]-np.array([.02,0,0])))<1e-7
    out=ROOT/'runs/compositional-dynamics-v17/native-physics-smoke';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'state.npz',positions=result,velocities=velocity,masses=np.array([.001,.002]))
    paths=[Path(__file__),AUTHOR/'qqtt/model/diff_simulator/spring_mass_warp.py',out/'state.npz']
    report=dict(passed=True,scope='Unmodified author simulator complete step, unequal SI masses, freefall with native gravity9.8; no contact, fitting or benchmark performance claim.',author_commit='81c718790a37e5e0102eb77af2c6edd34a9db25f',device='physical GPU0 only',velocity_max_error_m_s=error,wall_seconds=time.monotonic()-start,torch=torch.__version__,warp=wp.__version__,sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (ROOT/'reports/compositional-dynamics-v17/native-phystwin-physics-smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
