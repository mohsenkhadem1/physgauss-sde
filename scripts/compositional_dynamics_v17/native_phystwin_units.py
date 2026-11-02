"""Actual unchanged native kernel dimensional-invariance check; no benchmark data."""
import os,sys,json,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['PYNPUT_BACKEND']='dummy'
from scripts.compositional_dynamics_v17.native_phystwin_fit import AUTHOR,ROOT,sha,write
sys.path.insert(0,str(AUTHOR))
import numpy as np
import torch,warp as wp
from qqtt.utils import cfg
from qqtt.model.diff_simulator.spring_mass_warp import SpringMassSystemWarp

def main():
    cfg.device='cuda:0';cfg.data_type='synthetic';cfg.use_graph=True
    states=[];start=time.monotonic()
    for scale in [1.,.00216]:
        x=torch.tensor([[0.,0.,.2],[.021,0.,.2]],device='cuda:0')
        sim=SpringMassSystemWarp(x,torch.tensor([[0,1]],dtype=torch.int32,device='cuda:0'),torch.tensor([.02],device='cuda:0'),torch.tensor([1.,1.5],device='cuda:0')*scale,dt=.000025,num_substeps=40,spring_Y=30000.*scale,spring_Y_min=0.,spring_Y_max=100000.*scale,dashpot_damping=100.*scale,drag_damping=3.,collide_elas=0.,collide_fric=0.,num_object_points=2,gt_object_points=torch.stack([x,x]),disable_backward=True)
        sim.set_init_state(sim.wp_init_vertices,sim.wp_init_velocities,pure_inference=True);sim.step();wp.synchronize()
        states.append((sim.wp_states[-1].wp_x.numpy().copy(),sim.wp_states[-1].wp_v.numpy().copy()))
    dx=float(np.max(abs(states[0][0]-states[1][0])));dv=float(np.max(abs(states[0][1]-states[1][1])))
    assert dx<1e-7 and dv<1e-5
    write(ROOT/'reports/compositional-dynamics-v17/native-phystwin-unit-scaling.json',dict(passed=True,position_max_m=dx,velocity_max_m_s=dv,scale=.00216,wall_seconds=time.monotonic()-start,scope='Actual author spring/dashpot/integrator acceleration invariance under simultaneous m,Y,c scaling; drag unchanged. No contact/fitting data.',source_sha256={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),AUTHOR/'qqtt/model/diff_simulator/spring_mass_warp.py']}))
if __name__=='__main__':main()
