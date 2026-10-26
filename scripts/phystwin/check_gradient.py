"""Directional finite-difference audit of the unchanged official physics gradient."""
import json,os,pickle,sys
from pathlib import Path
os.environ['PHYSTWIN_HEADLESS']='1';os.environ['WANDB_MODE']='disabled'
root=Path('runs/third_party/PhysTwin').resolve()
sys.path.insert(0,str(root))
import torch
import warp as wp
from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg
cfg.load_from_yaml(str(root/'configs/real.yaml'))
with (root/'experiments_optimization/single_lift_sloth/optimal_params.pkl').open('rb') as f:
    cfg.set_optimal_params(pickle.load(f))
trainer=InvPhyTrainerWarp(str(root/'data/different_types/single_lift_sloth/final_data.pkl'),
    str(Path('runs/phystwin-pilot-v01/gradient').resolve()),train_frame=59,pure_inference_mode=True)
s=trainer.simulator
s.set_init_state(s.wp_init_vertices,s.wp_init_velocities)
for frame in range(1,40):
    s.set_controller_target(frame,pure_inference=True)
    wp.capture_launch(s.forward_graph)
    s.set_init_state(s.wp_states[-1].wp_x,s.wp_states[-1].wp_v)
x=wp.clone(s.wp_states[0].wp_x);v=wp.clone(s.wp_states[0].wp_v)
y=wp.to_torch(s.wp_spring_Y).detach().clone()

def reset(delta):
    s.tape.zero();s.clear_loss();s.set_init_state(x,v)
    s.set_controller_target(40)
    s.set_spring_Y(y+delta)

reset(0)
wp.capture_launch(s.graph)
autograd=float(wp.to_torch(s.wp_spring_Y.grad).sum().item())
base_loss=float(wp.to_torch(s.loss).item())
checks=[]
for eps in [0.03,0.01,0.003]:
    losses=[]
    for sign in [-1,1]:
        reset(sign*eps)
        wp.capture_launch(s.graph)  # Same captured objective and normalization as autodiff.
        losses.append(float(wp.to_torch(s.loss).item()))
    fd=(losses[1]-losses[0])/(2*eps)
    checks.append(dict(epsilon=eps,finite_difference=fd,
        relative_difference=abs(fd-autograd)/max(abs(fd),abs(autograd),1e-15)))
result=dict(frame=40,base_loss=base_loss,autograd=autograd,checks=checks,
    scope='Finite differences and autodiff use the identical captured graph. One scalar direction: uniform log-spring-stiffness change, one training frame, fixed starting state and prescribed controls. Does not validate all gradients or long-horizon derivatives.')
p=Path('runs/phystwin-pilot-v01/gradient-check.json');p.write_text(json.dumps(result,indent=2)+'\n')
print('GRADIENT_RESULT',json.dumps(result))
