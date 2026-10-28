"""Stronger-load v08 references; old sealed campaigns are read-only."""
from pathlib import Path
import argparse,itertools,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin import sequential_reference as old
from scripts.phystwin.sequential_fixture import SequentialFixture,LAYOUTS,ACTIONS
from experiments.campaign_static_uniaxial_bar import StaticFailure
ROOT=old.ROOT;BASE=ROOT/'runs/phystwin-informative-v08';OUT=ROOT/'reports/phystwin-informative-v08';BOUNDS=old.BOUNDS;write=old.write
class StrongFixture(SequentialFixture):
 def set_action(self,action):
  super().set_action(action)
  gravity=np.zeros_like(self.load);gravity[:,2]=-.02*self.volume_weights
  self.load=gravity+8*(self.load-gravity)
 def solve(self,initial=None,**kwargs):
  def recoverable(error):return isinstance(error,StaticFailure) or 'PCG' in str(error)
  try:return super().solve(initial=initial,**kwargs)
  except RuntimeError as error:
   if not recoverable(error):raise
   print('SOLVER_RECOVERY',self.n,str(error),flush=True);self.clear_linear_cache()
   if 'PCG' in str(error):
    try:return super().solve(initial=initial,**kwargs)
    except RuntimeError as retry:
     if not recoverable(retry):raise
  target=self.load.copy();state=None;fraction=0.;step=.125
  try:
   while fraction<1.-1e-12:
    following=min(1.,fraction+step);self.load=target*following;self.clear_linear_cache()
    try:candidate=super().solve(initial=None if state is None else state['displacement_local_m'],**kwargs)
    except RuntimeError as error:
     if not recoverable(error) or step<1/128:raise
     step/=2;print('LOAD_STEP_RETRY',self.n,following,step,str(error),flush=True);continue
    state=candidate;fraction=following;step=min(.25,step*2)
   return state
  finally:self.load=target
def protocol():
 p=BASE/'protocol.json'
 if p.exists():return json.loads(p.read_text())
 s=dict(version=8,load_scale=8,reference_levels=[4,6,8,10,12,14],bank_grid=4,noise_repeats=3,planning_grid=65,prediction_grid=65,utility_samples=64,costs=[1.]*6+[.25],cost_penalty_m2=1e-8,action_names=[*ACTIONS[1:7],'deformed_end_view'],fixed_sequence=[4,1],scope='Eightfold applied forces; gravity unchanged. New right-end view measures nine additional points under the stronger initial load. Same nonlinear material and two uncertain supports. Static reference holds, synthetic tracked RGB-D, known forces/geometry. PhysTwin ensemble disagreement is a heuristic, not calibrated information gain.')
 for split,n,seed in [('development',12,810),('calibration',24,811),('test',24,812)]:
  rng=np.random.default_rng(seed);s[split]=[dict(layout=LAYOUTS[i%3],parameters=np.exp(rng.uniform(BOUNDS[:,0],BOUNDS[:,1])).tolist()) for i in range(n)]
 write(p,s);return s
def configure():
 old.BASE=BASE;old.OUT=OUT;old.protocol=protocol;old.SequentialFixture=StrongFixture

def main():
 import time,traceback
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=6);p.add_argument('--bank-only',action='store_true');p.add_argument('--references-only',action='store_true');a=p.parse_args();s=protocol();configure()
 status=BASE/f"{'bank' if a.bank_only else 'reference'}-{a.worker}-status.json"
 try:
  write(status,dict(state='running',time=time.time()))
  splits=['bank'] if a.bank_only else (['development','calibration','test'] if a.references_only else ['development','bank','calibration','test'])
  for split in splits:
   for i,(name,r) in enumerate(old.jobs(split)):
    if i%a.workers!=a.worker:continue
    if split=='bank':old.solve_level(name,r,4,'cuda:0')
    else:old.reference(name,r,'cuda:0')
  write(status,dict(state='complete',time=time.time()))
 except BaseException as e:write(status,dict(state='failed',time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
