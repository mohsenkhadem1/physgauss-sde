import numpy as np
from scipy.integrate import solve_ivp
from experiments.continue_p2_contact_dop853_v02 import manual_dop853,CONFIGS

def test_manual_step_matches_solve_ivp_accepted_times_states():
 M=np.array([[2.,.3],[.3,1.]]);K=np.array([[900.,-100.],[-100.,500.]]);y=np.array([.01,-.03,.2,-.1]);outputs=np.linspace(0,.01,9);c=CONFIGS[1];atol=np.full(4,1e-12)
 def rhs(t,y):return np.r_[y[2:],np.linalg.solve(M,-K@y[:2])]
 a=manual_dop853(rhs,y,[0,.01],outputs,c,atol);b=solve_ivp(rhs,[0,.01],y,method='DOP853',rtol=c['rtol'],atol=atol,max_step=1/30000,dense_output=True,t_eval=outputs)
 np.testing.assert_array_equal(a['accepted_times'],b.sol.ts)
 np.testing.assert_allclose(a['accepted_states'],b.sol(b.sol.ts),atol=1e-16,rtol=0)
 np.testing.assert_allclose(a['output_states'],b.y,atol=1e-16,rtol=0)
 assert a['nfev']<b.nfev

def test_freefall_all_accepted_callback_prefix_and_aux():
 g=-9.81;m=.96;t0=1/12;end=.1;y=np.array([.5*g*t0*t0,g*t0,0.]);seen=[]
 def rhs(t,y):return np.array([y[1],g,m*g])
 def accepted(info):
  seen.append(info['accepted_times'][-1]);assert len(info['accepted_states'])==len(info['accepted_times'])
 a=manual_dop853(rhs,y,[t0,end],np.linspace(t0,end,9),CONFIGS[1],np.full(3,1e-12),accepted)
 np.testing.assert_array_equal(a['accepted_times'],seen)
 np.testing.assert_allclose(a['accepted_states'][0],.5*g*a['accepted_times']**2,atol=1e-14)
 np.testing.assert_allclose(m*(a['accepted_states'][1]-y[1])-a['accepted_states'][2],0,atol=1e-14)
