import numpy as np
from scripts.compositional_dynamics_v17.native_newton_dynamics import build,p1_map,surface_queries,energy

def test_native_mesh_mass_and_affine_surface_transfer():
    m,X,tets,mass,params=build([4,2,2],'drop','cpu')
    assert abs(mass.sum()-.27)<1e-7
    assert params['native_grid_helper_unused_mass_kg']>.27
    q=surface_queries();ids,w=p1_map(X,tets,q);A=np.array([[1.1,.2,0],[0,.9,.1],[0,0,1.05]]);shift=np.array([.01,.02,.03])
    pred=np.einsum('nk,nkj->nj',w,(X@A.T+shift)[ids]);assert np.allclose(pred,q.reshape(-1,3)@A.T+shift,atol=1e-12)
    assert np.isclose(params['vertex_contact_ke_N_m']*15,4e7*.12*.06)

def test_native_energy_reference_rotation_and_uniform_compression_contact():
    _,X,tets,mass,params=build([4,2,2],'drop','cpu');R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    e,J,p=energy(X@R.T,np.zeros_like(X),X,tets,mass,params,'released_stretch')
    assert abs(e[1])<1e-12 and abs(J-1)<1e-12
    moved=X.copy();moved[:,2]-=.021
    e,J,p=energy(moved,np.zeros_like(X),X,tets,mass,params,'drop')
    assert np.isclose(p,.001)and np.isclose(e[3],.5*4e7*.12*.06*.001**2)

def test_failure_retains_initialized_prefix():
    import tempfile,json
    from pathlib import Path
    from scripts.compositional_dynamics_v17 import native_newton_dynamics as module
    original=module.newton.solvers.SolverVBD;oldbase=module.BASE
    class BrokenStep:
        def __init__(self,*args,**kwargs):self.native=original(*args,**kwargs)
        def observables(self,*args,**kwargs):return self.native.observables(*args,**kwargs)
        def step(self,*args,**kwargs):raise RuntimeError('Deliberate fixture failure')
    with tempfile.TemporaryDirectory()as folder:
        module.BASE=Path(folder);module.newton.solvers.SolverVBD=BrokenStep
        try:
            try:module.run('drop-time-0.0005','cpu',smoke_steps=2)
            except RuntimeError as e:assert 'Deliberate' in str(e)
            else:raise AssertionError('Injected failure lost')
            record=json.loads((Path(folder)/'api-preflight/complete.json').read_text())
            assert record['complete']is False and record['partial_trajectory']is True and record['last_saved_time_s']==0.
            with np.load(Path(folder)/'api-preflight/trajectory.npz')as a:assert len(a['times_s'])==1 and a['positions'].shape[0]==1
        finally:module.BASE=oldbase;module.newton.solvers.SolverVBD=original
