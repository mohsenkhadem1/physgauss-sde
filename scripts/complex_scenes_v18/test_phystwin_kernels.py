import numpy as np
import warp as wp
from scripts.complex_scenes_v18.phystwin_scene import kernels,graph


def test_author_force_balance_and_rest_scaled_stiffness():
    native=kernels();device='cpu'
    def arr(a,t=wp.float32):return wp.array(np.asarray(a),dtype=t,device=device)
    x=arr([[0,0,1],[.022,0,1]],wp.vec3);v=arr(np.zeros((2,3)),wp.vec3)
    empty=arr(np.empty((0,3)),wp.vec3);f=wp.zeros(2,dtype=wp.vec3,device=device)
    wp.launch(native.eval_springs,dim=1,inputs=[x,v,empty,empty,2,arr([[0,1]],wp.vec2i),arr([.02]),arr([np.log(1000)]),0.,0.,1e8,f],device=device)
    np.testing.assert_allclose(f.numpy(),[[100,0,0],[-100,0,0]],atol=.001)


def test_graph_is_deterministic_unique_and_undirected():
    rest=np.stack(np.meshgrid(np.arange(3),np.arange(3),np.arange(3),indexing='ij'),-1).reshape(-1,3)
    a=graph(rest);np.testing.assert_array_equal(a,graph(rest))
    assert (a[:,0]<a[:,1]).all()and len(a)==len(np.unique(a,axis=0))
