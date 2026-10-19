from pathlib import Path
import sys,math,unittest,json
import torch
from test_pacnerf import fixture
from physgauss.neural_marginal import BlockLoadElimination
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pacnerf_floor import FloorRenderer,FloorConditionedModel,FloorInnovation
from pacnerf_discrepancy import DiscrepancyModel
from pacnerf_conditioned_transport import ConditionedTransport


def model(force='marginal'):
    base=fixture('neural',force);base.renderer=FloorRenderer(base.renderer)
    return FloorConditionedModel(DiscrepancyModel(base))


def translation(m,dy=0.,dz=0.):
    delta=torch.zeros_like(m.data['rest']);delta[:,1]=dy;delta[:,2]=dz
    return torch.einsum('nc,ncm->m',delta,m.basis)


def direct_floor_reference(renderer,q,specs):
    optical,fractions=renderer.source.components(q,specs)
    depth,ground,_=renderer.geometry(q,specs)
    cdf=torch.cat((torch.zeros_like(fractions[:,:,:1]),fractions.cumsum(2)),2)
    near=.5*(1+torch.erf(-depth/(math.sqrt(2)*renderer.radius)))
    safe=torch.where(torch.isfinite(ground),ground,torch.full_like(ground,1e6))
    limit=.5*(1+torch.erf((safe[None,:,None]-depth[...,None])/(math.sqrt(2)*renderer.radius)))
    # Deliberately allocate the complete slab*splat*pixel tensor: independent
    # direct clipping, with no floor-slab classification or partial-bin logic.
    clipped=torch.minimum(torch.maximum(cdf[...,None],near[:,:,None,:,None]),limit[:,:,None])
    mass=(clipped[:,:,1:]-clipped[:,:,:-1]).clamp_min(0)
    tau=torch.einsum('btsgp,btgp->btsp',mass,optical)
    source=torch.einsum('btsgp,btgp,gc->btspc',mass,optical,renderer.colors)
    phi=torch.where(tau>0,-torch.expm1(-tau)/tau.clamp_min(1e-300),torch.ones_like(tau))
    rgb=(torch.exp(-(tau.cumsum(2)-tau))[...,None]*phi[...,None]*source).sum(2)
    bg=torch.stack([renderer.data['backgrounds'][c].flatten(0,1) for _,c in specs])
    rgb+=torch.exp(-tau.sum(2))[...,None]*bg[None]
    return rgb.reshape(len(q),len(specs),renderer.data['size'],renderer.data['size'],3)


class FloorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_complete_worker_model_specification_is_serializable(self):
        spec=model().specification()
        self.assertEqual(spec['version'],'floor-conditioned-hierarchical-v2')
        self.assertEqual(spec['no_grad_spatial_tail_alpha_cutoff'],2.**-80)
        json.dumps(spec,allow_nan=False)

    def test_plane_intersections_match_analytic_camera_rays(self):
        m=model();q=torch.zeros(1,1,m.modes,dtype=torch.float64)
        _,depth,_=m.renderer.geometry(q,[(0,0)])
        rows=m.renderer.pixels[...,1].flatten()
        expected=torch.where(rows>3.5,6/(rows+.5-4),torch.full_like(rows,torch.inf))
        torch.testing.assert_close(depth[0],expected,atol=1e-12,rtol=1e-12)

    def test_partial_slab_compositor_matches_full_ray_clipping(self):
        m=model();q=torch.stack((translation(m,-.2),translation(m,-.5),translation(m,-.8)))[None]
        specs=[(1,0),(2,2),(3,4)]
        with torch.no_grad():
            actual=m.renderer(q,specs);expected=direct_floor_reference(m.renderer,q,specs)
        torch.testing.assert_close(actual,expected,atol=5e-13,rtol=0)

    def test_floor_suppresses_buried_splats_and_near_plane_hides_behind_camera(self):
        m=model();specs=[(1,0)]
        pose=translation(m,-1.5)
        with torch.no_grad():
            buried=m.renderer(pose[None,None],specs)
            original=m.renderer.source(pose[None,None],specs)
            behind=m.renderer(translation(m,0.,4.)[None,None],specs)
        error=float((buried[0,0]-m.data['backgrounds'][0]).abs().max())
        previous=float((original[0,0]-m.data['backgrounds'][0]).abs().max())
        # The retained projected screen Gaussian includes .3px anti-alias
        # variance and unbounded tails. The floor clips radiance behind the
        # plane, not a hard mask on Gaussian centers. Feasibility conditioning
        # separately excludes material centers deeply below that plane.
        self.assertLess(error,.001);self.assertLess(error,previous/100)
        torch.testing.assert_close(behind[0,0],m.data['backgrounds'][0],atol=1e-10,rtol=0)

    def test_selected_pixels_match_full_floor_render(self):
        m=model();q=translation(m,-.45)[None,None];ids=torch.tensor([[0,12,31,55]])
        with torch.no_grad():
            full=m.renderer(q,[(1,0)]).flatten(2,3)
            part=m.renderer(q,[(1,0)],pixel_indices=ids)
        torch.testing.assert_close(part,full[:,:,ids[0]],atol=5e-13,rtol=0)

    def test_floor_render_directional_gradient_near_contact(self):
        m=model();q=translation(m,-.4).requires_grad_();direction=torch.linspace(-1,1,m.modes,dtype=q.dtype);direction/=direction.norm()
        value=m.renderer(q[None,None],[(1,0)]).square().sum();gradient,=torch.autograd.grad(value,q)
        eps=1e-5
        finite=(m.renderer((q+eps*direction)[None,None],[(1,0)]).square().sum()-m.renderer((q-eps*direction)[None,None],[(1,0)]).square().sum())/(2*eps)
        torch.testing.assert_close(finite,gradient@direction,atol=2e-5,rtol=2e-4)

    def test_feasibility_event_keeps_force_marginalization_exact(self):
        marginal=model();joint=model('joint');z=torch.zeros(1,marginal.dimension,dtype=torch.float64)
        pose=translation(marginal,-.5);start=2+marginal.modes+4*2*marginal.modes
        z[:,start:start+marginal.modes]=pose/.1
        self.assertGreater(float(marginal.floor_penalty(z)),1.)
        residual,cov,g,_=marginal.terms(z);elimination=BlockLoadElimination(g[0],cov)
        mean=elimination.conditional_mean(residual.flatten(1))
        explicit=torch.cat((z[:,:marginal.base_dimension],mean,z[:,-3:]),-1)
        difference=marginal.objective(z)-joint.objective(explicit)
        expected=.5*(elimination.logdet-elimination.logdet_q)-.5*marginal.force_dimension*math.log(2*math.pi)
        torch.testing.assert_close(difference[0],expected[0],atol=1e-7,rtol=1e-8)

    def test_floor_innovation_density_and_guide_jacobian_are_exact(self):
        for force in ['marginal','joint']:
            m=model(force);innovation=FloorInnovation(m)
            z=torch.zeros(1,m.dimension,dtype=torch.float64)
            physical,jac=innovation.transform(z)
            torch.testing.assert_close(innovation.objective(z),m.objective(physical)-jac,atol=2e-7,rtol=1e-10)
            anchor=torch.zeros(m.dimension,dtype=z.dtype);guide=ConditionedTransport(m,anchor)
            actual,jac=guide.transform(z)
            torch.testing.assert_close(guide.objective(z),m.objective(actual)-jac,atol=2e-7,rtol=1e-10)

    def test_boundary_quadratic_matches_autodiff(self):
        m=model();q=translation(m,-.4).requires_grad_()
        def penalty(p):return torch.nn.functional.softplus(-(m.floor_rest+p@m.floor_basis.T)/m.floor_tolerance).sum()
        gradient,=torch.autograd.grad(penalty(q),q)
        hessian=torch.autograd.functional.hessian(penalty,q)
        g,h=m.floor_quadratic(q[None])
        torch.testing.assert_close(g[0],gradient,atol=1e-8,rtol=1e-10)
        torch.testing.assert_close(h[0],hessian,atol=1e-7,rtol=1e-10)

    def test_floor_conditioned_transport_parameter_gradients(self):
        m=model();anchor=torch.zeros(m.dimension,dtype=torch.float64)
        for t in range(13):
            start=2+m.modes+t*2*m.modes
            anchor[start:start+m.modes]=translation(m,-.4)/.1
        guide=ConditionedTransport(m,anchor)
        z=torch.zeros(1,m.dimension,dtype=torch.float64,requires_grad=True)
        value=guide.objective(z);gradient,=torch.autograd.grad(value.sum(),z)
        for index in [0,1,m.dimension-3,m.dimension-2,m.dimension-1]:
            direction=torch.zeros_like(z);direction[:,index]=1;eps=2e-5
            finite=(guide.objective(z+eps*direction)-guide.objective(z-eps*direction))/(2*eps)
            torch.testing.assert_close(finite[0],gradient[0,index],atol=.003,rtol=3e-4)
