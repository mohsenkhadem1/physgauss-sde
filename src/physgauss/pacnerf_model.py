"""Reduced elastic-graph transfer model for the official PAC-NeRF images.

Frame-zero geometry and appearance are supplied. Twelve modes of a central
spring graph approximate deformation. Linear elasticity is propagated by a
matrix exponential, with an explicit smooth ground reaction held over a frame.
This is a declared reduced model, NOT the original MPM solver or a FEM replica.
Internal noise is balanced along edges before modal projection. A full-rank
acceleration-noise floor gives a proper density. The old independently trained
network is transferred with declared amplitude scaling, never retrained on test
trajectories. Its calibration is not claimed to transfer to these objects.
"""
import math
import torch
from .calibrated_diffusion import CalibratedDiffusion
from .neural_marginal import BlockLoadElimination
from .pac_integrals import integrals as modal_integrals,covariance as integrated_covariance


def graph_modes(rest, count=12):
    n = len(rest)
    distances = torch.cdist(rest, rest)
    pairs = set()
    for i in range(n):
        for j in distances[i].argsort()[1:9].tolist(): pairs.add(tuple(sorted((i, j))))
    edges = torch.tensor(sorted(pairs))
    delta = rest[edges[:, 1]]-rest[edges[:, 0]]
    length = delta.norm(dim=-1); direction = delta/length[:, None]
    incidence = rest.new_zeros(n, len(edges))
    incidence[edges[:, 0], torch.arange(len(edges))] = -1
    incidence[edges[:, 1], torch.arange(len(edges))] = 1
    extension = torch.einsum('ne,ec->enc', incidence, direction).flatten(1)
    # Equal lumped volumes: E V/(ell^2) divided by rho V, and edge-degree
    # normalization. Frequency scale is inferred, not reported as recovered E.
    stiffness = extension.T@((100./length.square()/8)[:, None]*extension)
    eigen, vectors = torch.linalg.eigh(stiffness)
    if int((eigen < 1e-7).sum()) != 6:
        raise ValueError('Elastic graph must have exactly six infinitesimal rigid modes')
    eigen = torch.where(eigen < 1e-7, 0., eigen)
    # Eigenvector sign has no physical effect, but freeze it for portability.
    columns = vectors[:, :count]
    signs = torch.sign(columns.gather(0, columns.abs().argmax(0)[None]))[0]
    columns = columns*signs
    return edges, incidence, length, eigen[:count], columns.reshape(n, 3, count)


class PACRenderer:
    def __init__(self, data, basis):
        self.data = data
        self.gaussian_basis = torch.einsum('gn,ncm->gcm', data['weights'], basis)
        self.colors = data['gaussians'].new_full((128, 3), .4)
        self.radius, self.opacity = .045, .7
        yy, xx = torch.meshgrid(torch.arange(data['size']), torch.arange(data['size']), indexing='ij')
        self.pixels = torch.stack((xx, yy), -1).double()
        self.boundaries = torch.linspace(1., 5., 23, dtype=torch.float64)

    def components(self, q, specs, pixel_indices=None):
        centers = self.data['gaussians'][None, None]+torch.einsum('btm,gcm->btgc', q, self.gaussian_basis)
        eyes, rotations, intrinsics = [torch.stack([self.data['cameras'][c][i] for _, c in specs]) for i in range(3)]
        camera = torch.einsum('tij,btgj->btgi', rotations, centers-eyes[None, :, None])
        depth = camera[..., 2]
        safe = depth.clamp_min(.1)
        # Image coordinates represent pixel centers; upstream rays use i+.5.
        uv = camera[..., :2]/safe[..., None]*intrinsics[None, :, None, [0, 1], [0, 1]]+intrinsics[None, :, None, :2, 2]-.5
        sd = self.radius*intrinsics[None, :, None, 0, 0]/safe
        pixels = self.pixels.flatten(0,1)
        pixels = pixels[None].expand(len(specs),-1,-1) if pixel_indices is None else pixels[pixel_indices]
        delta = pixels[None,:,None]-uv[...,None,:]
        alpha = self.opacity*torch.exp(-.5*delta.square().sum(-1)/(sd.square()[...,None]+.3))
        optical = -torch.log1p(-alpha)
        cdf = .5*(1+torch.erf((self.boundaries[None, None, :, None]-depth[:, :, None])/(math.sqrt(2)*self.radius)))
        cdf = torch.cat((torch.zeros_like(cdf[:, :, :1]), cdf, torch.ones_like(cdf[:, :, :1])), 2)
        return optical, cdf[:, :, 1:]-cdf[:, :, :-1]

    def __call__(self, q, specs, colors=None, pixel_indices=None):
        optical, fractions = self.components(q, specs, pixel_indices)
        colors = self.colors if colors is None else colors
        extinction = torch.einsum('btsg,btgp->btsp', fractions, optical)
        source = torch.einsum('btsg,btgp,gc->btspc', fractions, optical, colors)
        before = extinction.cumsum(2)-extinction
        small = extinction < 1e-3
        series = 1+extinction*(-.5+extinction*(1/6+extinction*(-1/24+extinction/120)))
        safe = extinction.clamp_min(1e-3)
        phi = torch.where(small, series, -torch.expm1(-safe)/safe)
        bg = torch.stack([self.data['backgrounds'][c].flatten(0,1) for _, c in specs])
        if pixel_indices is not None: bg = torch.gather(bg,1,pixel_indices[...,None].expand(-1,-1,3))
        rgb = (torch.exp(-before)[..., None]*phi[..., None]*source).sum(2)+torch.exp(-extinction.sum(2))[..., None]*bg[None]
        return rgb.reshape(*rgb.shape[:2],self.data['size'],self.data['size'],3) if pixel_indices is None else rgb

    def calibrate(self, steps=250):
        specs = [(0, c) for c in range(11)]
        target = torch.stack([self.data['images'][c, 0].flatten(0,1) for _, c in specs])
        q = target.new_zeros(1, 11, self.gaussian_basis.shape[-1])
        with torch.no_grad():
            optical, fractions = self.components(q, specs)
            extinction = torch.einsum('btsg,btgp->btsp', fractions, optical)
            before = extinction.cumsum(2)-extinction
            phi = torch.where(extinction < 1e-3, 1-extinction/2+extinction.square()/6,
                              -torch.expm1(-extinction)/extinction.clamp_min(1e-3))
            weights = torch.einsum('btsp,btsg,btgp->btpg', torch.exp(-before)*phi, fractions, optical)[0]
            bg = torch.exp(-extinction.sum(2))[0, ..., None]*torch.stack([self.data['backgrounds'][c].flatten(0,1) for _, c in specs])
        logits = torch.zeros(128, 3, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=.08)
        for _ in range(steps):
            prediction = weights@logits.sigmoid()+bg
            loss = (prediction-target).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.colors = logits.detach().sigmoid()
        return {'static_rgb_rmse': float(loss.detach().sqrt()), 'steps': steps,
                'geometry': 'official frame-zero particles only', 'appearance_views': specs,
                'qualification': 'oracle initial geometry and multiview appearance bootstrap; no later images used'}


class PACModel:
    def __init__(self, data, calibration, noise='constant', force='marginal', frames=(0, 2, 4, 10, 13), schedule=0, mechanics=None):
        self.data, self.noise, self.force = data, noise, force
        self.mechanics = graph_modes(data['rest']) if mechanics is None else mechanics
        self.edges, self.incidence, self.length, self.eigen, self.basis = self.mechanics
        self.modes = len(self.eigen); self.frames = list(frames)
        self.specs = [(f, (slot*2+schedule)%11) for slot, f in enumerate(frames)]
        self.obs = torch.stack([data['images'][c, f] for f, c in self.specs])
        self.renderer = PACRenderer(data, self.basis)
        self.network = CalibratedDiffusion()
        self.network.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in calibration['state_dict'].items()})
        for p in self.network.parameters(): p.requires_grad_(False)
        self.constant = calibration['constant_mle_amplitude'] if 'constant_mle_amplitude' in calibration else calibration['constant_amplitude']
        self.dt = 1/24
        self.base_dimension = 2+self.modes+13*2*self.modes
        # Four temporal knots and two fixed spatial load profiles (uniform and
        # lower-rest-region), each xyz. Coefficients N(0,I) in both arms.
        knots = torch.linspace(0, 13/24, 4, dtype=torch.float64)
        temporal = torch.exp(-.5*((data['times'][:-1, None]-knots)/.14).square())
        temporal = temporal/temporal.square().sum(-1, keepdim=True).sqrt()
        lower = torch.softmax(-data['rest'][:, 1]/.1, 0)*len(data['rest'])
        profiles = torch.stack((torch.ones_like(lower), lower))
        projected = torch.einsum('pn,ncm->mcp', profiles, self.basis).reshape(self.modes, -1)
        # Modal acceleration prior, not a calibrated Newton force measurement.
        self.acceleration_mapping = torch.einsum('tk,mc->tmkc', temporal, projected).flatten(2)*2.
        self.force_dimension = self.acceleration_mapping.shape[-1]
        self.dimension = self.base_dimension+(self.force_dimension if force == 'joint' else 0)
        self.scale = torch.cat((torch.ones(2), torch.full((self.modes,), .5),
            torch.tensor([.10]*self.modes+[.5]*self.modes).repeat(13))).double()
        self.acc_noise_floor = .15
        self.internal_scale = .20

    def decode(self, z):
        qv = (z[:, 2+self.modes:self.base_dimension]*self.scale[2+self.modes:]).reshape(len(z), 13, 2*self.modes)
        zero = z.new_zeros(len(z), 1, self.modes)
        initial_v = z[:, 2:2+self.modes]*.5
        q = torch.cat((zero, qv[:, :, :self.modes]), 1)
        v = torch.cat((initial_v[:, None], qv[:, :, self.modes:]), 1)
        return q, v

    def transition(self, z, duration):
        b, m = len(z), self.modes
        omega2 = self.eigen[None]*torch.exp(z[:, :1]*.7)
        damping = torch.exp(z[:, 1:2]*.7)*.8
        argument = (damping.square()/4-omega2)*duration**2
        positive = argument.clamp_min(1e-6).sqrt()
        negative = (-argument).clamp_min(1e-6).sqrt()
        small = argument.abs() < 1e-6
        cosine = torch.where(small, 1+argument/2+argument.square()/24+argument.pow(3)/720,
            torch.where(argument >= 0, positive.cosh(), negative.cos()))
        sinch = torch.where(small, 1+argument/6+argument.square()/120+argument.pow(3)/5040,
            torch.where(argument >= 0, positive.sinh()/positive, negative.sin()/negative))
        decay = torch.exp(-damping*duration/2)
        exp = torch.stack((decay*(cosine+damping*duration*sinch/2), decay*duration*sinch,
            -decay*omega2*duration*sinch, decay*(cosine-damping*duration*sinch/2)), -1).reshape(b,m,2,2)
        # State ordering [all q, all v].
        output = z.new_zeros(b, 2*m, 2*m)
        for row in range(2):
            for col in range(2):
                output[:, row*m:(row+1)*m, col*m:(col+1)*m] = torch.diag_embed(exp[:, :, row, col])
        return output

    def integration_operators(self,z):
        return modal_integrals(self.eigen[None]*torch.exp(.7*z[:,:1]),.8*torch.exp(.7*z[:,1]),self.dt)

    def terms(self, z):
        q, v = self.decode(z); m = self.modes
        x = self.data['rest'][None, None]+torch.einsum('btm,ncm->btnc', q[:, :-1], self.basis)
        velocity = torch.einsum('btm,ncm->btnc', v[:, :-1], self.basis)
        delta = x[:, :, self.edges[:, 1]]-x[:, :, self.edges[:, 0]]
        direction = delta/(delta.square().sum(-1, keepdim=True)+1e-12).sqrt()
        strain = (delta.square().sum(-1)+1e-12).sqrt()/self.length-1
        radial = ((velocity[:, :, self.edges[:, 1]]-velocity[:, :, self.edges[:, 0]])*direction).sum(-1)
        features = torch.stack((strain, radial), -1)
        amp = self.network(features)/self.constant if self.noise == 'neural' else torch.ones_like(strain)
        edge_basis = self.basis[self.edges[:,1]]-self.basis[self.edges[:,0]]
        factor = torch.einsum('ecm,btec->btme', edge_basis, direction)*amp[:,:,None,:]
        # Same fixed normalization in both arms, defined from rest geometry.
        factor = factor*self.internal_scale/math.sqrt(len(self.edges)/len(self.data['rest']))
        acc_cov = factor@factor.transpose(-1, -2)+self.acc_noise_floor**2*torch.eye(m, dtype=z.dtype)
        transition = self.transition(z, self.dt)
        # Exact modal integrals retain all cross-mode acceleration covariance.
        # A fixed quadrature is inaccurate in high-stiffness posterior tails.
        drive,kernel=self.integration_operators(z)
        covariance=integrated_covariance(acc_cov,kernel)
        # Known gravity; smooth explicit floor reaction, held over a frame.
        nodal_acc = x.new_zeros(x.shape); nodal_acc[..., 1] = -9.8
        nodal_acc[..., 1] += 600.*.01*torch.nn.functional.softplus(-x[..., 1]/.01)
        acceleration = torch.einsum('btnc,ncm->btm', nodal_acc, self.basis)
        previous = torch.cat((q[:, :-1], v[:, :-1]), -1)
        following = torch.cat((q[:, 1:], v[:, 1:]), -1)
        residual = following-torch.einsum('bij,btj->bti', transition, previous)-torch.einsum('bij,btj->bti', drive, acceleration)
        mapping = torch.einsum('bij,tjk->btik', drive, self.acceleration_mapping)
        return residual, covariance, mapping, features

    def dynamics(self, z):
        residual, covariance, mapping, _ = self.terms(z)
        # Batched Woodbury elimination; same algebra as BlockLoadElimination,
        # but G may differ between chains' material/damping parameters.
        chol_q = torch.linalg.cholesky(covariance)
        white_g = torch.linalg.solve_triangular(chol_q, mapping, upper=False)
        white_r = torch.linalg.solve_triangular(chol_q, residual[...,None], upper=False).squeeze(-1)
        logdet_q = 2*chol_q.diagonal(dim1=-2,dim2=-1).log().sum((-1,-2))
        n = residual.shape[1]*residual.shape[2]
        if self.force == 'joint':
            coefficients = z[:,self.base_dimension:]
            white = white_r-torch.einsum('btdk,bk->btd',white_g,coefficients)
            square = white.square().sum((-1,-2))+coefficients.square().sum(-1)
            values = .5*(square+logdet_q+(n+self.force_dimension)*math.log(2*math.pi))
        else:
            precision = torch.eye(self.force_dimension,dtype=z.dtype)+torch.einsum('btdk,btdl->bkl',white_g,white_g)
            chol_a = torch.linalg.cholesky(precision)
            rhs = torch.einsum('btdk,btd->bk',white_g,white_r)
            mean = torch.cholesky_solve(rhs[...,None],chol_a).squeeze(-1)
            white = white_r-torch.einsum('btdk,bk->btd',white_g,mean)
            square = white.square().sum((-1,-2))+mean.square().sum(-1)
            logdet_a = 2*chol_a.diagonal(dim1=-2,dim2=-1).log().sum(-1)
            values = .5*(square+logdet_q+logdet_a+n*math.log(2*math.pi))
        prior = .5*z[:, :2+self.modes].square().sum(-1)
        # Triangular residual Jacobian = 1 in physical q/v coordinates. The
        # fixed coordinate scaling changes density by a constant only.
        return values+prior-self.scale[2+self.modes:].log().sum()

    def image_residual(self, z):
        q, _ = self.decode(z)
        return ((self.renderer(q[:, self.frames], self.specs)-self.obs[None])/.08).flatten(1)

    def objective(self, z):
        return self.dynamics(z)+.5*self.image_residual(z).square().sum(-1)

    def conditional(self, z):
        r, cov, g, _ = self.terms(z)
        e = BlockLoadElimination(g[0], cov)
        return e, e.conditional_mean(r.flatten(1))

    def force_conditionals(self, z):
        residual, covariance, mapping, _ = self.terms(z)
        chol_q = torch.linalg.cholesky(covariance)
        g = torch.linalg.solve_triangular(chol_q,mapping,upper=False)
        r = torch.linalg.solve_triangular(chol_q,residual[...,None],upper=False).squeeze(-1)
        p = torch.eye(self.force_dimension,dtype=z.dtype)+torch.einsum('btdk,btdl->bkl',g,g)
        chol_a = torch.linalg.cholesky(p)
        rhs = torch.einsum('btdk,btd->bk',g,r)
        mean = torch.cholesky_solve(rhs[...,None],chol_a).squeeze(-1)
        covariance = torch.cholesky_inverse(chol_a)
        normalizers = torch.stack((2*chol_q.diagonal(dim1=-2,dim2=-1).log().sum((-1,-2)),
            2*chol_a.diagonal(dim1=-2,dim2=-1).log().sum(-1)),-1)
        return mean,covariance,normalizers
