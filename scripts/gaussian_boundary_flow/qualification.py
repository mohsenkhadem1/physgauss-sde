"""Independent discretization and observability diagnostics, never training inputs."""
from pathlib import Path
from itertools import permutations,combinations
import json
import torch
from physgauss.physics import Mesh,material_response,deformation
from physgauss.gaussian_boundary import surface_quadrature
from scripts.gaussian_boundary_flow.model import Block,observe


def refined_mesh(n,dtype=torch.float64):
    a=torch.linspace(-.05,.05,n+1,dtype=dtype)
    rest=torch.cartesian_prod(a,a+.25,a)
    ids=torch.arange(len(rest)).reshape(n+1,n+1,n+1);cells=[]
    for i,j,k in __import__('itertools').product(range(n),repeat=3):
        for order in permutations(range(3)):
            p=[i,j,k];tet=[int(ids[tuple(p)])]
            for axis in order:
                p[axis]+=1;tet.append(int(ids[tuple(p)]))
            xyz=rest[tet]
            if torch.linalg.det((xyz[1:]-xyz[:1]).T)<0:tet[1],tet[2]=tet[2],tet[1]
            cells.append(tet)
    cells=torch.tensor(cells);dm=(rest[cells[:,1:]]-rest[cells[:,:1]]).transpose(-1,-2)
    volumes=torch.linalg.det(dm)/6;mass=torch.zeros(len(rest),dtype=dtype)
    mass.index_add_(0,cells.flatten(),(500*volumes[:,None].expand(-1,4)/4).flatten())
    edges=torch.tensor(sorted({tuple(sorted(e))for c in cells.tolist()for e in combinations(c,2)}))
    return Mesh(rest,cells,torch.linalg.inv(dm),volumes,mass,rest[:,1]==.2,edges,(rest[edges[:,0]]-rest[edges[:,1]]).norm(dim=-1))


def refined_block(n):
    b=Block(dtype=torch.float64);b.mesh=refined_mesh(n);x=b.mesh.rest.flatten().requires_grad_(True)
    b.K=-torch.autograd.functional.jacobian(lambda q:material_response(q.reshape(-1,3),b.mesh,torch.tensor(1.,dtype=x.dtype))[0].flatten(),x)
    b.p,b.a,b.N=surface_quadrature(b.mesh);b.bottom=(b.p[:,1]-.2).abs()<1e-6;b.top=(b.p[:,1]-.3).abs()<1e-6
    return b


def main():
    torch.set_num_threads(2)
    out=Path('reports/gaussian-boundary-flow-v01');out.mkdir(exist_ok=True,parents=True)
    data=torch.load('runs/gaussian-boundary-flow-v01/evaluation.pt',weights_only=True);z=data['truth'][:16].double();base=Block(dtype=torch.float64)
    result={'scope':'Development diagnostics; test-derived, no model tuning from these results','temporal':[],'spatial':[]}
    reference=None
    with torch.no_grad():
        for dt,stride in ((.005,5),(.0025,10),(.00125,20)):
            u=base.rollout(z,dt=dt,stride=stride)
            result['temporal'].append({'dt_s':dt,'difference_from_previous_mm':None if reference is None else float((u-reference).square().mean().sqrt()*1000)})
            reference=u
        previous=None
        for n in (1,2,3):
            # Tangent construction requires autodiff even inside an evaluation block.
            with torch.enable_grad():b=refined_block(n)
            u=b.rollout(z[:4],dt=.00125,stride=20)
            corners=torch.cdist(base.mesh.rest,b.mesh.rest).argmin(-1);common=u[:,:,corners]
            result['spatial'].append({'subdivisions':n,'nodes':len(b.mesh.rest),'cases':4,'common_corner_difference_from_previous_mm':None if previous is None else float((common-previous).square().mean().sqrt()*1000)})
            previous=common
        u=data['trajectories']['sine'].double();F=deformation(u+base.mesh.rest,base.mesh);strain=.5*(F+F.transpose(-1,-2))-torch.eye(3,dtype=u.dtype)
        result['linear_regime']={'maximum_absolute_strain':float(strain.abs().max()),'median_case_max_absolute_strain':float(strain.abs().flatten(1).max(-1).values.median()),'minimum_J':float(torch.linalg.det(F).min()),'fraction_cases_above_10pct_strain':float((strain.abs().flatten(1).max(-1).values>.1).double().mean())}
        # Finite differences in logit coordinates, whitened by observation noise.
        zs=data['posterior'][:16].mean(1).double();J=[];Jnew=[]
        for k in range(6):
            d=torch.zeros_like(zs);d[:,k]=.001
            plus=base.rollout(zs+d);minus=base.rollout(zs-d)
            J.append((observe(plus,False)-observe(minus,False))/(.002*.00005))
            basis=plus.new_tensor([[-.5,0.,.8660254],[0.,1.,0.]])
            new=torch.einsum('btni,ki->btnk',plus[:,0:17:2]-minus[:,0:17:2],basis).flatten(1)
            Jnew.append(new/(.002*.00005))
        J=torch.stack(J,-1);Jnew=torch.stack(Jnew,-1)
        s=torch.linalg.svdvals(J);snew=torch.linalg.svdvals(torch.cat([J,Jnew],1))
        result['local_observability']={'parameterization':'six logit coordinates; linearized at inferred posterior means','cases':16,'single_view_singular_values':s.tolist(),'two_view_singular_values':snew.tolist(),'interpretation':'Local sensitivity only; not global identifiability or calibrated posterior covariance'}
    (out/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='local_observability'},indent=2))

if __name__=='__main__':main()
