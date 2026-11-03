"""Common volumetric controls sampled from each method's saved physical state."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
from scripts.complex_scenes_v18.pair_contact import interpolation_matrix
from scripts.complex_scenes_v18.particle_tracks import binding


def controls(folder,k=5):
    folder=Path(folder);status=json.loads((folder/'complete.json').read_text())
    if not status['completed']:raise ValueError('Complete physical query required for rendering')
    q=np.stack(np.meshgrid(*[np.linspace(0,1,k)]*3,indexing='ij'),-1).reshape(-1,3)
    if 'particle_state_sha256'in status:
        path=folder/'particle-state.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=status['particle_state_sha256']:raise ValueError('Changed particle state')
        with np.load(path)as a:
            index,weight=binding(q);x=(a['positions_m'][:,index]*weight[None,...,None]).sum(2)[:,None];times=a['times_s']
    else:
        for name,key in [('geometry.npz','geometry_sha256'),('nodal-state.npz','nodal_state_sha256')]:
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=status[key]:raise ValueError('Changed nodal state')
        with np.load(folder/'geometry.npz')as g,np.load(folder/'nodal-state.npz')as a:
            samples=[];times=a['times_s']
            for body in range(a['positions_m'].shape[1]):
                elements=g[f'elements_{body}'];n=round((len(elements)/4)**(1/3))
                if 4*n**3!=len(elements):raise ValueError('Expected declared Q2 topology')
                model=SimpleNamespace(elements=elements,cells=(4*n,n,n),axes=(np.linspace(0,.24,4*n+1),np.linspace(-.04,.04,n+1),np.linspace(-.04,.04,n+1)),rest=g[f'rest_{body}'])
                B=interpolation_matrix(model,q).toarray()
                expected=q*np.array([.12,.06,.06])+[0,-.03,-.03]
                np.testing.assert_allclose(B@model.rest,expected,atol=1e-10)
                samples.append(np.einsum('pn,tnc->tpc',B,a['positions_m'][:,body]))
            x=np.stack(samples,axis=1)
    x=np.real_if_close(x)
    if np.iscomplexobj(x)or not np.isfinite(x).all():raise ValueError('Invalid physical control coordinates')
    return times,x
