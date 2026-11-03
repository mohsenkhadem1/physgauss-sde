"""Net nongravitational impulse from saved velocities and known body masses.

This measures the sum of fixture/contact forces integrated over time, not their
individual decomposition. It does not infer force inputs for a predictor.
"""
import hashlib,json
from pathlib import Path
import numpy as np


def trace(folder):
    folder=Path(folder);status=json.loads((folder/'complete.json').read_text())
    if 'particle_state_sha256'in status:
        path=folder/'particle-state.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=status['particle_state_sha256']:raise ValueError('Particle state integrity')
        with np.load(path)as a:
            times=a['times_s'];velocities=a['velocities_m_s'][:,None];masses=[a['mass_kg']];gravity=[a['gravity_m_s2']]
    else:
        for name,key in [('geometry.npz','geometry_sha256'),('nodal-state.npz','nodal_state_sha256')]:
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=status[key]:raise ValueError('Nodal state integrity')
        with np.load(folder/'nodal-state.npz')as a,np.load(folder/'geometry.npz')as g:
            times=a['times_s'];velocities=a['velocities_m_s'];count=velocities.shape[1]
            masses=[g[f'nodal_mass_kg_{i}']for i in range(count)];gravity=[g[f'gravity_m_s2_{i}']for i in range(count)]
    impulses=[]
    for i,(mass,g)in enumerate(zip(masses,gravity)):
        if not np.isfinite(mass).all()or np.any(mass<=0):raise ValueError('Positive finite masses required')
        momentum=np.einsum('n,tnc->tc',mass,velocities[:,i])
        impulses.append(momentum-momentum[0]-(times-times[0])[:,None]*mass.sum()*g)
    impulse=np.stack(impulses,axis=1)
    if not np.isfinite(impulse).all():raise ValueError('Nonfinite impulse')
    return times,impulse


def compare(reference,prediction):
    t,a=trace(reference);s,b=trace(prediction)
    if a.shape!=b.shape or not np.allclose(t,s,atol=1e-9):raise ValueError('Matched complete state clock required')
    d=b-a
    return dict(net_nongravitational_impulse_rms_Ns=float(np.sqrt(np.mean(np.sum(d**2,axis=-1)))),final_impulse_vector_error_Ns_per_body=np.linalg.norm(d[-1],axis=-1).tolist())
