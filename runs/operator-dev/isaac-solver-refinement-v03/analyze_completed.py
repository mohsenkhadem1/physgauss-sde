"""Posthoc read-only mechanics context; no simulation or parameter fitting.

Nominal-mass K/gravity and logJ-NH proxy are NOT native energy conservation.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from experiments.isaac_block_gravity_suite import mass_and_reference

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'runs/gaussian-campaign-v01/isaac-solver-refinement-v03'
PREVIOUS=ROOT/'runs/gaussian-campaign-v01/isaac-solver-ablation-v02'

def analyze():
    execution=json.loads((OUT/'execution.json').read_text())
    if execution.get('terminal') is not True or execution.get('completed') is not True:raise ValueError('All4 terminal completed required')
    rows=[];arrays={}
    for hz,it in [(480,128),(960,128),(1920,128),(3840,128),(3840,255)]:
        path=(PREVIOUS if hz==480 else OUT)/f'gravity_left_hz{hz}_it{it}'/'trajectory.npz'
        with np.load(path,allow_pickle=False) as data:
            q=data['positions_m'].astype(float);v=data['velocities_m_s'].astype(float);X=data['rest_m'];T=data['tets'];times=data['times_s'];centers=data['step_centers_m']
        arrays[(hz,it)]=(q,X)
        masses,w,_=mass_and_reference(X,T)
        native_rest=X.astype(np.float32).astype(float)
        Dm=(native_rest[T[:,1:]]-native_rest[T[:,:1]]).transpose(0,2,1);vol=np.linalg.det(Dm)/6
        Ds=(q[:,T[:,1:]]-q[:,T[:,:1]]).transpose(0,1,3,2);F=Ds@np.linalg.inv(Dm);J=np.linalg.det(F);singular=np.linalg.svd(F,compute_uv=False)
        if np.min(J)<=0:raise ValueError('Positive native-rest-relative J required')
        kinetic=.5*np.einsum('n,tnc,tnc->t',masses,v,v)
        gravity=9.81*np.einsum('n,tn->t',masses,q[:,:,2]-q[0,None,:,2])
        mu=300000/(2*1.35);lam=300000*.35/(1.35*.3)
        density=mu/2*(np.sum(F*F,axis=(-2,-1))-3)-mu*np.log(J)+lam/2*np.log(J)**2
        elastic_proxy=density@vol;proxy=kinetic+gravity+elastic_proxy-elastic_proxy[0]
        nonrigid=[];rotation=[]
        for frame in q:
            xc=w@native_rest;yc=w@frame;A=((native_rest-xc)*w[:,None]).T@(frame-yc);u,_,vh=np.linalg.svd(A);R=vh.T@u.T
            if np.linalg.det(R)<0:vh[-1]*=-1;R=vh.T@u.T
            residual=frame-((native_rest-xc)@R.T+yc)
            nonrigid.append(float(np.sqrt(np.einsum('n,nc,nc->',w,residual,residual))))
            rotation.append(float(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))*180/np.pi))
        peaks,_=find_peaks(-centers[:,2],distance=int(.15*hz),prominence=.005)
        rows.append(dict(hz=hz,iterations=it,archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            peak_best_rigid_fit_nonrigid_vector_RMS_m=max(nonrigid),final_nonrigid_vector_RMS_m=nonrigid[-1],peak_bestfit_rotation_degrees=max(rotation),
            native_rest_relative_J_range=[float(J.min()),float(J.max())],principal_stretch_range=[float(singular.min()),float(singular.max())],
            COM_down_peak_times_s=(peaks/hz).tolist(),COM_period_s=np.diff(peaks/hz).tolist(),
            times_s=times.tolist(),nominal_kinetic_J=kinetic.tolist(),nominal_gravity_potential_change_J=gravity.tolist(),
            conditional_logJ_NH_total_energy_proxy_J=proxy.tolist(),
            energy_NOT_native_conservation_test=True,native_mass_and_constitutive_energy_not_observed=True,
            first_cycle_saved_peak_kinetic_J=float(kinetic[times<.25].max()),second_cycle_saved_peak_kinetic_J=float(kinetic[times>=.25].max())))
    components=[]
    for a,b in [((960,128),(1920,128)),((1920,128),(3840,128)),((3840,128),(3840,255))]:
        delta=arrays[a][0]-arrays[b][0];distance=np.linalg.norm(delta,axis=-1);t,n=np.unravel_index(distance.argmax(),distance.shape)
        components.append(dict(a=a,b=b,time_s=float(times[t]),node=int(n),rest_world_location_m=arrays[a][1][n].tolist(),
            error_a_minus_b_world_xyz_m=delta[t,n].tolist(),norm_m=float(distance[t,n])))
    E=300000.;rho=625.;L=.24;width=height=.08;area=width*height;I=width*height**3/12
    omega=1.875104068711961**2*np.sqrt(E*I/(rho*area*L**4))
    result=dict(schema='posthoc-native-refinement-mechanics-context-v01',completed=True,no_physical_or_parameter_queries=True,
        quantities=rows,maximum_error_components=components,
        rough_linear_Euler_Bernoulli_context=dict(angular_frequency_rad_s=float(omega),frequency_Hz=float(omega/(2*np.pi)),
            gravity_static_tip_deflection_m=float(rho*area*9.81*L**4/(8*E*I)),
            NOT_matched_physics_reference='L/h3, finite~22degreebending/~23percentlocalstretch, native constitutive law unverified'),
        decay_context='Temporal pair max5.2099/3.4484/.9930mm; apparentorders .595/1.796 are notstableasymptotics. No rigorous extrapolation.',
        no_further_simulations=True)
    return result

if __name__=='__main__':
    path=Path(__file__).with_name('mechanics-context.json')
    if path.exists():raise FileExistsError('Preserve completed posthoc diagnosis')
    path.write_text(json.dumps(analyze(),indent=2,allow_nan=False)+'\n')
