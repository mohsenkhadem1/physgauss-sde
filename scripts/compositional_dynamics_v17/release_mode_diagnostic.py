"""Read-only large-release modal/phase diagnosis; no new trajectories or tuning."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize_scalar
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write
from scripts.compositional_dynamics_v17.dynamics import DynamicContactModel,VelocityVerlet
from scripts.compositional_dynamics_v17.materials import StableNeoHookean
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
PLAN=REPORT/'release-mode-diagnostic-plan.json';OUT=ROOT/'runs/compositional-dynamics-v17/release-mode-diagnostic'

def prepare():
 if PLAN.exists()or OUT.exists():raise ValueError('Preserve existing diagnostic')
 files=closure([Path(__file__)]);inputs={}
 for n in [2,3,4]:
  folder=ROOT/'runs/compositional-dynamics-v17/refinement'/f'release-n{n}-dt0.00002500-vq4-sq8'
  for name in ['complete.json','trajectory.npz']:inputs[str((folder/name).relative_to(ROOT))]=sha(folder/name)
 write(PLAN,dict(scope='Diagnostic ONLY: linearizedreference modes andexisting nonlinear20%release trajectories; no newdynamicrun, modefilter or acceptance change.',meshes=[2,3,4],eigenpairs=30,shift=-1.,modal_fit_relative_frequency_range=[.75,1.25],top_energy_modes=8,minimum_energy_fraction=1e-6,seconds=300,workers=1,source_sha256={str(f.relative_to(ROOT)):sha(f)for f in files},input_sha256=inputs))
def verify(p):
 if any(sha(ROOT/f)!=h for field in ['source_sha256','input_sha256']for f,h in p[field].items()):raise ValueError('Changed diagnostic input/source')
def reference_tangent_energy(F,z):
 law=StableNeoHookean();G=F-torch.eye(3,dtype=F.dtype,device=F.device);eps=.5*(G+G.transpose(-1,-2));return law.mu*eps.square().sum((-2,-1))+.5*law.lam*G.diagonal(dim1=-2,dim2=-1).sum(-1).square()

def fit_projection(t,signal,frequency):
 nyquist=.5/np.max(np.diff(t));cycles=frequency*(t[-1]-t[0])
 if 1.25*frequency>=nyquist:return dict(resolved=False,reason='Declared frequency search crosses sampled-displacement Nyquist',nyquist_Hz=float(nyquist),reference_cycles_in_window=float(cycles))
 def fit(f):
  A=np.c_[np.cos(2*np.pi*f*t),np.sin(2*np.pi*f*t),np.ones_like(t)];c=np.linalg.lstsq(A,signal,rcond=None)[0];return float(np.mean((A@c-signal)**2)),c
 grid=np.linspace(.75*frequency,1.25*frequency,151);scores=[fit(f)[0]for f in grid];i=int(np.argmin(scores));lo=grid[max(0,i-1)];hi=grid[min(len(grid)-1,i+1)];opt=minimize_scalar(lambda f:fit(f)[0],bounds=(lo,hi),method='bounded');f=opt.x;error,c=fit(f);referr,refc=fit(frequency)
 return dict(resolved=True,fit_bound_hit=i in (0,len(grid)-1),reference_cycles_in_window=float(cycles),fewer_than_two_reference_cycles=bool(cycles<2),nyquist_Hz=float(nyquist),observed_best_sinusoid_frequency_Hz=float(f),ratio_to_reference=float(f/frequency),amplitude=float(np.hypot(c[0],c[1])),phase_rad=float(np.arctan2(-c[1],c[0])),offset=float(c[2]),normalized_RMSE=float(np.sqrt(error)),at_reference_frequency_RMSE=float(np.sqrt(referr)),at_reference_phase_rad=float(np.arctan2(-refc[1],refc[0])),scope='Descriptive single-frequency fit to nonlinear modalprojection; no claim of stationaryfrequency or changedmechanics.')
def main():
 p=json.loads(PLAN.read_text());verify(p)
 if OUT.exists():raise ValueError('Preserve previous diagnostic')
 OUT.mkdir(parents=True);torch.set_num_threads(1);start=time.monotonic();result=dict(plan_sha256=sha(PLAN),rows=[],complete=False)
 F=torch.diag(torch.tensor([1.,1.,.8],dtype=torch.float64)).requires_grad_();law=StableNeoHookean();W=law(F,None);P=torch.autograd.grad(W,F)[0].detach().numpy();result['initial_reference_Piola_Pa']=P.tolist();result['initial_free_face_traction_norms_Pa']=np.linalg.norm(P,axis=0).tolist();result['interpretation']='Affinecompresseddisplacement isadmissible butnonzerofreefacetractions atrelease requireelasticwaves; not aninvalid weak initialcondition.'
 for n in p['meshes']:
  m=DynamicContactModel(GeometrySpec(dimensions=(.12,.06,.06)),n=n,order=4,support_mode='free',surface_order=8,bulk_energy=reference_tangent_energy);v=VelocityVerlet(m);K=m.evaluate(np.zeros_like(m.rest),tangent=True)['tangent'].tocsc();M=sparse.kron(v.M,sparse.eye(3),format='csc');lam,phi=eigsh(K,k=p['eigenpairs'],M=M,sigma=p['shift'],which='LM',tol=1e-9);order=np.argsort(lam);lam=lam[order];phi=phi[:,order]
  residual=np.linalg.norm(K@phi-(M@phi)*lam[None,:],axis=0)/(np.linalg.norm(K@phi,axis=0)+np.linalg.norm((M@phi)*lam[None,:],axis=0)+1e-12)
  positive=lam>1e-3
  orthogonality=float(np.max(abs(phi.T@(M@phi)-np.eye(len(lam)))))
  if int((~positive).sum())!=6 or np.max(abs(lam[~positive]))>1e-3 or orthogonality>1e-8 or np.max(residual[positive])>1e-6:raise ValueError('Unresolved modal eigenproblem/nullspace')
  folder=ROOT/'runs/compositional-dynamics-v17/refinement'/f'release-n{n}-dt0.00002500-vq4-sq8';status=json.loads((folder/'complete.json').read_text());data=np.load(folder/'trajectory.npz')
  if not status['completed']or sha(folder/'trajectory.npz')!=status['trajectory_sha256']:raise ValueError('Invalid source release')
  U=data['u'].reshape(len(data['times_s']),-1);A=U@(M@phi);linear=.5*U[0]@(K@U[0]);energies=.5*np.maximum(lam,0)*A[0]**2;selected=[i for i in np.argsort(energies)[::-1]if positive[i]and energies[i]>p['minimum_energy_fraction']*linear][:p['top_energy_modes']];modes=[]
  for i in selected:
   f=np.sqrt(lam[i])/(2*np.pi);modes.append(dict(index=int(i),reference_frequency_Hz=float(f),initial_modal_energy_fraction=float(energies[i]/linear),initial_mass_normalized_amplitude=float(A[0,i]),observed_fit=fit_projection(data['times_s'],A[:,i]/A[0,i],f)))
  ledger=data['ledger'];t=ledger[:,0];y=ledger[:,2]-ledger[:,2].mean();spec=np.abs(np.fft.rfft(y*np.hanning(len(y))))**2;freq=np.fft.rfftfreq(len(y),d=t[1]-t[0]);spec[0]=0;peaks=np.flatnonzero((spec[1:-1]>spec[:-2])&(spec[1:-1]>spec[2:]))+1;top=sorted(peaks,key=lambda i:spec[i],reverse=True)[:8]
  arrays=OUT/f'n{n}-modes.npz';np.savez_compressed(arrays,eigenvalues=lam,modes=phi,projections=A,times_s=data['times_s'],kinetic_spectrum_frequency_Hz=freq,kinetic_spectrum_power=spec)
  row=dict(n=n,nodes=len(m.rest),rigid_null_modes=6,rigid_eigenvalues=lam[~positive].tolist(),mass_orthogonality_max_error=orthogonality,reference_tangent_method='ExactlinearLameHessian atF=I; same stableNHreferenceHessian, avoids singular det(F-I) second-autograd derivative',positive_eigen_max_relative_residual=float(residual[positive].max()),linearized_initial_energy_J=float(linear),nonlinear_initial_energy_J=float(ledger[0,3]),captured_linear_energy_fraction=float(energies.sum()/linear),modes=modes,kinetic_energy_spectral_peaks=[dict(frequency_Hz=float(freq[i]),fraction=float(spec[i]/spec.sum()))for i in top],kinetic_spectral_power_above_500Hz_fraction=float(spec[freq>500].sum()/spec.sum()),spectrum_scope='Kineticenergyfrequency neednot equal displacementfrequency (twice forpureharmonic mode); diagnostic spectralweights notphysical modalenergy.',artifact_sha256=sha(arrays));result['rows'].append(row);write(REPORT/'release-mode-diagnostic.json',result)
 verify(p);result.update(complete=True,seconds=time.monotonic()-start);write(REPORT/'release-mode-diagnostic.json',result)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args();prepare()if a.prepare else main()
