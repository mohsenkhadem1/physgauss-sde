"""Freeze prefix appearance, infer a voxel volume and project observed motion."""
import sys,argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
 import torch,numpy as np
 from scipy.sparse import coo_matrix,eye,diags
 from scipy.sparse.linalg import factorized
 from scripts.pidg_solver_forecast.common import BASE,DATA,load,positions,appearance_at
 from scripts.pidg_solver_forecast.geometry import volume,separated_volume
 torch.set_num_threads(2);g,d=load(a.scene);out=BASE/a.scene/'solver';out.mkdir(exist_ok=True);frames=json.loads((DATA/a.scene/'fit.json').read_text())['frames'];t0=frames[0]['time'];end=frames[-1]['time'];times=np.linspace(t0,end,121)
 with torch.no_grad():appearance=appearance_at(g,d,t0)
 centers=appearance['centers'].cpu().numpy();geom=separated_volume(centers,appearance['opacity'].cpu().numpy(),appearance['features'].cpu().numpy(),6) if a.scene=='balls_reaction' else volume(centers,appearance['opacity'].cpu().numpy());origin=np.median(centers,axis=0);length=geom['length'];rest=(geom['nodes']-origin)/length;ref=(centers-origin)/length;e=geom['elements'];ids=e[geom['bind_tet']];N=geom['bind_N'];n=len(rest);B=coo_matrix((N.ravel(),(np.repeat(np.arange(len(N)),4),ids.ravel())),shape=(len(N),n)).tocsr();w=appearance['opacity'].cpu().numpy().ravel()/np.maximum(1,np.max(abs(N),axis=1)**2);W=diags(w)
 edges=np.unique(np.sort(np.concatenate([e[:,[i,j]] for i in range(4) for j in range(i+1,4)]),1),axis=0);L=coo_matrix((np.r_[np.ones(len(edges)),-np.ones(len(edges))],(np.r_[np.arange(len(edges)),np.arange(len(edges))],np.r_[edges[:,0],edges[:,1]])),shape=(len(edges),n)).tocsr();solve=factorized((B.T@W@B+.02*(L.T@L)+1e-7*eye(n)).tocsc());dm=(rest[e[:,1:]]-rest[e[:,:1]]).transpose(0,2,1);inv=np.linalg.inv(dm);grad=np.concatenate([-inv.sum(1,keepdims=True),inv],1);xs=[];shrink=[];fiterrors=[]
 for t in times:
  with torch.no_grad():target=(positions(g,d,float(t)).cpu().numpy()-origin)/length
  u=solve(B.T@(w[:,None]*(target-ref)));x=rest+u
  weights=w/w.sum();c0=(ref*weights[:,None]).sum(0);c1=(target*weights[:,None]).sum(0);U,_,V=np.linalg.svd(((ref-c0)*weights[:,None]).T@(target-c1));R=U@V
  if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@V
  rigid=(rest-c0)@R+c1;res=x-rigid;alpha=1.
  while np.linalg.det(np.einsum('eai,eaj->eij',(rigid+alpha*res)[e],grad)).min()<.2 and alpha>1e-5:alpha*=.5
  x=rigid+alpha*res;xs.append(x);shrink.append(alpha);fiterrors.append(float(np.sqrt(np.mean(np.sum((ref+B@(x-rest)-target)**2,axis=1)))))
 from scipy.signal import savgol_filter
 xs=np.asarray(xs);dt=times[1]-times[0];vs=savgol_filter(xs,9,3,deriv=1,delta=dt,axis=0)
 np.savez_compressed(out/'adapter.npz',rest=rest,elements=e,ids=ids,N=N,origin=origin,length=length,component_labels=geom['component_labels'],times=times,x=xs,v=vs)
 torch.save({k:v.cpu() for k,v in appearance.items()},out/'appearance.pt')
 report=dict(completed=True,segmentation=('Six visible instances declared from first fitting RGB; deterministic spatial/colour clustering, no future labels.' if a.scene=='balls_reaction' else 'Occupancy-connected components'),instance_details=geom.get('instance_details'),scene=a.scene,nodes=n,elements=len(e),gaussians=len(N),components=geom['components'],extrapolated_fraction=geom['extrapolated_fraction'],frame_times=times.tolist(),projection_position_rms_normalized=fiterrors,orientation_shrink=shrink,appearance_reference_time=t0,last_observed_time=end,units='Reconstruction length divided by object extent; normalized sequence time. No calibrated SI scale or mass.',geometry='Occupied voxels from high-opacity prefix-reconstructed Gaussians; hole fill, disconnected components preserved. Fitting-only released monocular depth enters native reconstruction where enabled, not metric ground truth.',future_frames_read=False,appearance_sha256=hashlib.sha256((out/'appearance.pt').read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256((out/'adapter.npz').read_bytes()).hexdigest())
 (out/'adapter.json').write_text(json.dumps(report,indent=2));print({k:report[k] for k in ['scene','nodes','elements','gaussians','components','extrapolated_fraction']})
if __name__=='__main__':main()
