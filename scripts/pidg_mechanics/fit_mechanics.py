import os
"""Fit paper NeuralFEM attachment strengths on training reconstruction positions."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scipy.optimize import minimize
from scripts.real_transfer_v19.neural_fem import NeuralFEM,parameters
from scripts.mechanics_gaussian_bridge.tetra import TetraBinding

def vertices():
 p=(1+np.sqrt(5))/2
 v=np.array([(0,a,b*p) for a in [-1,1] for b in [-1,1]]+[(a,b*p,0) for a in [-1,1] for b in [-1,1]]+[(b*p,0,a) for a in [-1,1] for b in [-1,1]])
 return np.r_[np.zeros((1,3)),v/np.linalg.norm(v,axis=1,keepdims=True)]

def payload(g,b,times):
 from scipy.interpolate import PchipInterpolator
 c=g['centers'][b];radius=g['radii'][b];v=vertices();nodes=c+radius*v
 translation=PchipInterpolator(g['times'],g['translations'][:,b],axis=0,extrapolate=False)(np.clip(times,g['times'][0],g['times'][-1]))
 affine=PchipInterpolator(g['times'],g['affines'][:,b],axis=0,extrapolate=False)(np.clip(times,g['times'][0],g['times'][-1]))
 targets=np.einsum('tij,nj->tni',affine,nodes-c)+translation[:,None]
 anchors=np.r_[np.eye(3),-np.eye(3)];weights=np.exp(-((v[None]-anchors[:,None])**2).sum(-1)/(2*.45**2))
 return dict(nodes=nodes,controller_targets=np.repeat(targets[:,None],6,axis=1),attachment_weights=weights,attachment_active=np.ones((len(times),6)),floor=float(nodes[:,2].min()-100),observed=targets,times=times)

def main():
 torch.set_num_threads(1);out=ROOT/os.environ.get('PIDG_MECHANICS_ROOT','runs/pidg-mechanics-v01');g=np.load(out/'training-geometry.npz');records=[]
 for b in range(6):
  data=payload(g,b,g['times']);m=NeuralFEM(data,device='cuda:1',materials=1);theta,bounds=parameters(m);theta[0]=-20;theta[m.i_floor]=-20;theta[1:7]=2
  observed=m.t((data['observed']-m.origin)/m.scale);ids=np.unique(np.linspace(0,len(g['times'])-1,5,dtype=int));history=[]
  def objective(z):
   th=m.t(theta);th[1:7]=float(z[0]);th.requires_grad_();loss=th.new_zeros(())
   for frame in ids:
    x=m.predict(th,int(frame),observed[frame].detach());loss=loss+(x-observed[frame]).square().mean()/len(ids)
   loss=loss+1e-5*(th[1]-2)**2;loss.backward();value=float(loss.detach());grad=float(th.grad[1:7].sum());history.append({'log_attachment':float(z[0]),'loss':value});return value,np.array([grad])
  result=minimize(objective,[2.],jac=True,method='L-BFGS-B',bounds=[(-1.,7.)],options={'maxiter':12,'maxls':10,'ftol':1e-9});theta[1:7]=result.x[0]
  record={'body':b,'parameters':theta.tolist(),'history':history,'optimizer_success':bool(result.success),'message':str(result.message),'mesh_nodes':len(m.rest),'mesh_tets':len(m.elements),'fit_training_frames':ids.tolist()};records.append(record);(out/'mechanics-fit.json').write_text(json.dumps(records,indent=2));print(json.dumps(record),flush=True)
 # Seal fitted parameter state before evaluation camera times are consulted.
 (out/'mechanics-seal.json').write_text(json.dumps({'completed':True,'test_images_read':False,'scope':'Paper NeuralFEM equilibrium adapter; six inferred moving Gaussian attachment patches per object. No identified true force/contact or dynamic forecast claim.','bodies':6,'known_limits':'Inferred controls come from a training-only PIDG reconstruction. Body contact is not solved in this adapter. Gravity set negligible under undetermined physical scale. Spherical initialization and coarse tetrahedra are disclosed priors.'},indent=2))
if __name__=='__main__':main()
