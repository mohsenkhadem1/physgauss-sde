"""Generic geometry-conditioned DeepSets supervised control, not MeshGraphNet.

Root dataset construction must supply the same declared physical descriptors and
frozen bulk/interface fingerprints available to mechanics. This baseline does
not enforce equilibrium, objectivity, contact or force/torque balance. Feature
and target normalization are fitted only on declared training scene IDs.
"""
import hashlib,json,time
from pathlib import Path
import torch
from torch import nn
FIELDS=('nodefeatures','globalfeatures','queryfeatures','displacement','wrenches')

def mlp(input_dim,output_dim,width,hidden_layers=3,zero_output=False):
 layers=[]
 for i in range(hidden_layers):layers.extend([nn.Linear(input_dim if i==0 else width,width),nn.SiLU()])
 layers.append(nn.Linear(width,output_dim))
 if zero_output:nn.init.zeros_(layers[-1].weight);nn.init.zeros_(layers[-1].bias)
 return nn.Sequential(*layers)

class DirectMechanicsBaseline(nn.Module):
 def __init__(self,node_dim,global_dim,query_dim,width=128,zero_output=False):
  super().__init__()
  if min(node_dim,global_dim,query_dim,width)<=0:raise ValueError('Positive feature dimensions required')
  self.config=dict(node_dim=node_dim,global_dim=global_dim,query_dim=query_dim,width=width,zero_output=zero_output)
  self.node_encoder=mlp(node_dim,width,width,3);self.global_encoder=mlp(global_dim,width,width,3)
  self.displacement_decoder=mlp(3*width+query_dim,3,width,3,zero_output)
  self.wrench_head=mlp(3*width,12,width,3,zero_output)
 def forward(self,nodefeatures,globalfeatures,queryfeatures):
  if nodefeatures.ndim!=3 or globalfeatures.ndim!=2 or queryfeatures.ndim!=3:raise ValueError('Expected BND,BG,BQD tensors')
  B,N,D=nodefeatures.shape
  if N==0 or queryfeatures.shape[1]==0 or globalfeatures.shape[0]!=B or queryfeatures.shape[0]!=B:raise ValueError('Nonempty consistent batch required')
  if (D,globalfeatures.shape[-1],queryfeatures.shape[-1])!=(self.config['node_dim'],self.config['global_dim'],self.config['query_dim']):raise ValueError('Feature dimensions differ from model contract')
  encoded=self.node_encoder(nodefeatures)
  pooled=torch.cat([encoded.mean(1),encoded.max(1).values,self.global_encoder(globalfeatures)],-1)
  query=torch.cat([queryfeatures,pooled[:,None].expand(-1,queryfeatures.shape[1],-1)],-1)
  return self.displacement_decoder(query),self.wrench_head(pooled).reshape(B,2,6)

class TrainingNormalization(nn.Module):
 def __init__(self,node_dim,global_dim,query_dim):
  super().__init__();self.fitted=False;self.training_ids=[]
  for key,size in zip(FIELDS,(node_dim,global_dim,query_dim,3,6)):
   self.register_buffer(key+'_mean',torch.zeros(size));self.register_buffer(key+'_std',torch.ones(size))
 def fit(self,pack,declared_training_ids):
  if self.fitted:raise ValueError('Normalization already fitted; never refit on validation/test')
  validate_pack(pack,'train',declared_training_ids,exact=True)
  for key in FIELDS:
   x=pack[key].detach().reshape(-1,pack[key].shape[-1])
   if not bool(torch.isfinite(x).all()):raise ValueError('Nonfinite training feature/target')
   mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(1e-8)
   getattr(self,key+'_mean').copy_(mean);getattr(self,key+'_std').copy_(std)
  self.training_ids=list(pack['scene_ids']);self.fitted=True
 def normalize(self,key,value):
  if not self.fitted:raise ValueError('Training normalization not fitted')
  return (value-getattr(self,key+'_mean'))/getattr(self,key+'_std')
 def restore(self,key,value):return value*getattr(self,key+'_std')+getattr(self,key+'_mean')

class NormalizedDirectBaseline(nn.Module):
 def __init__(self,model,normalization):super().__init__();self.model=model;self.normalization=normalization
 def forward(self,nodefeatures,globalfeatures,queryfeatures):
  d,w=self.model(*(self.normalization.normalize(key,value)for key,value in zip(FIELDS[:3],(nodefeatures,globalfeatures,queryfeatures))))
  return self.normalization.restore('displacement',d),self.normalization.restore('wrenches',w)

def validate_pack(pack,role,allowed_ids,exact=False):
 if pack.get('split')!=role:raise ValueError('Incorrect dataset split for this operation')
 ids=list(pack['scene_ids']);allowed=set(allowed_ids)
 if not ids or not set(ids)<=allowed or (exact and set(ids)!=allowed):raise ValueError('Scene IDs violate declared split')
 if any(not torch.is_tensor(pack[k])or pack[k].shape[0]!=len(ids)for k in FIELDS):raise ValueError('Supervised arrays must align with declared scene IDs')
 if pack['displacement'].shape!=(len(ids),pack['queryfeatures'].shape[1],3)or pack['wrenches'].shape!=(len(ids),2,6):raise ValueError('Target shape mismatch')
 return ids

def save_checkpoint(path,wrapped,feature_contract,metadata):
 path=Path(path)
 if path.exists():raise ValueError('Preserve existing direct baseline checkpoint')
 if not wrapped.normalization.fitted:raise ValueError('Cannot save unfitted normalization')
 path.parent.mkdir(parents=True,exist_ok=True)
 torch.save(dict(model_config=wrapped.model.config,state_dict=wrapped.state_dict(),training_ids=wrapped.normalization.training_ids,normalization_fitted=True,feature_contract=feature_contract,metadata=metadata,scope='Custom DeepSets supervised baseline, not authors MeshGraphNet or a physics solver.'),path)

def load_checkpoint(path,map_location='cpu'):
 saved=torch.load(path,map_location=map_location,weights_only=False)
 cfg=saved['model_config'];model=DirectMechanicsBaseline(**cfg);norm=TrainingNormalization(cfg['node_dim'],cfg['global_dim'],cfg['query_dim'])
 wrapped=NormalizedDirectBaseline(model,norm)
 # Preserve stored parameter dtype rather than silently coercing a float64 checkpoint.
 dtype=next(v.dtype for v in saved['state_dict'].values()if v.is_floating_point());wrapped=wrapped.to(dtype=dtype)
 wrapped.load_state_dict(saved['state_dict']);norm.fitted=bool(saved['normalization_fitted']);norm.training_ids=saved['training_ids'];wrapped.eval()
 return wrapped,saved

def train_direct(train,validation,*,split_manifest,feature_contract,checkpoint_path,steps=3000,batch_size=16,learning_rate=.001,device='cuda:0',seed=931301,maximum_seconds=1800):
 """Fixed-budget train-only fitting; validation reports, never selects checkpoint.

This helper deliberately has no test-pack argument. Root freezes the split and
feature contract before invoking it; matching information is a dataset audit,
not something the architecture can infer from feature names alone.
"""
 if device not in ('cpu','cuda:0','cuda:1','cuda:2'):raise ValueError('Only authorized devices allowed')
 if steps<=0 or batch_size<=0 or maximum_seconds<=0:raise ValueError('Positive fixed budget required')
 train_ids=validate_pack(train,'train',split_manifest['train'],exact=True)
 val_ids=validate_pack(validation,'validation',split_manifest['validation'],exact=True)
 if set(train_ids)&set(val_ids):raise ValueError('Training/validation scenes overlap')
 if not feature_contract.get('bulk_law_fingerprint')or not feature_contract.get('interface_law_fingerprint')or not feature_contract.get('physical_descriptor_fields'):raise ValueError('Explicit matched physical information/fingerprint contract required')
 torch.manual_seed(seed);torch.set_num_threads(1)
 dtype=train['nodefeatures'].dtype
 if not train['nodefeatures'].is_floating_point():raise ValueError('Floating supervised arrays required')
 for pack in (train,validation):
  if any(pack[k].dtype!=dtype or not bool(torch.isfinite(pack[k]).all())for k in FIELDS):raise ValueError('Finite arrays with common dtype required')
 dims=[train[k].shape[-1]for k in FIELDS[:3]];norm=TrainingNormalization(*dims).to(dtype=dtype);norm.fit(train,train_ids)
 model=DirectMechanicsBaseline(*dims,width=128).to(dtype=dtype);wrapped=NormalizedDirectBaseline(model,norm).to(device)
 data={key:norm.normalize(key,train[key].to(device))for key in FIELDS}
 optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps)
 start=time.monotonic();history=[]
 for step in range(steps):
  if time.monotonic()-start>maximum_seconds:raise TimeoutError('Direct baseline training budget exhausted; no complete checkpoint')
  ix=torch.randint(len(train_ids),(min(batch_size,len(train_ids)),),device=device)
  d,w=model(*(data[k][ix]for k in FIELDS[:3]));loss=(d-data['displacement'][ix]).square().mean()+(w-data['wrenches'][ix]).square().mean()
  optimizer.zero_grad();loss.backward();optimizer.step();scheduler.step()
  if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite supervised objective')
  if step%250==0:history.append(dict(step=step,loss=float(loss)))
 wrapped.eval()
 with torch.no_grad():
  d,w=wrapped(*(validation[k].to(device)for k in FIELDS[:3]));dv=d-validation['displacement'].to(device);wv=w-validation['wrenches'].to(device)
  val=dict(displacement_coordinate_rmse_m=float(dv.square().mean().sqrt()),force_coordinate_rmse_N=float(wv[...,:3].square().mean().sqrt()),torque_coordinate_rmse_Nm=float(wv[...,3:].square().mean().sqrt()))
 metadata=dict(steps=steps,seed=seed,seconds=time.monotonic()-start,history=history,validation=val,checkpoint_selection='Final fixed-budget iterate only',split_manifest=split_manifest,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 save_checkpoint(checkpoint_path,wrapped,feature_contract,metadata)
 return wrapped,metadata
