import torch
from scripts.compositional_neural_mechanics.stable_zero_control import corrected,replacement_unit

def test_only_exact_zero_input_columns_change():
 train={k:torch.tensor([[0.,2.,0.,1.],[0.,2.,1.,2.]])for k in ('nodefeatures','globalfeatures','queryfeatures')}
 state={}
 for k in (*train,'displacement','wrenches'):
  state['normalization.'+k+'_mean']=torch.tensor([0.,2.,.5,1.5])
  state['normalization.'+k+'_std']=torch.tensor([1e-8,1e-8,.5,.5])
 state['model.weights']=torch.randn(4,4)
 original={'state_dict':state};new,changes=corrected(original,train)
 assert len(changes)==3
 for key,value in state.items():
  if key in ['normalization.'+k+'_std'for k in train]:assert torch.equal(new['state_dict'][key][1:],value[1:])
  else:assert torch.equal(new['state_dict'][key],value)
 assert all(c['index']==0 for c in changes)
 assert replacement_unit('globalfeatures',7)==.02
 assert replacement_unit('globalfeatures',9)==.3
 assert replacement_unit('globalfeatures',17)==9.81
 assert replacement_unit('globalfeatures',20)==1
