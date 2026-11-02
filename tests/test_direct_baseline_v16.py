import torch
import pytest
from scripts.compositional_neural_mechanics.direct_baseline import DirectMechanicsBaseline,TrainingNormalization,NormalizedDirectBaseline,save_checkpoint,load_checkpoint

def pack():
 torch.manual_seed(34)
 return dict(split='train',scene_ids=['a','b'],nodefeatures=torch.randn(2,7,5),globalfeatures=torch.randn(2,4),queryfeatures=torch.randn(2,9,3),displacement=torch.randn(2,9,3)*.001,wrenches=torch.randn(2,2,6)*.02)

def test_permutation_invariance_query_equivariance_arbitrary_query_count():
 p=pack();model=DirectMechanicsBaseline(5,4,3,width=32).double()
 n,g,q=[p[k].double()for k in ('nodefeatures','globalfeatures','queryfeatures')]
 a,w=model(n,g,q);b,v=model(n[:,torch.randperm(7)],g,q)
 torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12);torch.testing.assert_close(w,v,atol=1e-12,rtol=1e-12)
 order=torch.randperm(9);c,z=model(n,g,q[:,order]);torch.testing.assert_close(c,a[:,order]);torch.testing.assert_close(z,w)
 d,_=model(n[:,:4],g,q[:,:2]);assert d.shape==(2,2,3)
 (a.square().mean()+w.square().mean()).backward();assert all(x.grad is not None for x in model.parameters())

def test_train_only_normalization_and_checkpoint(tmp_path):
 p=pack();normal=TrainingNormalization(5,4,3)
 with pytest.raises(ValueError):normal.fit({**p,'split':'test'},['a','b'])
 with pytest.raises(ValueError):normal.fit(p,['a','different'])
 normal.fit(p,['a','b'])
 with pytest.raises(ValueError):normal.fit(p,['a','b'])
 model=NormalizedDirectBaseline(DirectMechanicsBaseline(5,4,3,width=32),normal);x=[p[k]for k in ('nodefeatures','globalfeatures','queryfeatures')]
 expected=model(*x);path=tmp_path/'checkpoint.pt';save_checkpoint(path,model,{'fixture':'synthetic'},{})
 loaded,record=load_checkpoint(path);actual=loaded(*x)
 for a,b in zip(expected,actual):torch.testing.assert_close(a,b)
 assert record['training_ids']==['a','b']
 saved=normal.nodefeatures_mean.clone();_ = model(p['nodefeatures']+100,*x[1:]);torch.testing.assert_close(normal.nodefeatures_mean,saved)
