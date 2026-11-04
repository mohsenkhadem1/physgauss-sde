import numpy as np
import torch
from .supervision import causal_example,available_loss

def test_future_observations_do_not_enter_encoder():
    seq=dict(rgb=np.zeros((6,2,2,3)),tracks=np.zeros((6,3,3)),tool_positions=np.ones((6,3)))
    before,_=causal_example(seq,3);seq['rgb'][3:]=99;seq['tracks'][3:]=np.nan
    after,target=causal_example(seq,3)
    for key in before['observations']:np.testing.assert_array_equal(before['observations'][key],after['observations'][key])
    assert len(after['actions']['tool_positions'])==6 and np.isnan(target['tracks']).all()

def test_missing_labels_neither_bias_loss_nor_poison_gradients():
    x=torch.tensor([1.,10.],requires_grad=True)
    loss,terms=available_loss({'tracks':x},{'tracks':torch.tensor([0.,float('nan')])})
    loss.backward();assert loss.item()==1
    torch.testing.assert_close(x.grad,torch.tensor([2.,0.]))
