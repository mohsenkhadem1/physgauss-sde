import sys,tempfile,hashlib,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
from gaussian_operator import NeuralGreenOperator
from gaussian_modal_operator import NeuralModalGreenOperator,NeuralModalOperator
from train_gaussian_inverse import load_frozen_operator

class OperatorLoadingTests(unittest.TestCase):
    def test_modal_checkpoint_roundtrip_and_source_guard(self):
        rest=np.array([[0.,0,0],[.1,0,0],[0,.1,0],[0,0,.1]],dtype=np.float32)
        modes=np.ones((2,4,3),dtype=np.float32)*.01
        basis={'rest_points_m':rest,'modes':modes,'rbf_weights':modes.transpose(1,0,2),'width_m':.02}
        for cls in (NeuralModalGreenOperator,NeuralModalOperator):
            model=cls(basis,hidden=16,depth=2)
            hashes={name:hashlib.sha256(Path(__file__).parents[1].joinpath('experiments',name).read_bytes()).hexdigest() for name in ['gaussian_operator.py','gaussian_modal_operator.py']}
            data={'model_class':cls.__name__,'model_kwargs':model.checkpoint()['config'],
                  'model_state':model.state_dict(),'modal_checkpoint':model.checkpoint(),'source_sha256':hashes}
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'model.pt';torch.save(data,path)
                restored=load_frozen_operator(path,'cpu')
                self.assertIsInstance(restored,cls)
                self.assertFalse(any(p.requires_grad for p in restored.parameters()))
                for key,value in model.state_dict().items():
                    if isinstance(value,torch.Tensor):torch.testing.assert_close(value,restored.state_dict()[key])
                data['source_sha256']['gaussian_modal_operator.py']='tampered';torch.save(data,path)
                with self.assertRaises(ValueError):load_frozen_operator(path,'cpu')

    def test_old_model_does_not_depend_on_modal_source(self):
        model=NeuralGreenOperator(hidden=16,rank=2,depth=2)
        data={'model_class':'NeuralGreenOperator','model_kwargs':{'hidden':16,'rank':2,'depth':2},
              'model_state':model.state_dict(),'source_sha256':{'gaussian_modal_operator.py':'irrelevant'}}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt';torch.save(data,path)
            self.assertIsInstance(load_frozen_operator(path,'cpu'),NeuralGreenOperator)

if __name__=='__main__':unittest.main()
