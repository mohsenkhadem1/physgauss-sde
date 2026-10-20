import importlib.util,tempfile,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reflected_recovery',ROOT/'experiments/recover_bird_reflected.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class PrefixTests(unittest.TestCase):
    def test_exact_prefix_and_warmup_progress_and_detect_changed_densities(self):
        rng=torch.Generator().manual_seed(15)
        new={'white':torch.randn(1,32,7,generator=rng),'log_ratio':torch.randn(1,32,generator=rng),
            'energy_errors':torch.randn(1,32,generator=rng),'protocol':{'kernel':'same'},'retained':32,'completed_warmup':500}
        old={**new,'retained':16,**{k:new[k][:,:16].clone() for k in ['white','log_ratio','energy_errors']}}
        (ROOT/'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as folder:
            before=Path(folder)/'before.pt';after=Path(folder)/'after.pt';torch.save(old,before);torch.save(new,after)
            self.assertTrue(module.prefix_proof(before,after)['prefix_bit_identical'])
            warm={'protocol':new['protocol'],'retained':0,'completed_warmup':250};torch.save(warm,before)
            self.assertEqual(module.prefix_proof(before,after)['original_prefix_draws'],0)
            torch.save(old,before);new['log_ratio'][0,0]+=.001;torch.save(new,after)
            with self.assertRaisesRegex(ValueError,'changed saved prefix'):module.prefix_proof(before,after)
