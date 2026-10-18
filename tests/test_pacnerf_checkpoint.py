"""Checkpoint scheduling must preserve the original sampler's exact stream."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from physgauss.split_hmc import SplitHMCConfig,sample_split_hmc

scripts=Path(__file__).resolve().parents[1]/'scripts'
spec=importlib.util.spec_from_file_location('pac_checkpoint',scripts/'checkpoint_pacnerf.py')
checkpoint=importlib.util.module_from_spec(spec)
sys.path.insert(0,str(scripts))
try:spec.loader.exec_module(checkpoint)
finally:sys.path.pop(0)


def target(w):
    value=-.1*w.pow(4).sum(-1)
    gradient=-.4*w.pow(3)
    return value,gradient,~torch.isfinite(value)


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def compare(self,a,b):
        for field in ['white','log_ratio','energy_errors','rng_state']:
            torch.testing.assert_close(a[field],b[field],atol=0,rtol=0)
        self.assertEqual(a['counts'],b['counts']);self.assertEqual(a['step_size'],b['step_size'])

    def test_fine_checkpoints_preserve_warmup_and_retained_stream(self):
        config=SplitHMCConfig(warmup=24,draws=150,step_size=.2,min_time=.3,max_time=.8,seed=901)
        expected=sample_split_hmc(target,3,config)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            actual=checkpoint.sample_in_chunks(target,3,config,folder,max_chunk=40)
            self.compare(expected,actual)
            saved=torch.load(Path(folder)/'chains.pt',weights_only=True);self.compare(actual,saved)
            self.assertGreater(actual['gradient_evaluations'],expected['gradient_evaluations'])
            self.assertFalse((Path(folder)/'chains.tmp.pt').exists())

    def test_fine_checkpoints_preserve_a_resumed_prefix_and_rng(self):
        config=SplitHMCConfig(warmup=24,draws=160,step_size=.2,min_time=.3,max_time=.8,seed=902)
        expected=sample_split_hmc(target,3,config)
        prefix=sample_split_hmc(target,3,replace(config,draws=64))
        continuation=replace(config,warmup=0,draws=96,step_size=prefix['step_size'])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            actual=checkpoint.sample_in_chunks(target,3,continuation,folder,max_chunk=40,resume=prefix)
            self.compare(expected,actual)
            torch.testing.assert_close(actual['white'][:,:64],prefix['white'],atol=0,rtol=0)

    def test_interrupted_warmup_preserves_adaptation_and_rng(self):
        config=SplitHMCConfig(warmup=64,draws=16,step_size=.2,min_time=.3,max_time=.8,seed=903)
        expected=sample_split_hmc(target,3,config)
        class StopProbe(Exception):pass
        def interrupt(t):
            if t==25:raise StopProbe()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            state=Path(folder)/'warmup.pt'
            with self.assertRaises(StopProbe):checkpoint.sample_with_warmup_checkpoints(target,3,config,state,checkpoint_hook=interrupt)
            self.assertEqual(torch.load(state,weights_only=True)['next_iteration'],25)
            actual=checkpoint.sample_with_warmup_checkpoints(target,3,config,state)
            self.compare(expected,actual)
            self.assertEqual(actual['gradient_evaluations'],expected['gradient_evaluations']+1)

    def test_checkpoint_at_warmup_boundary_and_config_guard(self):
        config=SplitHMCConfig(warmup=32,draws=16,step_size=.2,min_time=.3,max_time=.8,seed=904)
        expected=sample_split_hmc(target,3,config)
        class StopProbe(Exception):pass
        def interrupt(t):
            if t==config.warmup:raise StopProbe()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            state=Path(folder)/'warmup.pt'
            with self.assertRaises(StopProbe):checkpoint.sample_with_warmup_checkpoints(target,3,config,state,checkpoint_hook=interrupt)
            with self.assertRaises(ValueError):checkpoint.sample_with_warmup_checkpoints(target,3,replace(config,seed=999),state)
            actual=checkpoint.sample_with_warmup_checkpoints(target,3,config,state)
            self.compare(expected,actual)


if __name__=='__main__':unittest.main()
