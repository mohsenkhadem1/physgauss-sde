import argparse,importlib.util,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reflected_validation',ROOT/'experiments/validate_reflected_bird.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class UncertaintyGateTests(unittest.TestCase):
    def test_failed_fresh_raw_gate_prevents_any_future_label_load(self):
        gate={'all_monitored_quantities_pass':False,'draws_per_chain':2000,'warmup':500,
            'retained_invalid':0,'retained_divergent':0}
        fake=types.SimpleNamespace(aggregate=lambda *a:gate)
        args=argparse.Namespace(worker=ROOT,source_ref='frozen',inputs='unavailable-labels',shards='unused',output='unused',draws=2000)
        with patch.dict(sys.modules,{'aggregate_pacnerf_reflected':fake}),patch.object(module.subprocess,'check_output',return_value='frozen\n'),patch.object(module.torch,'load',side_effect=AssertionError('Future labels opened before convergence')):
            with self.assertRaisesRegex(ValueError,'future labels must remain unopened'):module.run(args)
