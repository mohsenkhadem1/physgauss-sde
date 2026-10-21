import json
from pathlib import Path
import tempfile
import unittest
from experiments.campaign_numerical_lowrate import analyze

class FailedGridReportingTests(unittest.TestCase):
    def test_failed_or_missing_rates_cannot_form_a_selected_plateau(self):
        config=json.loads(Path('experiments/campaign_numerical_lowrate_config.json').read_text())
        ledger={'cells':[{'condition':c['name'],'physics_hz':h,'state':'failed','exit_code':1} for c in config['coordinate_conditions'] for h in config['physics_rates_hz']]}
        with tempfile.TemporaryDirectory() as directory:
            result=analyze(directory,config,ledger)
        self.assertEqual(len(result['rows']),6)
        self.assertFalse(result['translation_gate_passed'])
        self.assertFalse(result['native_reference_qualified'])
        self.assertTrue(all(not r['lowrate_refinement_gate_passed'] for r in result['assessments']))
        self.assertTrue(all(not r['all_three_rates_available'] for r in result['assessments']))
