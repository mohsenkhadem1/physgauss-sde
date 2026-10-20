import copy,importlib.util,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reflected_routing',ROOT/'experiments/route_reflected_bird.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class RoutingTests(unittest.TestCase):
    def report(self):
        return {'case_index':8,'source_fingerprint':module.FINGERPRINT,'reference_sha256':module.REFERENCE,
            'source_protocol':{'reference_sha256':module.REFERENCE,'warmup':500,'interweave':True,
                'frozen_model':{'version':'floor-conditioned-hierarchical-v2'}},
            'four_independent_chains':True,'density_replay':{'all_pass':True},'draws_per_chain':2000,
            'all_monitored_quantities_pass':False,'worst_rhat':1.005,'minimum_bulk_ess':450,'minimum_tail_ess':430,
            'retained_invalid':0,'retained_divergent':0}

    def test_bounded_extensions_do_not_dispatch_a_larger_comparison(self):
        report=self.report()
        for n in [2000,4000,8000]:
            report['draws_per_chain']=n;route=module.decision(report,42)
            self.assertFalse(route['larger_comparison_dispatched'])
            if n<8000:self.assertEqual(route['inputs']['draws'],str(2*n))
            else:self.assertNotIn('workflow',route)

    def test_pass_routes_to_validation_and_inconsistent_or_foreign_reports_fail(self):
        report=self.report();report['all_monitored_quantities_pass']=True
        self.assertEqual(module.decision(report,42)['action'],'validate_uncertainty')
        for key,value in [('worst_rhat',1.02),('minimum_tail_ess',399),('retained_invalid',1),('reference_sha256','different')]:
            changed=copy.deepcopy(report);changed[key]=value
            with self.assertRaises(ValueError):module.decision(changed,42)
