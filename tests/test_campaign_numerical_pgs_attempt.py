import unittest
from experiments.campaign_numerical_pgs_attempt import solver_log_evidence

class SolverEvidenceTests(unittest.TestCase):
    def test_supported_request_not_inferred_from_startup_or_unrelated_warning(self):
        logs='app startup --PGS\n[Warning] unrelated texture issue\n[Error] PGS is not supported; fallback TGS\n'
        self.assertEqual(solver_log_evidence(logs),['[Error] PGS is not supported; fallback TGS'])
        self.assertEqual(solver_log_evidence('solver=PGS\nSimulation complete'),[])
