"""Prevent a successful job or wide intervals from bypassing the Bird gate."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from publish_pacnerf_bird_check import readiness


class BirdReadinessTests(unittest.TestCase):
    def check(self,converged=True,coverage=.85,new_crps=.05,old_crps=.06,verified=True):
        return readiness({'valid_for_ranking':converged,'settings':{'model':{'version':'floor-conditioned-hierarchical-v2'}},
            'posterior_ground_diagnostic':{'all_constrained_points_fraction_below_height':{'-0.02':0.}}},
            {'all_pass':verified,'independent_prediction_scores':{'mean_crps_m':new_crps,'coverage_curve':[{'nominal':.9,'coverage':coverage}]}},
            {'all_pass':True,'prediction_scores':{'mean_crps_m':old_crps}})

    def test_requires_convergence_and_raw_replay(self):
        self.assertTrue(self.check()['ready_for_larger_comparison'])
        self.assertFalse(self.check(converged=False)['ready_for_larger_comparison'])
        self.assertFalse(self.check(verified=False)['ready_for_larger_comparison'])

    def test_rejects_undercoverage_and_worse_proper_score(self):
        self.assertFalse(self.check(coverage=.5)['ready_for_larger_comparison'])
        self.assertFalse(self.check(coverage=1.,new_crps=.1)['ready_for_larger_comparison'])

    def test_old_unphysical_candidate_cannot_pass_without_boundary_checks(self):
        result=readiness({'valid_for_ranking':True},
            {'all_pass':True,'independent_prediction_scores':{'mean_crps_m':.05,'coverage_curve':[{'nominal':.9,'coverage':.95}]}},
            {'all_pass':True,'prediction_scores':{'mean_crps_m':.06}})
        self.assertFalse(result['ready_for_larger_comparison'])
