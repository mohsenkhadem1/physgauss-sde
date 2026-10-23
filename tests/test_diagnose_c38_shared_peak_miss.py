from experiments.diagnose_c38_shared_peak_miss import CONFIGS,diagnose,manufactured

def test_single_and_bimodal_shared_miss_are_retained_not_certified():
 for c in CONFIGS:
  r=diagnose(c)
  assert r['shared_missed_peak_demonstrated']
  assert r['maximum_reported_cross_integrator_CDF_difference']<1e-10
  assert r['maximum_true_CDF_error']>.1 and r['log_normalizer_error']>20
  assert not r['generic_helper_certified']
  assert len(r['reported_integrator_comparison']['comparisons'])==2

def test_analytic_bimodal_mass_and_prior_support():
 _,F,_=manufactured(CONFIGS[1]);assert abs(F(0))<1e-15 and abs(F(1)-1)<1e-15
 assert abs(F(.5)-.4)<1e-12
