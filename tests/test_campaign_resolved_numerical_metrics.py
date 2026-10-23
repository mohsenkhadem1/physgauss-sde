"""Analytic polynomials/atoms/roundtrip effects only, no campaign data."""
import copy
import numpy as np
import pytest
from experiments import campaign_resolved_numerical_metrics as m


def prediction():
 u=np.zeros((9,3,3));u[0,0]=[1e-21,5e-21,9e-21];details=[]
 for point in m.POINT_IDS:
  for axis in range(3):
   continuous=point==m.POINT_IDS[0]and axis==0;qs=[]
   for p in m.PROBABILITIES:
    qs.append(dict(kind='continuous_relative_position_and_probability_qualified_quantile'if continuous else'common_constant_atom',CDF_at=dict(cdf=p,lower=p-1e-8,upper=p+1e-8)if continuous else dict(cdf=1.,lower=1.,upper=1.),CDF_left=dict(cdf=p,lower=p-1e-8,upper=p+1e-8)if continuous else dict(cdf=0.,lower=0.,upper=0.)))
   details.append(dict(point_id=int(point),axis=axis,quantiles=qs))
 row=dict(completed=True,canonical_quantile_details=details,canonical_displacement_equal_tail_m=u,canonical_position_equal_tail_m=m.POINTS[m.POINT_IDS,:,None]+u)
 return dict(methods={name:copy.deepcopy(row)for name in m.METHODS})


def test_exact_polynomial_extrema_and_shared_category_constant_mixture():
 a=np.zeros((3,189));a[:,0]=[1.,-2.,1.];a[1,1]=1e-21;b=a.copy();b[0,2]=.001
 r=m.polynomial_ranges({'a':a,'b':b});assert r['range_m'][0,0]==.25 and r['range_m'][0,2]==.001
 assert r['nonconstant'][0].tolist()==[True,True,True]and not r['nonconstant'][1:].any()
 assert r['range_m'][0,1]==pytest.approx(.5e-21,rel=1e-15)


def test_continuous_CDF_vs_atoms_and_displacement_position_zero_widths():
 p=prediction();non=np.zeros((63,3),bool);non[m.POINT_IDS[0],0]=True;r=m.quantile_diagnostics(p,dict(nonconstant=non))['adaptive']
 assert r['continuous_quantiles']==3 and r['atom_quantiles']==78 and r['unqualified_atom_jumps']==0
 assert r['maximum_continuous_CDF_error_bound']<1.1e-8
 assert r['displacement']['zero_width_nonconstant_count']==0 and r['position']['zero_width_nonconstant_continuous_bounds_count']==1
 bad=copy.deepcopy(p);row=bad['methods']['adaptive'];row['canonical_displacement_equal_tail_m'][0,0]=0
 row['canonical_quantile_details'][0]['quantiles'][0]['CDF_at']=dict(cdf=.9,lower=.899,upper=.901)
 d=m.quantile_diagnostics(bad,dict(nonconstant=non))['adaptive'];assert d['displacement']['zero_width_nonconstant_continuous_bounds_count']==1 and d['continuous_CDF_error_count_greater_than']['0.01']==1
 atom=copy.deepcopy(bad)
 for q in atom['methods']['adaptive']['canonical_quantile_details'][0]['quantiles']:q.update(kind='category_mixture_atom',CDF_left=dict(cdf=0.,lower=0.,upper=0.),CDF_at=dict(cdf=1.,lower=1.,upper=1.))
 d=m.quantile_diagnostics(atom,dict(nonconstant=non))['adaptive'];assert d['displacement']['zero_width_nonconstant_atom_bounds_count']==1 and d['continuous_quantiles']==0


def test_oracle_field_errors_and_halfwidth_ratios_no_zero_floor_or_widening():
 c=np.zeros((3,189));c[1,2]=.001;x=.75;truth=m.POINTS.copy();truth[0,2]+=.001*x
 r=m.oracle_error(c,100000.,truth);assert r['all63_position']['coordinate_RMS_m']==0
 assert r['all63_displacement']['coordinate_RMS_m']<1e-17
 p=prediction();error=np.zeros((63,3));error[m.POINT_IDS[0],0]=1e-20;error[m.POINT_IDS[1],1]=1e-12
 q=m.error_relative_halfwidth(p,dict(displacement_error_m=error));assert q['positive_halfwidth_count']==1 and q['positive_halfwidth_error_larger_than_halfwidth_count']==1 and q['zero_halfwidth_positive_oracle_error_count']==1
 assert q['maximum_finite_error_over_halfwidth']==pytest.approx(2.5)


def test_bit_group_classification_is_no_tolerance_and_missing_quantiles_retained():
 p=dict(passed=False,bit_mismatch_paths=['GL128/canonical_mean_positions_m','adaptive/material_equal_tail_Pa/0']);r=m.core_group_status(p)
 assert not r['all_core_bit_identical']and not r['material']and not r['mean']and r['variance']
 d=m.quantile_diagnostics({'methods':{}},dict(nonconstant=np.zeros((63,3),bool)))
 assert all(x['missing_quantiles']==81 and x['displacement']['missing_coordinates']==27 for x in d.values())
