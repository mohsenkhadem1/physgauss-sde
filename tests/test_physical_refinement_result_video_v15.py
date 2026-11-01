import pytest
from scripts.physical_refinement_v15_reporting.result_video import select_cases

def test_fixed_selection_retains_worst_and_lower_median():
 rows=[];worlds=[f'reserved-{i:02d}'for i in range(12)]
 for i,w in enumerate(worlds):
  for budget,error in ((0,30.),(2,float(i))):
   rows.append(dict(world=w,track='VF',repeat=0,policy='physical_risk',budget=budget,highest_weight_metrics={'normalized_design_mse':error}))
 chosen=select_cases(rows,worlds)
 assert chosen[0][1]['world']=='reserved-06'
 assert chosen[1][1]['world']=='reserved-11'
 with pytest.raises(ValueError):select_cases(rows[:-1],worlds)
 rows[-1]['highest_weight_metrics']['normalized_design_mse']=float('nan')
 with pytest.raises(ValueError):select_cases(rows,worlds)
