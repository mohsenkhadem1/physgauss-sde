import copy
import numpy as np
from scripts.prefix_ridge_switch_v13.verify_results import ridge_independent,metric_summary,Audit,check_summary,aggregation
from scripts.prefix_ridge_switch_v13.selector import ridge
from scripts.external_benchmark.evaluate import expand_prediction,score
from scripts.prefix_ridge_switch_v13.report import summarize

def fixture():
    nodes=np.array([[x,y,z] for x in [0.,.02] for y in [0.,.03] for z in [0.,.04]])
    T=12;motion=np.arange(T)[:,None,None]*np.array([.0001,.0002,0.])[None,None,:];observed=nodes[None]+motion
    return dict(dataset='SoftVTBench',nodes=nodes,observed_initial=nodes,observed=observed[:7],observed_mask=np.ones((7,8),bool),observation_matrix=np.eye(8),controller_targets=(nodes[None]+motion)[:,None],attachment_active=np.ones((T,1)),train_end=7,times=np.arange(T)/20,evaluation={'points':observed,'mask':np.ones((T,8),bool)})

def test_independent_ridge_matches_fixed_formula():
    d=fixture();d.pop('evaluation');a=ridge_independent(d);b,_=ridge(d);np.testing.assert_allclose(a,b,atol=1e-14,rtol=1e-12)

def test_independent_geometry_score_and_manual_failure():
    d=fixture();pred=np.broadcast_to(d['nodes'],(12,8,3)).copy();pred[9]=np.nan
    a=metric_summary(d,pred);b=score(d,expand_prediction(d,pred),7,12)['summary'];check=Audit();check_summary(check,b,a,'synthetic');assert not check.errors and a['failed_frames']==1
    d['dataset']='PhysTwin';d['evaluation']['manual']=d['evaluation']['points'].copy();a=metric_summary(d,pred);b=score(d,expand_prediction(d,pred),7,12)['summary'];check=Audit();check_summary(check,b,a,'manual');assert not check.errors
    pred[0]=np.nan;assert metric_summary(d,pred)['manual_track_mean_mm'] is None

def test_negative_scores_and_bad_counts_rejected():
    a=Audit();check_summary(a,{'coordinate_rmse_mm':-1.,'total_frames':2,'valid_frames':2,'failed_frames':1},{'coordinate_rmse_mm':-1.},'bad');assert any('nonnegative' in x for x in a.errors) and any('accounting' in x for x in a.errors)

def test_independent_aggregate_checks_case_counts_and_failure_identity():
    method='ridge_switch_selected';row={'name':'toy','cohort':'development','dataset':'SoftVTBench','attempt_state':'timeout','switched':False,'results':{method:{'summary':{'coordinate_rmse_mm':2.,'total_frames':3,'valid_frames':3,'failed_frames':0}},'model_selection_selected':{'summary':{'coordinate_rmse_mm':3.,'total_frames':3,'valid_frames':3,'failed_frames':0}}}}
    p={'cases':[{'cohort':'development'}]};summary={'overall_datasets':{d:summarize([row] if d=='SoftVTBench' else [],d) for d in ['SoftVTBench','PhysTwin']}};summary['cohorts']={'development':copy.deepcopy(summary['overall_datasets'])}
    a=Audit();aggregation(a,p,[row],summary);assert not a.errors
    summary['cohorts']['development']['SoftVTBench']['attempt_states']={'complete':1}
    summary['overall_datasets']['SoftVTBench']['methods'][method]['incomplete_names']=['invented']
    a=Audit();aggregation(a,p,[row],summary);assert any('attempt state' in x for x in a.errors) and any('incomplete case' in x for x in a.errors)
