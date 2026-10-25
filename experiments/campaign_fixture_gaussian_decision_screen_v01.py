"""Pure fixed36 DEVELOPMENT membership/observation/ranking assembly.

No model, data, RNG or GPU executes here. Matching LINEAR n6 observations are
noiseless development diagnostics, never calibrated NH or held-test evidence.
"""
import numpy as np
E_GRID=(2000.,4000.,8000.);K_GRID=(5.,float(np.sqrt(150.)),30.)

def worklist():
    # Center case first is a timing checkpoint IN the complete fixed queue.
    center=(4000.,float(np.sqrt(150.)));parameters=[center]+[(e,k)for e in E_GRID for k in K_GRID if(e,k)!=center]
    return [dict(index=i,layout=layout,E_Pa=e,K_N_m=k,n=6)for i,(layout,e,k)in enumerate((layout,e,k)for layout in range(4)for e,k in parameters)]

def prefix_observation(values):
    """The ONLY numerical observation passed into inference/action selection."""
    a=np.asarray(values,float)
    if a.shape!=(4,)or not np.isfinite(a).all():raise ValueError('Exactly four finite prefix channels; no candidate/held truth')
    return a.copy()

def summarize(rows):
    expected=worklist()
    if len(rows)!=36 or any(type(r.get('index'))is not int for r in rows)or{r['index']for r in rows}!=set(range(36)):raise ValueError('All36 unique planned rows required, including failures')
    by={r['index']:r for r in rows};winner_counts=[0]*8;tie_counts=[0]*8;unresolved=[];unique=[]
    for spec in expected:
        row=by[spec['index']]
        if row.get('case')!=spec:raise ValueError('Immutable fixed36 membership')
        rank=row.get('ranking')
        if not isinstance(rank,dict)or rank.get('qualified')is not True:
            unresolved.append(spec['index']);continue
        if rank.get('all8_retained')is not True:raise ValueError('Every candidate retained')
        ties=rank.get('numerical_ties');winner=rank.get('unique_best_action')
        if not isinstance(ties,list)or not ties or len(set(ties))!=len(ties)or any(type(i)is not int or not 0<=i<8 for i in ties):raise ValueError('Exact candidate numerical tie membership')
        if winner is None:
            if len(ties)<2:raise ValueError('Single numerical tie must be unique winner')
            for i in ties:tie_counts[i]+=1
        else:
            if type(winner)is not int or ties!=[winner]:raise ValueError('Qualified unique winner must match sole tie')
            winner_counts[winner]+=1;unique.append(winner)
    complete=not unresolved
    return dict(planned_cases=36,qualified_cases=36-len(unresolved),unresolved_cases=len(unresolved),unique_ranking_cases=len(unique),tied_cases=36-len(unresolved)-len(unique),winner_counts=winner_counts,tie_counts=tie_counts,ranking_heterogeneity_observed=len(set(unique))>1,universal_unique_ranking=bool(complete and len(unique)==36 and len(set(unique))==1),all36_retained=True,no_continuous_NH_or_active_advantage_admission=True,scope='Fixed noiseless matching-LINEAR development only; heterogeneity is a prerequisite diagnostic, never closed-loop advantage evidence.')

ADEQUACY=dict(minimum_winner_fraction=.20,minimum_distinct_winners=2,maximum_layout_probability=.90,minimum_between_layout_fraction=.10,minimum_between_layout_RMS_SD_m=2e-6,minimum_ambiguous_fraction=.25,minimum_regret_fraction_of_prior=.05,minimum_regret_case_fraction=.25)
def adequacy(rows):
    """All36 development only; indicators are not rigorous uncertainty bounds."""
    basic=summarize(rows);out=dict(basic,adequacy_gates=ADEQUACY,adequacy_qualified=False,comparison_blocked=basic['unresolved_cases']>0)
    if out['comparison_blocked']:return out
    ordered=sorted(rows,key=lambda r:r['index']);risk=np.array([[a['risk']for a in r['ranking']['actions']]for r in ordered]);indicator=np.array([[a['uncertainty']for a in r['ranking']['actions']]for r in ordered]);prior=np.array([r['posterior_screen']['prior_variance']for r in ordered]);between=np.array([r['posterior_screen']['between_layout_fraction']for r in ordered]);between_indicator=np.array([r['posterior_screen']['between_layout_fraction_indicator']for r in ordered]);maxp=np.array([r['posterior_screen']['maximum_layout_probability']for r in ordered]);tv=np.array([r['posterior_screen']['layout_TV']for r in ordered])
    if risk.shape!=(36,8)or not np.isfinite(np.concatenate([risk.ravel(),indicator.ravel(),prior,between,between_indicator,maxp,tv])).all()or np.any(prior<=0)or np.any(indicator<0)or np.any(between_indicator<0)or np.any(tv<0):raise ValueError('Complete finite positive development summaries')
    fixed=int(np.argmin(np.mean(risk,axis=0)));best=np.argmin(risk,axis=1);regret=risk[:,fixed]-risk[np.arange(36),best];regret_indicator=indicator[:,fixed]+indicator[np.arange(36),best];ambiguous=(maxp+tv<=ADEQUACY['maximum_layout_probability'])&(between-between_indicator>=ADEQUACY['minimum_between_layout_fraction'])&(np.maximum(0,between-between_indicator)*prior>=ADEQUACY['minimum_between_layout_RMS_SD_m']**2);regret_pass=(regret-regret_indicator>=ADEQUACY['minimum_regret_fraction_of_prior']*prior);counts=basic['winner_counts'];diverse=sum(c>=ADEQUACY['minimum_winner_fraction']*36 for c in counts)>=ADEQUACY['minimum_distinct_winners'];ambiguous_count=int(ambiguous.sum());regret_count=int(regret_pass.sum())
    out.update(distinct_winners_with_minimum_fraction=sum(c>=ADEQUACY['minimum_winner_fraction']*36 for c in counts),ambiguity_and_between_layout_cases=ambiguous_count,fixed_action_mean_risk_selected_from_all36_development=True,strong_fixed_regret_cases=regret_count,adequacy_qualified=bool(diverse and ambiguous_count>=ADEQUACY['minimum_ambiguous_fraction']*36 and regret_count>=ADEQUACY['minimum_regret_case_fraction']*36),indicators_are_not_rigorous_error_bounds=True)
    return out
