import pytest
from scripts.complex_scenes_v18.world_aggregation import paired


def test_queries_are_averaged_within_world_and_failure_is_visible():
    rows=[]
    for world,values in [('a',[1,3]),('b',[5,7]),('failed',[9,9])]:
        for i,value in enumerate(values):
            rows.append(dict(world_id=world,query_index=i,method='ours',state='complete',metrics={'vector_rms_mm':value}))
            rows.append(dict(world_id=world,query_index=i,method='baseline',state='rollout_failed'if world=='failed'and i==1 else'complete',metrics={'vector_rms_mm':2}))
    r=paired(rows,'ours','baseline');assert r['included_worlds']==['a','b'] and r['excluded_worlds']==['failed']
    assert r['paired_world_differences']==[0,4] and r['mean_difference']==2
    assert r['slot_counts']['baseline']['rollout_failed']==1
    assert r['world_bootstrap_95_interval']==[0,4]
    with pytest.raises(ValueError,match='Duplicate'):paired(rows+[rows[0]],'ours','baseline')
