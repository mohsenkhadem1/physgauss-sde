from scripts.compositional_dynamics_v17.cycle_refinement import analyze,task

def test_failed_and_missing_are_unqualified():
 t=task(4);r=dict(name=t['name'],completed=False,error='Synthetic failure')
 a=analyze([r],dict(gates={}),3,4)
 assert len(a['checks'])==5 and not a['passed']
 assert all(not x['passed']for x in a['checks'])
