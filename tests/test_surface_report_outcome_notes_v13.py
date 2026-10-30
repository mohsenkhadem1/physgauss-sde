from scripts.coupled_contact_v13.build_surface_report_annotated import outcome_notes, add_outcome_note, BUILDER

def test_values_and_frozen_insertion_points():
    methods = {'fem':dict(complete=24,coordinate_mean_mm=5.,shape_mean_mm=.973114),
               'rigid':dict(complete=24,coordinate_mean_mm=7.,shape_mean_mm=.975289),
               'native':dict(complete=16,expected=24)}
    summary={'forecast_cohorts':{'reserved':{'primary':dict(expected=24,reference_qualified=22,methods=methods,fem_comparisons={'native':dict(pairs=16,wins=13,losses=3,ties=0,mean_gain_mm=3.)}),
        'stress':dict(expected=8,reference_qualified=0,methods={'native':dict(complete=0)})}}}
    notes=outcome_notes(summary)
    assert '0.002175 mm' in notes['shape_scope']
    assert '16/24' in notes['native_pairs'] and '16 completed matched pairs' in notes['native_pairs']
    assert '0/8 stress' in notes['stress_quality'] and '22/24 primary' in notes['stress_quality']
    counts={k:0 for k in notes}
    prefixes=['World error measures positions in the shared scene.', 'Rigid contact fits friction while suppressing deformation.', 'The drop family remains development-unqualified']
    source=BUILDER.read_text()
    for prefix in prefixes:
        assert source.count(prefix)==1
        assert len(add_outcome_note(prefix,notes,counts))>len(prefix)
    assert set(counts.values())=={1}
