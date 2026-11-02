from pathlib import Path

def test_only_two_damping_assignments_and_selected_source_change():
    root=Path(__file__).resolve().parents[1]
    original=(root/'scripts/compositional_dynamics_v17/native_phystwin_replay.py').read_text()
    actual=(root/'scripts/compositional_dynamics_v17_reporting/native_zero_damping_replay.py').read_text()
    expected=original.replace("SOURCE=ROOT/'scripts/compositional_dynamics_v17/native_phystwin_fit.py'","SOURCE=ROOT/'scripts/compositional_dynamics_v17/native_phystwin_budget_fit.py'")
    needle="        with (fit_directory/'optimal_params.pkl').open('rb')as f:cfg.set_optimal_params(pickle.load(f))"
    expected=expected.replace(needle,needle+"\n        cfg.drag_damping=0.0;cfg.dashpot_damping=0.0  # Sole post-hoc combined-removal intervention, no refit.")
    assert actual==expected
