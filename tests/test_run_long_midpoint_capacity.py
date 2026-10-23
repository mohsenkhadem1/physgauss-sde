import builtins,inspect,symtable
from experiments import run_long_midpoint_capacity as m


def test_all_new_worker_and_controller_global_references_resolve_before_GPU_startup():
    for function in [m.worker,m.launch,m.prepare,m.validate]:
        table=symtable.symtable(inspect.getsource(function),'new-midpoint-capacity','exec')
        def check(t):
            for symbol in t.get_symbols():
                if symbol.is_referenced() and symbol.is_global():assert symbol.get_name() in vars(m) or hasattr(builtins,symbol.get_name()),symbol.get_name()
            for child in t.get_children():check(child)
        check(table)


def test_dependency_closure_contains_independent_CPU_midpoint_and_full_mass_operators():
    names=m.sources()
    assert {'experiments/campaign_midpoint_capacity_controls.py','experiments/campaign_p2_consistent_transient.py','experiments/campaign_p2_smooth_midpoint.py','experiments/campaign_long_supported_block.py','experiments/campaign_c38_verified_warp_solver.py','experiments/campaign_p2_jacobian_certificate.py','experiments/campaign_owned_gpu_budget_v02.py'}<=set(names)
    assert len(names)==len(set(names)) and all(not n.startswith('/') and '..' not in n.split('/') for n in names)
