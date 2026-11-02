import ast
from scripts.compositional_dynamics_v17.native_phystwin_fit import ROOT
from scripts.compositional_dynamics_v17.native_phystwin_budget_fit import configure

def test_only_training_budgets_change_native_recipe():
    def functions(path):return {x.name:x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef)}
    old=functions(ROOT/'scripts/compositional_dynamics_v17/native_phystwin_fit.py');new=functions(ROOT/'scripts/compositional_dynamics_v17/native_phystwin_budget_fit.py')
    for name in ['native_data','runtime_classes','guard']:assert ast.dump(old[name])==ast.dump(new[name])
    class Cfg:
        def load_from_yaml(self,path):self.init_spring_Y=30000.;self.spring_Y_max=100000.;self.dashpot_damping=100.;self.drag_damping=3.;self.base_lr=.001;self.iterations=200
    cfg=Cfg();alpha=configure(cfg,dict(object_masses_kg=[.00216]*125))
    assert cfg.iterations==1000 and cfg.object_radius==.04 and alpha==.00216 and cfg.base_lr==.001 and cfg.drag_damping==3.
    calls=[n for n in ast.walk(new['run'])if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='optimize']
    assert len(calls)==1 and calls[0].keywords[0].arg=='max_iter'and calls[0].keywords[0].value.value==100
