"""Additive finish routing: replace only the verifier entrypoint, retain deadline."""
from pathlib import Path
import importlib.util,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.surface_verify_scoped import guard_amendment


def implementation():
    name='scripts.coupled_contact_v13._postfreeze_scoped_finish'
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/coupled_contact_v13/surface_finish.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    if sum(name=='surface_verify' for name,_ in module.STAGES)!=1:raise RuntimeError('Expected exactly one original verification stage')
    module.STAGES=[('surface_verify_scoped' if name=='surface_verify' else name,args) for name,args in module.STAGES]
    original=module.implementation
    def guarded_implementation():
        guard_amendment()
        return original()
    module.implementation=guarded_implementation
    return module


def main():
    guard_amendment();implementation().main()

if __name__=='__main__':main()
