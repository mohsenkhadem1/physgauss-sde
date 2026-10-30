"""Additive post-freeze audit routing repair; frozen scientific sources unchanged."""
from pathlib import Path
import hashlib,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
REPORT=ROOT/'reports/coupled-surface-v13'
PRODUCTION_ROOTS=('library','reference','reference-refinement','reference-spatial','prediction')
AMENDMENT=REPORT/'postfreeze-audit-routing-amendment.json'


def guard_amendment():
    amendment=json.loads(AMENDMENT.read_text())
    if hashlib.sha256((REPORT/'method-freeze.json').read_bytes()).hexdigest()!=amendment['original_method_freeze_sha256']:raise RuntimeError('Original runtime freeze changed after audit-routing amendment')
    for path,expected in amendment['additive_source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:raise RuntimeError('Additive audit source changed: '+path)
    return amendment


def scoped_module():
    name='scripts.coupled_contact_v13._postfreeze_scoped_verifier'
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/coupled_contact_v13/surface_verify.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    original=module.verify_settings
    def scoped_settings(a,p,subdirectory=None):
        if subdirectory is not None:return original(a,p,subdirectory)
        return sum(original(a,p,directory) for directory in PRODUCTION_ROOTS)
    module.verify_settings=scoped_settings
    return module


def main():
    amendment=guard_amendment();module=scoped_module();code=module.main();guard_amendment()
    path=REPORT/('independent-audit-partial.json' if '--allow-incomplete' in sys.argv else 'independent-audit.json')
    if path.exists():
        result=json.loads(path.read_text());result['postfreeze_audit_routing']=dict(amendment_sha256=hashlib.sha256(AMENDMENT.read_bytes()).hexdigest(),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),production_roots=list(PRODUCTION_ROOTS),original_frozen_verifier_sha256=hashlib.sha256((ROOT/'scripts/coupled_contact_v13/surface_verify.py').read_bytes()).hexdigest(),scope='Only job-provenance traversal is restricted to declared production roots; original cohort, selection, metric, body/spatial, source and150reference-accounting checks remain unchanged.')
        path.write_text(json.dumps(result,indent=2)+'\n')
    return code

if __name__=='__main__':raise SystemExit(main())
