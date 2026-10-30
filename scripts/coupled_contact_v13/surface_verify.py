"""Independent artifact verification for the separately frozen surface study.

Reuses geometry and saved-selection auditing in a private namespace, not any
predictor, calibration selector or experiment scorer. Exact cohort is two
existing development objects and eight newly declared reserved objects.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/coupled-surface-v13';REPORT=ROOT/'reports/coupled-surface-v13'
_IMPL=None


def implementation():
    global _IMPL
    if _IMPL is None:
        name='scripts.coupled_contact_v13._surface_isolated_verifier'
        spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/coupled_contact_v13/verify.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        module.BASE=BASE;module.REPORT=REPORT
        original_audit=module.Audit
        class CoupledAudit(original_audit):
            def reference(self,object_id,scenario,p):
                result=super().reference(object_id,scenario,p)
                if 'independent_body_check' in result:return result
                levels=[];valid=True
                for directory in ['reference','reference-refinement']:
                    path=BASE/directory/object_id/scenario
                    try:
                        status=module.read(path/'complete.json')
                        if not status['success']:valid=False;continue
                        a=module.arrays(path/'trajectory.npz');c,t=a['centers'],a['times']
                        valid=valid and c.ndim==3 and c.shape[-1]==3 and len(c)==len(t) and len(t)>1 and bool(np.all(np.diff(t)>0)) and bool(np.isfinite(c).all()) and bool(np.isfinite(t).all())
                        if scenario=='moving_sphere':valid=valid and c.shape[1]==3
                        levels.append((c,t))
                    except (OSError,KeyError,ValueError):valid=False
                error=None
                if len(levels)!=2:valid=False
                else:
                    (a,ta),(b,tb)=levels
                    matched=a.shape==b.shape and ta.shape==tb.shape and np.allclose(ta,tb,rtol=0,atol=1e-10)
                    valid=valid and matched
                    if matched and scenario=='moving_sphere' and a.ndim==3 and a.shape[1]>0:
                        error=float(np.sqrt(np.mean((a[1:,-1]-b[1:,-1])**2)))
                        valid=valid and np.isfinite(error) and error<=p.get('body_temporal_coordinate_rms_limit_m',.00025)
                result['qualified']=bool(result['qualified'] and valid)
                result['independent_body_check']=dict(qualified=bool(valid),coordinate_rms_m=error)
                return result
            def split(self,split,p):
                result=super().split(split,p)
                path=REPORT/f'{split}-results.json'
                if path.exists():
                    for row in module.read(path)['cases']:
                        for scene,key in [(row['scenario'],'reference_qualification_reasons'),('squeeze','calibration_qualification_reasons')]:
                            body=self.reference(row['object'],scene,p)['independent_body_check']
                            labeled=any('body' in reason.lower() for reason in row.get(key,[]))
                            self.check(labeled==(not body['qualified']),row['object']+'/'+scene+' body failure reason matches independent qualification')
                return result
        module.Audit=CoupledAudit
        _IMPL=module
    return _IMPL


def validate_cohort(a,p):
    objects=p['objects'];ids=[o['id'] for o in objects]
    a.check(len(objects)==10 and len(set(ids))==10,'10 unique surface-study objects')
    a.check(sum(o['split']=='development' for o in objects)==2,'exactly two development objects')
    a.check(sum(o['split']=='reserved' for o in objects)==8,'exactly eight reserved objects')
    a.check(all(o['split'] in ('development','reserved') for o in objects),'no undeclared surface-study split')
    a.check(len(p['forecast_scenarios'])==4 and len(set(p['forecast_scenarios']))==4,'four forecast families')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-incomplete',action='store_true');args=parser.parse_args()
    module=implementation();a=module.Audit();prior=a.hashes();p=module.read(REPORT/'protocol.json')
    validate_cohort(a,p)
    splits={s:a.split(s,p) for s in ('development','reserved')}
    result=dict(passed=not a.errors and not a.pending,checks=a.checks,errors=a.errors,pending=a.pending,
                prior_preserved_files=prior,splits=splits,cases=a.case_audit,
                references=[dict(object=k[0],scenario=k[1],**v) for k,v in a.reference_cache.items()],
                scope='Independent saved-array surface-contact metrics, determinant geometry, discrete selection reconstruction and frozen-file preservation; integrated unsaved diagnostics are checked but not independently replayed. No predictor or experiment scorer invoked.',
                verifier_sha256=module.digest(Path(__file__)),reused_independent_verifier_sha256=module.digest(ROOT/'scripts/coupled_contact_v13/verify.py'))
    outfile=REPORT/('independent-audit-partial.json' if args.allow_incomplete else 'independent-audit.json')
    outfile.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','references')},indent=2))
    return 1 if a.errors or (a.pending and not args.allow_incomplete) else 0


if __name__=='__main__':sys.exit(main())
