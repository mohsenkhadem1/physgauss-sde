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
                from scripts.coupled_contact_v13.surface_spatial_verify import inspect
                spatial=inspect(BASE,object_id,scenario,p,module.minimum_j)
                spatial_path=BASE/'reference'/object_id/scenario/'spatial-quality.json'
                if spatial_path.exists():
                    saved=module.read(spatial_path)
                    self.check(saved['qualified']==spatial['qualified'],object_id+'/'+scenario+' spatial quality matches')
                    for independent_key,saved_key in [('sparse','common27'),('dense','dense2678')]:
                        if independent_key in spatial and isinstance(saved.get(saved_key),dict):
                            for metric in ['world_rms_m','shape_rms_m','points']:
                                self.close(saved[saved_key][metric],spatial[independent_key][metric],object_id+'/'+scenario+' '+saved_key+' '+metric)
                    if 'body_center_rms_m' in spatial and 'body_center_coordinate_rms_m' in saved:
                        self.close(saved['body_center_coordinate_rms_m'],spatial['body_center_rms_m'],object_id+'/'+scenario+' spatial body RMS')
                result['independent_spatial_check']=spatial
                result['qualified']=bool(result['qualified'] and spatial['qualified'])
                return result
            def split(self,split,p):
                result=super().split(split,p)
                path=REPORT/f'{split}-results.json'
                if path.exists():
                    for row in module.read(path)['cases']:
                        for scene,key in [(row['scenario'],'reference_qualification_reasons'),('squeeze','calibration_qualification_reasons')]:
                            body=self.reference(row['object'],scene,p)['independent_body_check']
                            labeled=any('body' in reason.lower() and 'spatial' not in reason.lower() for reason in row.get(key,[]))
                            self.check(labeled==(not body['qualified']),row['object']+'/'+scene+' body failure reason matches independent qualification')
                            spatial=self.reference(row['object'],scene,p)['independent_spatial_check']
                            labeled=any('spatial' in reason.lower() for reason in row.get(key,[]))
                            self.check(labeled==(not spatial['qualified']),row['object']+'/'+scene+' spatial failure reason matches independent qualification')
                return result
        module.Audit=CoupledAudit
        _IMPL=module
    return _IMPL


def verify_settings(a,p,subdirectory=None):
    """Resolve provenance independently; never import the predictor resolver."""
    root=BASE/subdirectory if subdirectory else BASE
    protocol_hash=hashlib.sha256((REPORT/'protocol.json').read_bytes()).hexdigest()
    count=0
    for path in sorted(root.rglob('complete.json')):
        if subdirectory is None and 'ablation' in path.relative_to(BASE).parts:continue
        status=json.loads(path.read_text());job=status.get('job',{});label=str(path.relative_to(BASE))
        reference=bool(job.get('reference',False));scenario=job.get('scenario')
        override=p.get('reference_settings_by_scenario',{}).get(scenario,{}) if reference else {}
        cells=list(override['companion_cells'] if job.get('spatial',False) else override.get('cells',p['reference_cells']) if reference else p['coarse_cells'])
        dt=float(override.get('dt',p['dt_reference']) if reference else p['dt_coarse'])
        if reference and job.get('refinement',False):dt/=2
        vertex=job.get('family')=='vertex'
        expected=dict(cells=cells,dt=dt,duration_seconds=p['duration_seconds'],scene_source='engineered_scenarios.py',mass_operator='consistent-P1',
                      contact_formulation='vertex-nodal-mass-damping' if vertex else 'clipped-normal-refined-friction-sparse-sphere',
                      surface_spacing_m=None if vertex else p['surface_spacing_m'],
                      friction_quadrature_level=None if vertex else p['friction_quadrature_level'])
        a.check(status.get('simulation_settings')==expected,label+' independently resolved settings')
        a.check(status.get('protocol_sha256')==protocol_hash,label+' exact protocol hash')
        a.check(job.get('path')==str(path.parent.relative_to(BASE)),label+' job path matches artifact')
        if job.get('spatial',False):
            a.check(reference and not job.get('refinement',False),label+' companion is base-step reference')
            a.check(cells!=list(override.get('cells',p['reference_cells'])),label+' distinct spatial companion mesh')
        if reference:
            directory='reference-spatial' if job.get('spatial',False) else 'reference-refinement' if job.get('refinement',False) else 'reference'
            a.check(path.relative_to(BASE).parts[0]==directory,label+' reference job type matches directory')
        if reference and not job.get('spatial',False):
            a.check(all(int(n)%int(c)==0 for n,c in zip(cells,p['coarse_cells'])),label+' exact nested reference mesh')
        if status.get('success'):
            a.close(status['diagnostics']['dt'],dt,label+' actual diagnostic timestep')
            with np.load(path.parent/'trajectory.npz') as arr:
                a.check(len(arr['rest'])==int(np.prod(np.asarray(cells)+1)),label+' actual node count')
                a.check(len(arr['tets'])==6*int(np.prod(cells)),label+' actual tetrahedron count')
        count+=1
    return count


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
    from scripts.coupled_contact_v13.surface_scope_audit import validate_scope
    try:validate_scope(p,module.read(REPORT/'physics-development-qualification.json'),module.read(REPORT/'method-freeze.json'),ROOT)
    except (ValueError,KeyError,OSError) as error:a.check(False,'Scope adoption/evidence invalid: '+str(error))
    expected={BASE/d/o['id']/scene/'complete.json' for d in ['reference','reference-refinement','reference-spatial'] for o in p['objects'] for scene in ['squeeze']+p['forecast_scenarios']}
    actual={path for d in ['reference','reference-refinement','reference-spatial'] for path in (BASE/d).glob('*/*/complete.json')}
    a.check(len(expected)==150,'Exactly150 declared reference completion records')
    a.check(not actual-expected,'No undeclared reference completion records')
    for path in sorted(expected-actual):a.pending.append('Missing reference completion status: '+str(path.relative_to(BASE)))
    verify_settings(a,p)
    splits={s:a.split(s,p) for s in ('development','reserved')}
    from scripts.coupled_contact_v13.surface_scope_audit import verify_main_cohorts
    if (REPORT/'summary.json').exists():verify_main_cohorts(a,p,REPORT)
    else:a.pending.append('Main fixed-cohort summary missing')
    result=dict(passed=not a.errors and not a.pending,checks=a.checks,errors=a.errors,pending=a.pending,
                prior_preserved_files=prior,splits=splits,cases=a.case_audit,
                references=[dict(object=k[0],scenario=k[1],**v) for k,v in a.reference_cache.items()],
                scope='Independent saved-array surface-contact metrics, determinant geometry, discrete selection reconstruction and frozen-file preservation; integrated unsaved diagnostics are checked but not independently replayed. No predictor or experiment scorer invoked.',
                audited_report_sha256={name:module.digest(REPORT/name) for name in ['protocol.json','method-freeze.json','physics-development-qualification.json','summary.json','development-results.json','reserved-results.json','ablation-results.json'] if (REPORT/name).exists()},
                verifier_sha256=module.digest(Path(__file__)),reused_independent_verifier_sha256=module.digest(ROOT/'scripts/coupled_contact_v13/verify.py'))
    outfile=REPORT/('independent-audit-partial.json' if args.allow_incomplete else 'independent-audit.json')
    outfile.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','references')},indent=2))
    return 1 if a.errors or (a.pending and not args.allow_incomplete) else 0


if __name__=='__main__':sys.exit(main())
