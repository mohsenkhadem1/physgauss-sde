"""Independent secondary-grid selection/geometry/provenance audit; no predictor calls."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
ORIGINAL=ROOT/'runs/coupled-surface-v13';ORIGINAL_REPORT=ROOT/'reports/coupled-surface-v13'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Routes:
    """Only original reference roots may be shared, never inferred predictions."""
    def __init__(self,fine,original):self.fine=Path(fine);self.original=Path(original)
    def __truediv__(self,key):return (self.original if str(key) in ('reference','reference-refinement','reference-spatial') else self.fine)/key

def audit_module(base,report,original=ORIGINAL):
    name='scripts.coupled_contact_v13._fine_grid_independent_surface'
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/coupled_contact_v13/surface_verify.py');module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    module.BASE=Routes(base,original);module.REPORT=Path(report)
    return module.implementation()

def provenance(a,base,report,p,original,original_report,evidence):
    expected_settings=dict(cells=p['coarse_cells'],dt=p['dt_coarse'],duration_seconds=p['duration_seconds'],scene_source='engineered_scenarios.py',mass_operator='consistent-P1',contact_formulation='clipped-normal-refined-friction-sparse-sphere',surface_spacing_m=p['surface_spacing_m'],friction_quadrature_level=p['friction_quadrature_level'])
    parent_audit=read(original_report/'library-coverage-audit.json')
    a.check(parent_audit.get('passed') is True and not parent_audit.get('errors'),'original library independent audit passed')
    evidence[str(original_report/'library-coverage-audit.json')]=sha(original_report/'library-coverage-audit.json')
    for directory in ('library','prediction'):
        for file in sorted((base/directory).rglob('complete.json')):
            st=read(file);job=st['job'];label=str(file.relative_to(base))
            a.check(type(st.get('success')) is bool,label+' Boolean completion status')
            a.check(st.get('protocol_sha256')==sha(report/'protocol.json'),label+' fine protocol')
            a.check(st.get('simulation_settings')==expected_settings,label+' unchanged simulation settings')
            a.check(job['path']==str(file.parent.relative_to(base)),label+' exact job path')
            evidence[str(file)]=sha(file)
            own=st.get('artifact_sha256',{})
            a.check(({'trajectory.npz','diagnostics.json'} if st['success'] else set())<=set(own),label+' own artifact inventory')
            for name,h in own.items():
                a.check(Path(name).name==name and sha(file.parent/name)==h,label+' own artifact hash '+name)
            reuse=st.get('reuse')
            expected_reuse=job['family']=='oracle' or (directory=='library' and job['friction'] in read(original_report/'protocol.json')['friction_grid'])
            a.check(bool(reuse)==expected_reuse,label+' declared reuse policy')
            if not reuse:continue
            source_job=reuse['source_job'];source=original/source_job['path'];source_status=read(source/'complete.json')
            if directory=='library':
                matched=[r for r in parent_audit['candidates'] if r['expected_job']==source_job]
                a.check(len(matched)==1,label+' unique audited parent candidate')
                if len(matched)==1:a.check(matched[0]['file_sha256']=={str((source/n).relative_to(ROOT)):h for n,h in reuse['source_file_sha256'].items()},label+' audited original file inventory')
            a.check(source_status['job']==source_job,label+' source job identity')
            for key in ('family','scenario','scale','friction','reference'):
                a.check(job[key]==source_job[key],label+' reused '+key)
            a.check(st['success']==source_status['success'],label+' reuse success unchanged')
            a.check(st.get('diagnostics')==source_status.get('diagnostics'),label+' reuse diagnostics unchanged')
            a.check(st.get('error')==source_status.get('error'),label+' reuse error unchanged')
            a.check(source_status.get('simulation_settings')==expected_settings,label+' source settings')
            a.check(reuse['source_complete_sha256']==sha(source/'complete.json'),label+' source status hash')
            a.check(reuse['parent_protocol_sha256']==sha(original_report/'protocol.json'),label+' parent protocol hash')
            a.check(reuse['parent_method_freeze_sha256']==sha(original_report/'method-freeze.json'),label+' parent freeze hash')
            required={'trajectory.npz','diagnostics.json'} if st['success'] else set()
            a.check(required<=set(reuse['source_file_sha256']),label+' reuse file inventory')
            for name,h in reuse['source_file_sha256'].items():
                a.check(Path(name).name==name,label+' safe source filename')
                a.check(sha(source/name)==h,label+' exact source file '+name)
                if name!='complete.json':a.check(sha(file.parent/name)==h,label+' exact copied file '+name)
                evidence[str(source/name)]=h

def verify(base,report,original=ORIGINAL,original_report=ORIGINAL_REPORT,allow_incomplete=False):
    base,report=Path(base),Path(report);p=read(report/'protocol.json');old=read(original_report/'protocol.json')
    module=audit_module(base,report,original);a=module.Audit();evidence={}
    # Hash checks are deliberately independent of driver guard execution.
    for folder in (original_report,report):
        freeze=read(folder/'method-freeze.json');a.check(freeze['protocol_sha256']==sha(folder/'protocol.json'),str(folder)+' protocol hash')
        for path,expected in freeze['source_sha256'].items():a.check((ROOT/path).exists() and sha(ROOT/path)==expected,'source '+path)
        evidence[str(folder/'method-freeze.json')]=sha(folder/'method-freeze.json')
    allowed={'friction_grid','scope','fine_grid_extension'}
    a.check(set(p)-set(old)<=allowed,'no undeclared new protocol keys')
    for key in set(old)-allowed:a.check(key in p and p[key]==old[key],'unchanged '+key)
    fine_freeze=read(report/'method-freeze.json')
    a.check(fine_freeze.get('parent_protocol_sha256')==sha(original_report/'protocol.json'),'fine freeze parent protocol')
    a.check(fine_freeze.get('parent_method_freeze_sha256')==sha(original_report/'method-freeze.json'),'fine freeze parent method')
    a.check(np.allclose(p['friction_grid'],np.arange(17)*.05,atol=1e-15,rtol=0),'fixed uniform 17 friction values')
    a.check(len(p['fem_young_grid'])==7 and len(p['native_scale_grid'])==7,'seven unchanged scales')
    prior=read(original_report/'prior-preservation.json')
    for name,h in prior['sha256'].items():a.check(sha(ROOT/name)==h,'preserved '+name)
    declared_library={str(Path('library')/family/f'{i}-{j}'/'complete.json') for family,scales in [('fem',p['fem_young_grid']),('native',p['native_scale_grid']),('rigid',[6000.])] for i in range(len(scales)) for j in range(17)}
    actual_library={str(q.relative_to(base)) for q in (base/'library').rglob('complete.json')}
    a.check(not (actual_library-declared_library),'no undeclared library records')
    coverage={}
    for family,scales in [('fem',p['fem_young_grid']),('native',p['native_scale_grid']),('rigid',[6000.])]:
        rows=[]
        for i,scale in enumerate(scales):
            for j,mu in enumerate(p['friction_grid']):
                q=base/'library'/family/f'{i}-{j}';file=q/'complete.json'
                if not file.exists():a.pending.append(str(file));rows.append({'scale':scale,'friction':mu,'present':False});continue
                status=read(file);job=status['job'];label=f'{family}/{i}-{j}'
                a.check(job['family']==family and job['scenario']=='squeeze' and job['scale']==scale and job['friction']==mu and not job['reference'],label+' parameter identity')
                evidence[str(file)]=sha(file)
                if status['success']:
                    arr=module.arrays(q/'trajectory.npz');a.check(all(np.isfinite(arr[k]).all() for k in ('frames','rest','times','centers','velocities','reactions')),label+' finite trajectory')
                    a.check(len(arr['rest'])==np.prod(np.array(p['coarse_cells'])+1),label+' mesh size');a.close(status['diagnostics']['dt'],p['dt_coarse'],label+' timestep')
                    evidence[str(q/'trajectory.npz')]=sha(q/'trajectory.npz')
                rows.append({'scale':scale,'friction':mu,'present':True,'success':status['success'],'failure':status.get('error')})
        coverage[family]={'expected':len(scales)*17,'present':sum(r['present'] for r in rows),'complete':sum(r.get('success',False) for r in rows),'candidates':rows}
    provenance(a,base,report,p,original,original_report,evidence)
    for obj in p['objects']:
        folder=base/'inference'/obj['id'];record=folder/'observation-reuse.json'
        if not record.exists():a.pending.append(str(record));continue
        r=read(record);source=original/'inference'/obj['id']/'observations.npz'
        a.check(r['source']==str(source.relative_to(ROOT)) and r['sha256']==sha(source)==sha(folder/'observations.npz'),'exact unchanged observation '+obj['id'])
        a.check(r['parent_protocol_sha256']==sha(original_report/'protocol.json'),'observation parent '+obj['id'])
        evidence[str(record)]=sha(record);evidence[str(source)]=sha(source)
    original_read,original_arrays=module.read,module.arrays
    def tracked_read(path):
        evidence[str(path)]=sha(path);return original_read(path)
    def tracked_arrays(path):
        evidence[str(path)]=sha(path);return original_arrays(path)
    module.read,module.arrays=tracked_read,tracked_arrays
    splits={s:a.split(s,p) for s in ('development','reserved')}
    for folder,runroot in ((report,base),(original_report,original)):
        for split in ('development','reserved'):
            path=folder/f'{split}-results.json'
            if not path.exists():continue
            document=read(path)
            for obj in p['objects']:
                if obj['split']!=split:continue
                selection=runroot/'inference'/obj['id']/'selection.json'
                if selection.exists():
                    a.check(document.get('selections',{}).get(obj['id'])==read(selection),'report selection identity '+str(selection))
                    evidence[str(selection)]=sha(selection)
    # Spatial helper owns its own reader; bind its full declared reference inputs too.
    for directory in ('reference','reference-refinement','reference-spatial'):
        for obj in p['objects']:
            for scene in ['squeeze']+p['forecast_scenarios']:
                for file in (original/directory/obj['id']/scene).glob('*'):
                    if file.is_file() and file.suffix in ('.json','.npz'):evidence[str(file)]=sha(file)
    for folder in (original_report,report):
        for name in ('protocol.json','method-freeze.json','development-results.json','reserved-results.json'):
            file=folder/name
            if file.exists():evidence[str(file)]=sha(file)
    result={'passed':not a.errors and not a.pending,'checks':a.checks,'errors':a.errors,'pending':a.pending,'splits':splits,'cases':a.case_audit,'library':coverage,'evidence_sha256':evidence,'verifier_sha256':sha(Path(__file__)),'scope':'Independent recalculation from frozen observations, all candidate losses and saved forecast arrays. References retain original spatial/body checks. Integrated unsaved substeps are not independently replayed.'}
    path=report/('independent-audit-partial.json' if allow_incomplete else 'independent-audit.json');path.write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,default=ROOT/'runs/coupled-friction-v13');parser.add_argument('--report',type=Path,default=ROOT/'reports/coupled-friction-v13');parser.add_argument('--allow-incomplete',action='store_true');args=parser.parse_args()
    r=verify(args.base,args.report,allow_incomplete=args.allow_incomplete);print(json.dumps({k:r[k] for k in ('passed','checks','errors','pending')},indent=2));return int(bool(r['errors'] or (r['pending'] and not args.allow_incomplete)))
if __name__=='__main__':raise SystemExit(main())
