"""Read-only all150 reference inventory and numerical qualification audit."""
from pathlib import Path
import json,hashlib,collections,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13 import surface_verify
BASE=ROOT/'runs/coupled-surface-v13';REPORT=ROOT/'reports/coupled-surface-v13'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 p=read(REPORT/'protocol.json');audit=surface_verify.implementation().Audit();audit.hashes();evidence={};counts={};expected_paths=set();generation={}
 for directory,extra in [('reference',{}),('reference-refinement',{'refinement':True}),('reference-spatial',{'spatial':True})]:
  counts[directory]={'expected':50,'present':0,'successful_generation':0,'failed_generation':0}
  surface_verify.verify_settings(audit,p,directory)
  for obj in p['objects']:
   for scene in ['squeeze']+p['forecast_scenarios']:
    path=BASE/directory/obj['id']/scene;expected_paths.add(str(path/'complete.json'));file=path/'complete.json'
    if not file.exists():audit.check(False,'Missing '+str(file));continue
    status=read(file);counts[directory]['present']+=1;counts[directory]['successful_generation' if status['success'] else 'failed_generation']+=1
    expected=dict(path=str(path.relative_to(BASE)),family='fem',scenario=scene,scale=obj['young'],friction=obj['friction'],reference=True,**extra)
    audit.check(status['job']==expected,'Exact job '+str(path));generation[(directory,obj['id'],scene)]=bool(status['success'])
    for name in ('complete.json','diagnostics.json','trajectory.npz'):
     f=path/name
     if f.exists():evidence[str(f.relative_to(ROOT))]=sha(f)
 actual={str(f) for directory in counts for f in (BASE/directory).rglob('complete.json')};audit.check(actual==expected_paths,'Exact150 reference inventory with no extra cases')
 rows=[];lookup={}
 for obj in p['objects']:
  for scene in ['squeeze']+p['forecast_scenarios']:
   result=audit.reference(obj['id'],scene,p);lookup[(obj['id'],scene)]=result
   row=dict(object=obj['id'],split=obj['split'],scenario=scene,cohort='calibration' if scene=='squeeze' else 'primary' if scene in p['primary_forecast_scenarios'] else 'stress',all_three_generations_successful=all(generation.get((d,obj['id'],scene),False) for d in counts),qualification=result)
   rows.append(row);print(json.dumps(dict(object=obj['id'],scenario=scene,qualified=result['qualified'])),flush=True)
 summaries={}
 for split in ('development','reserved'):
  summaries[split]={}
  for cohort in ('calibration','primary','stress'):
   subset=[r for r in rows if r['split']==split and r['cohort']==cohort]
   summaries[split][cohort]=dict(expected=len(subset),generation_success=sum(r['all_three_generations_successful'] for r in subset),reference_qualified=sum(r['qualification']['qualified'] for r in subset),joint_with_calibration_qualified=sum(r['qualification']['qualified'] and lookup[(r['object'],'squeeze')]['qualified'] for r in subset),unqualified=[dict(object=r['object'],scenario=r['scenario'],qualification=r['qualification']) for r in subset if not r['qualification']['qualified']])
 for name in ('protocol.json','method-freeze.json','physics-development-qualification.json'):evidence[str((REPORT/name).relative_to(ROOT))]=sha(REPORT/name)
 sources=['scripts/coupled_contact_v13/reference_inventory_audit.py','scripts/coupled_contact_v13/surface_verify.py','scripts/coupled_contact_v13/verify.py','scripts/coupled_contact_v13/surface_spatial_verify.py']
 out=dict(audit_passed=not audit.errors and not audit.pending,checks=audit.checks,errors=audit.errors,pending=audit.pending,scope='All150 declared reference jobs; generation success is distinct from per-case numerical consistency. Fixed cohorts preserved; no selection filtering or new simulation.',inventory=counts,summaries=summaries,cases=rows,input_sha256=evidence,source_sha256={s:sha(ROOT/s) for s in sources},limitations=['Saved-frame arrays independently checked; integrated max penetration and minJ are logged diagnostics, not replayed substeps','Two-level temporal/spatial consistency checks, not continuum convergence proof','Sharp-edge stress remains development-unqualified regardless of individual case outcomes'])
 (REPORT/'reference-inventory-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(audit_passed=out['audit_passed'],errors=audit.errors,summary={sp:{co:{k:v for k,v in record.items() if k!='unqualified'} for co,record in vals.items()} for sp,vals in summaries.items()})),flush=True)
if __name__=='__main__':main()
