"""Additive full local-import provenance audit; frozen runners stay unchanged."""
import argparse,ast,datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/compositional-neural-mechanics-v16'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def closure(seeds):
 pending=list(seeds);seen=set()
 while pending:
  path=pending.pop().resolve()
  if path in seen:continue
  seen.add(path)
  relative=path.relative_to(ROOT);module='.'.join(relative.with_suffix('').parts);package=module.rsplit('.',1)[0]
  for node in ast.walk(ast.parse(path.read_text())):
   names=[]
   if isinstance(node,ast.Import):names=[a.name for a in node.names]
   elif isinstance(node,ast.ImportFrom):
    base=node.module or ''
    if node.level:
     parts=package.split('.');base='.'.join(parts[:len(parts)-node.level+1]+([base]if base else[]))
    names=[base]+[base+'.'+a.name for a in node.names if a.name!='*']
   for name in names:
    for prefix in (ROOT,ROOT/'src'):
     for candidate in (prefix/(name.replace('.','/')+'.py'),prefix/name.replace('.','/')/'__init__.py'):
      if candidate.is_file()and candidate not in seen:pending.append(candidate)
 return {str(p.relative_to(ROOT)):sha(p)for p in sorted(seen)}
def audit(label):
 prior_path=REPORT/'prior-preservation.json';prior=json.loads(prior_path.read_text());seeds=[ROOT/'scripts/compositional_neural_mechanics'/f for f in ('dataset_runner.py','train_direct_campaign.py')];inventory=closure(seeds)
 plans=[REPORT/'training-data-plan.json',REPORT/'direct-training-plan.json'];pinned={}
 for p in plans:pinned.update(json.loads(p.read_text())['source_sha256'])
 comparisons=[]
 for name,digest in inventory.items():
  if name in prior['sha256']:expected=prior['sha256'][name];origin='pre-campaign prior preservation'
  elif name in pinned:expected=pinned[name];origin='current pinned plan'
  else:expected=None;origin='missing baseline commitment'
  comparisons.append(dict(path=name,sha256=digest,expected_sha256=expected,origin=origin,passed=digest==expected))
 result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),label=label,scope='Additive provenance supplement; frozen source_inventory omits experiments imports. No frozen plan/source modified.',timing='Initial supplemental check occurs during dataset generation, compared against pre-campaign legacy hashes; it is not falsely described as a newly performed pre-launch audit.',prior_preservation_sha256=sha(prior_path),prior_created_utc=prior['created_utc'],plan_sha256={str(p.relative_to(ROOT)):sha(p)for p in plans},audit_source_sha256=sha(Path(__file__)),inventory_sha256=inventory,additional_not_in_frozen_inventory=[n for n in inventory if n not in pinned],comparisons=comparisons,all_passed=all(r['passed']for r in comparisons))
 target=REPORT/f'legacy-dependency-{label}.json'
 if target.exists():raise ValueError('Preserve prior audit snapshot')
 target.write_text(json.dumps(result,indent=2)+'\n')
 if not result['all_passed']:raise ValueError('Dependency preservation failure')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args();r=audit(a.label);print(len(r['comparisons']),'dependencies, extra',r['additional_not_in_frozen_inventory'],'allpassed',r['all_passed'])
