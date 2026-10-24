import json,os,hashlib
from pathlib import Path
from experiments import audit_surface_p2_study_v03_failure_v01 as a
b=a.LOCAL;d=a.AUDIT
sha=a.core.sha
assert sha(b/'controlled-stop-ownership.json')=='6e8882f2b082618a9a8da688bfea7857de8b898b427c188ad6a94a8825eec36a'
assert sha(b/'stopped-completed-manifest-LOCAL.json')=='9afed9ce5e80fbaf79e5e244a12332f6d8bc7cf1fe2317eccfffe009e62b2818'
assert sha(a.__file__)=='4435ffa2ed937c7bfb300de54a0b4684b8976c1ee7190ff8f8e23bc455dfab03'
assert sha('tests/test_audit_surface_p2_study_v03_failure_v01.py')=='1be65e5f6c118e7f021e1acff29b8e1049fa1429a37fc6eaf5e90fbf7de627a7'
m=a.core.read(b/'stopped-completed-manifest-LOCAL.json');own=a.core.read(b/'controlled-stop-ownership.json')
assert m['execution_SHA256']==own['execution_SHA256']==sha(b/'execution.json')
assert m['all16_physics_retained']and m['no_completed_bytes_modified']
for name,h in m['files'].items():assert sha(b/name)==h,name
sources={name:sha(name)for name in [a.__file__,'experiments/audit_surface_p2_study_v03.py','tests/test_audit_surface_p2_study_v03_failure_v01.py',__file__]}
assert not(d/'protocol.json').exists()
recipe=dict(schema='surface-p2-study-v03-negative-audit-v01',design=a.DESIGN,root_release='Existing unchanged CPU17 v01 root GO; no original rerun',study_protocol_SHA256=a.PROTOCOL,controlled_stop_ownership_SHA256=sha(b/'controlled-stop-ownership.json'),stopped_manifest_SHA256=sha(b/'stopped-completed-manifest-LOCAL.json'),source_SHA256=sources,independent_watchdog='GNU timeout900s TERM/killafter5s',manifest_files_verified=len(m['files']),no_response_arrays_deserialized=True)
(d/'protocol.json').write_text(json.dumps(recipe,indent=2)+'\n')
a.main()
# JSON diagnostic counters only; never opens posterior or response NPZ arrays.
p=a.core.read(b/'protocol.json');cache=a.core.read(b/'cache-ready.json');actions=sum(p['design']['actions'].values(),[])
recipehash=hashlib.sha256(json.dumps(dict(numerical_engine=p['numerical_engine'],tensor=p['integration_plan'],amplitude=p['amplitude_plan']),sort_keys=True).encode()).hexdigest()
counts=dict(posterior_attempts=0,qualified_attempts=0,failed_attempts=0,prefix_attempts=0,prefix_qualified=0,prefix_failed=0,joint_attempts=0,joint_qualified=0,joint_failed=0)
for name in m['files']:
 if not name.endswith('-LOCAL.json')or not any('/'+law+'-'in name for law in a.core.LAWS):continue
 prefix=name.endswith('-prefix-LOCAL.json');joint=any('-'+action+'-r'in name for action in actions)
 if not(prefix or joint):continue
 r=a.core.read(b/name);assert r['qualified']==r['diagnostics']['qualified']
 assert r['model_fingerprint']==cache['model_fingerprint']and r['recipe_SHA256']==recipehash and r['numerical_engine']==p['numerical_engine']
 if r.get('raw_SHA256'):assert m['files'][str(Path(name).with_suffix('.npz'))]==r['raw_SHA256']
 counts['posterior_attempts']+=1;counts['qualified_attempts']+=r['qualified'];counts['failed_attempts']+=not r['qualified']
 key='prefix'if prefix else'joint';counts[key+'_attempts']+=1;counts[key+'_qualified']+=r['qualified'];counts[key+'_failed']+=not r['qualified']
assert counts==dict(posterior_attempts=49,qualified_attempts=38,failed_attempts=11,prefix_attempts=21,prefix_qualified=11,prefix_failed=10,joint_attempts=28,joint_qualified=27,joint_failed=1)
counts.update(completed=True,passed=True,manifest_files_verified=len(m['files']),all_negative_counts_preserved=True,no_response_or_posterior_arrays_deserialized=True,observation_fingerprints_not_reconstructed=True)
(d/'stopped-counts-audit.json').write_text(json.dumps(counts,indent=2)+'\n')
print(json.dumps(counts))
