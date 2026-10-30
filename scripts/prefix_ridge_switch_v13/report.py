"""Descriptive fixed-cohort report, with no cross-dataset metric pooling."""
from pathlib import Path
import sys,math,collections
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.prefix_ridge_switch_v13.run import BASE,REPORT,read,write,sha
from scripts.prefix_ridge_switch_v13.score import METHOD,COMPARATORS
METRIC={'SoftVTBench':'coordinate_rmse_mm','PhysTwin':'manual_track_mean_mm'}
def finite(x):return isinstance(x,(int,float)) and math.isfinite(x)
def mean(values):return sum(values)/len(values) if values else None
def complete(s,key):return finite(s.get(key)) and s.get('failed_frames',0)==0 and s.get('valid_frames',0)==s.get('total_frames',0) and s.get('total_frames',0)>0

def summarize(rows,dataset):
 key=METRIC[dataset];out=dict(expected=len(rows),metric=key,methods={},paired_comparisons={},attempt_states=dict(collections.Counter(r['attempt_state'] for r in rows)),switched=sum(r['switched'] for r in rows))
 for method in (METHOD,*COMPARATORS):
  applicable=not (method=='official_phystwin' and dataset!='PhysTwin' or method=='rigid_scene_state' and dataset!='SoftVTBench')
  items=[(r,r['results'][method]['summary']) for r in rows if method in r['results']] if applicable else []
  full=[s[key] for _,s in items if complete(s,key)];partial=[s[key] for _,s in items if finite(s.get(key))]
  out['methods'][method]=dict(applicable=applicable,expected=len(rows) if applicable else 0,present=len(items),complete=len(full),complete_only_mean_mm=mean(full),finite_case_mean_mm=mean(partial),means_conditional_on_completion=len(full)!=len(rows),failed_frames=sum(s.get('failed_frames',0) for _,s in items),incomplete_names=[r['name'] for r in rows if applicable and not complete(r['results'].get(method,{}).get('summary',{}),key)])
  if method==METHOD or not applicable:continue
  pairs=[]
  for r in rows:
   a=r['results'].get(METHOD,{}).get('summary',{});b=r['results'].get(method,{}).get('summary',{})
   if complete(a,key) and complete(b,key):pairs.append(dict(name=r['name'],selected_mm=a[key],comparator_mm=b[key],gain_mm=b[key]-a[key]))
  out['paired_comparisons'][method]=dict(expected=len(rows),matched=len(pairs),mean_gain_mm=mean([v['gain_mm'] for v in pairs]),wins=sum(v['gain_mm']>1e-9 for v in pairs),losses=sum(v['gain_mm']< -1e-9 for v in pairs),ties=sum(abs(v['gain_mm'])<=1e-9 for v in pairs),pairs=pairs)
 return out

def main():
 p=read(REPORT/'protocol.json');f=read(REPORT/'method-freeze.json');scores=read(REPORT/'scores.json')
 if scores['protocol_sha256']!=sha(REPORT/'protocol.json') or f['protocol_sha256']!=sha(REPORT/'protocol.json') or scores['method_freeze_sha256']!=sha(REPORT/'method-freeze.json') or scores['attempts_sha256']!=sha(BASE/'attempts.json'):raise RuntimeError('Report inputs changed')
 for path,h in {**f['source_sha256'],**f['input_sha256'],**scores['evidence_sha256']}.items():
  if sha(ROOT/path)!=h:raise RuntimeError('Changed report evidence: '+path)
 if scores['scorer_sha256']!=sha(ROOT/'scripts/prefix_ridge_switch_v13/score.py'):raise RuntimeError('Scorer changed')
 rows=scores['cases'];lookup={r['name']:r for r in rows}
 if len(rows)!=148 or len(lookup)!=148 or any(lookup.get(c['name'],{}).get('case')!=c for c in p['cases']):raise RuntimeError('All148 exact case rows required')
 out=dict(scope='Historically inspected exploratory algorithm ablation; no new sealed generalization, physical-inference or coupled-model transfer claim',scores_sha256=sha(REPORT/'scores.json'),protocol_sha256=sha(REPORT/'protocol.json'),method_freeze_sha256=sha(REPORT/'method-freeze.json'),reporter_sha256=sha(Path(__file__)),cohorts={},overall_datasets={},all_cases_accounted=True)
 lines=[out['scope'],'Candidate: guarded nodal position ridge, fixed lambda0.1;70/30 observed-prefix validation; strict>2% improvement over reconstructed archived selected SHORT configuration. Otherwise exact archived forecast.','Original direct-support raw position_ridge remains a separate comparator; marker-to-node-to-support interpolation differs. Guarded ridge is not an elastic solve.','All148 cases retained. Dataset metrics are not pooled. Means are conditional; complete-only and finite-partial means distinguished. Positive paired gain favors ridge switch.','Official PhysTwin applies only to PhysTwin cases; no official SoftVT policy benchmark claim.']
 def block(title,subset):
  result={}
  for dataset,key in METRIC.items():
   selected=[r for r in subset if r['dataset']==dataset];value=summarize(selected,dataset);result[dataset]=value
   lines.append(f'\n{title}/{dataset}: {len(selected)} cases; {key}; terminalstates={value["attempt_states"]}; switches={value["switched"]}')
   for method,v in value['methods'].items():
    if v['applicable']:lines.append(f'{method}: present{v["present"]}/{v["expected"]}; complete{v["complete"]}; complete-only mean{v["complete_only_mean_mm"]}mm; finite-partial mean{v["finite_case_mean_mm"]}mm; failedframes{v["failed_frames"]}; incomplete{v["incomplete_names"]}')
   for method,v in value['paired_comparisons'].items():lines.append(f'switch vs {method}: matched{v["matched"]}/{v["expected"]}; gain{v["mean_gain_mm"]}mm; wins/losses/ties{v["wins"]}/{v["losses"]}/{v["ties"]}')
  return result
 for cohort in dict.fromkeys(c['cohort'] for c in p['cases']):out['cohorts'][cohort]=block(cohort,[r for r in rows if r['cohort']==cohort])
 out['overall_datasets']=block('ALL exploratory; development included',rows)
 out['independent_audit']=dict(present=(REPORT/'independent-audit.json').exists())
 if out['independent_audit']['present']:
  audit=read(REPORT/'independent-audit.json');out['independent_audit'].update(passed=audit.get('passed',False),sha256=sha(REPORT/'independent-audit.json'))
 lines.append('\nIndependent numerical audit status: '+str(out['independent_audit']))
 write(REPORT/'summary.json',out);(REPORT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
