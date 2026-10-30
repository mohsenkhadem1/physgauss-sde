"""Post-freeze evaluator-only rigid/deformation diagnosis, never predictor input."""
import sys,json
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.contact_benchmark_v12.common import ROOT,REPORT,write_json,sha256
from scripts.contact_benchmark_v12.data import load
from scripts.external_benchmark.data import rigid_map

MOTION_FLOOR_M=1e-6


def diagnose(points,mask,cut):
    """Same initial track identities, intersect initial/current visibility masks."""
    initial=points[0];initial_valid=mask[0]&np.isfinite(initial).all(1);frames=[]
    for t,target in enumerate(points):
        valid=initial_valid&mask[t]&np.isfinite(target).all(1);count=int(valid.sum())
        row=dict(frame=t,shared_valid_points=count,initial_valid_points=int(initial_valid.sum()),visible_points=int(mask[t].sum()))
        if count<3:
            row['excluded']='fewer than three shared initial/current points';frames.append(row);continue
        a,b=initial[valid],target[valid]
        if np.linalg.matrix_rank(a-a.mean(0),tol=1e-10)<2:
            row['excluded']='initial shared points collinear/degenerate';frames.append(row);continue
        R,offset=rigid_map(a,b);delta=b-a;residual=b-(a@R+offset)
        total=float(np.mean(np.sum(delta**2,axis=1)));remaining=float(np.mean(np.sum(residual**2,axis=1)))
        if remaining>total+1e-15:raise ValueError('Kabsch residual exceeds identity-motion error')
        row.update(total_mean_squared_displacement_m2=total,rigid_removed_mean_squared_displacement_m2=remaining,
                   total_vector_rms_mm=1000*np.sqrt(total),rigid_removed_vector_rms_mm=1000*np.sqrt(remaining),proper_rotation_determinant=float(np.linalg.det(R)))
        row['small_motion']=bool(np.sqrt(total)<MOTION_FLOOR_M)
        row['nonrigid_rms_ratio']=None if row['small_motion'] else float(np.sqrt(remaining/total))
        row['nonrigid_squared_displacement_fraction']=None if row['small_motion'] else remaining/total
        frames.append(row)
    segments={}
    for label,start,stop in [('prefix',0,cut),('suffix',cut,len(points)),('all',0,len(points))]:
        selected=[r for r in frames[start:stop] if 'excluded' not in r];n=sum(r['shared_valid_points'] for r in selected)
        if not n:segments[label]=dict(frames_expected=stop-start,frames_valid=0,shared_point_observations=0);continue
        total=sum(r['total_mean_squared_displacement_m2']*r['shared_valid_points'] for r in selected)/n
        remaining=sum(r['rigid_removed_mean_squared_displacement_m2']*r['shared_valid_points'] for r in selected)/n
        small=np.sqrt(total)<MOTION_FLOOR_M
        segments[label]=dict(frames_expected=stop-start,frames_valid=len(selected),frames_excluded=stop-start-len(selected),shared_point_observations=n,
          min_shared_points=min(r['shared_valid_points'] for r in selected),max_shared_points=max(r['shared_valid_points'] for r in selected),small_motion_frames=sum(r['small_motion'] for r in selected),
          total_vector_rms_mm=float(1000*np.sqrt(total)),rigid_removed_vector_rms_mm=float(1000*np.sqrt(remaining)),
          nonrigid_rms_ratio=None if small else float(np.sqrt(remaining/total)),nonrigid_squared_displacement_fraction=None if small else float(remaining/total),small_motion=bool(small))
    return dict(initial_points=len(initial),initial_visible_finite_points=int(initial_valid.sum()),train_end=cut,segments=segments,frames=frames)


def aggregate(rows):
    out={}
    for segment in ['prefix','suffix','all']:
        entries=[r['segments'][segment] for r in rows];valid=[e for e in entries if e.get('nonrigid_rms_ratio') is not None]
        summary=dict(sequences=len(rows),ratio_valid_sequences=len(valid),small_or_missing_motion_sequences=len(rows)-len(valid))
        for metric in ['total_vector_rms_mm','rigid_removed_vector_rms_mm','nonrigid_rms_ratio','nonrigid_squared_displacement_fraction']:
            values=[e[metric] for e in valid];summary[metric]=dict(mean=float(np.mean(values)),median=float(np.median(values)),min=float(np.min(values)),max=float(np.max(values))) if values else None
        summary['rms_ratio_below_1percent_sequences']=sum(e['nonrigid_rms_ratio']<.01 for e in valid)
        summary['rms_ratio_below_5percent_sequences']=sum(e['nonrigid_rms_ratio']<.05 for e in valid)
        out[segment]=summary
    return out


def main():
    for name in ['method-freeze.json','feasibility-freeze.json']:
        if not(REPORT/name).exists():raise RuntimeError('Diagnostic requires both method freezes first')
    protocol=json.loads((REPORT/'benchmark-protocol.json').read_text());fresh=json.loads((REPORT/'feasibility-protocol.json').read_text());cohorts={k:protocol[k] for k in ['development','regression','untouched_confirmation']};cohorts['fresh_confirmation']=fresh['fresh_confirmation']
    rows=[];hashes={};seen=set()
    for cohort,group in cohorts.items():
        for spec in group['SoftVTBench']+[dict(case=c) for c in group['PhysTwin']]:
            data=load(spec);name=data['name']
            if name in seen:raise ValueError('Repeated sequence across declared cohorts')
            seen.add(name);raw=data['source']
            if raw not in hashes:hashes[raw]=sha256(raw)
            row=diagnose(data['evaluation']['points'],data['evaluation']['mask'],data['train_end'])
            row.update(name=name,dataset=data['dataset'],cohort=cohort,group=name.split('-demo')[0] if data['dataset']=='SoftVTBench' else name,input_source=raw,input_sha256=hashes[raw]);rows.append(row)
            print(cohort,name,row['segments']['suffix'].get('nonrigid_rms_ratio'),flush=True)
    if len(rows)!=148:raise ValueError(f'Expected148 predeclared sequences, got{len(rows)}')
    groups=defaultdict(list)
    for row in rows:
        groups[f"{row['dataset']} / {row['group']}"].append(row)
        groups[f"{row['dataset']} / ALL"].append(row)
        groups[f"{row['cohort']} / {row['dataset']}"] .append(row)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Evaluator-only descriptive motion diagnosis after freeze, not a model-selection input',
      definition='Vector RMS sqrt(mean squared Euclidean displacement); Kabsch removes best proper rigid transform from initial geometry. Segment means pool shared visible point observations within each sequence. Group/dataset summaries weight sequences equally.',
      ratio='RMS ratio=residual RMS/total RMS. Squared-displacement fraction=(RMS ratio)^2; neither is mechanical strain energy.',
      masks='Initial/current visibility intersection and finite coordinates; same tracked identities. At least3 noncollinear initial points. No rematching; no future points become predictor inputs.',
      caveats=['Ratios reference initial frame, not frame-to-frame velocities','Visibility changes alter sampled surfaces','Reconstruction/tracking noise contributes to residual','Low deformation under these loads does not prove rigid material','No material/force identification claim','Multiple episodes share assets and are not independent objects'],
      small_motion_floor_m=MOTION_FLOOR_M,source_sha256=sha256(Path(__file__)),freezes={n:sha256(REPORT/n) for n in ['method-freeze.json','feasibility-freeze.json']},protocol_hashes={n:sha256(REPORT/n) for n in ['benchmark-protocol.json','feasibility-protocol.json']},sequence_count=len(rows),rows=rows,aggregates={k:aggregate(v) for k,v in groups.items()})
    write_json(REPORT/'motion-diagnosis.json',report)
    lines=['EVALUATOR-ONLY MOTION DIAGNOSIS —148 PREDECLARED SEQUENCES','',report['definition'],report['ratio'],'','Suffix summaries (equal sequence weight; median ratios):','group | n | total RMS mm(mean) | nonrigid RMS mm(mean) | RMS ratio median | squared-displacement fraction median']
    for key,value in report['aggregates'].items():
        if '/ ALL' not in key and not key.startswith(tuple(cohorts)):continue
        a=value['suffix'];lines.append(f"{key} | {a['sequences']} | {a['total_vector_rms_mm']['mean']:.3f} | {a['rigid_removed_vector_rms_mm']['mean']:.3f} | {a['nonrigid_rms_ratio']['median']:.5f} | {a['nonrigid_squared_displacement_fraction']['median']:.6f}")
    lines+=['','Interpretation: small residual relative to total motion helps explain a strong rigid-motion baseline. It is not proof that elasticity is absent or unidentifiable in every trial. Visibility and tracking noise can inflate residual. Full per-frame masks/counts and prefix/suffix summaries are in the JSON.','The 1 micrometer denominator floor marks tiny-motion ratios undefined; frame0 normally falls here. No scores or prediction outputs were read. This diagnostic must not be fed back into frozen predictor selection.']
    (REPORT/'motion-diagnosis.txt').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
