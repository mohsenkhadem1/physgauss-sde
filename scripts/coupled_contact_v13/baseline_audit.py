"""Evaluator-only baseline provenance and accounting audit; never changes forecasts."""
from pathlib import Path
import ast,collections,hashlib,json,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/coupled-contact-v13';BASE=ROOT/'runs/coupled-contact-v13'

def read(p):return json.loads(p.read_text())
def main():
    p=read(REPORT/'protocol.json');freeze=read(REPORT/'method-freeze.json')
    mismatch=[name for name,h in freeze['source_sha256'].items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h]
    upstream=ROOT/'runs/third_party/PhysTwin';source=upstream/'qqtt/model/diff_simulator/spring_mass_warp.py'
    committed=subprocess.check_output(['git','-C',str(upstream),'show','HEAD:qqtt/model/diff_simulator/spring_mass_warp.py'],text=True)
    def kernel(text):return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='eval_springs'),include_attributes=False)
    same=kernel(committed)==kernel(source.read_text())
    checkout=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    lines=['BASELINE FINAL AUDIT (evaluator only; status below)',f'Frozen source mismatches: {mismatch}',
        f'Author PhysTwin checkout: {checkout}; eval_springs AST identical: {same}',
        'Full local spring source SHA256: '+hashlib.sha256(source.read_bytes()).hexdigest(),
        'Native label: PhysTwin native spring-force core + shared v13 contact/integrator; NOT full PhysTwin.',
        'Native original supports multi-object particle collision. This comparison replaces its optimizer, graph, mass convention, collision and integrator.',
        'Equal 35 scalar-scale/friction hypotheses for FEM and native; rigid has 5 friction hypotheses. Same initial nodes, lumped masses, contact geometry, area weighting, time step and positive-J guard.',
        'Native effective spring scale is not FEM Young modulus. Native7point log grid 1e3–1e6; FEM7point grid2e3–1.8e4. Same count does not imply identical expressivity or equal viable candidates.',
        'The reference is finer-mesh same-constitutive-family FEM, favoring the FEM model family. Neither official PhysTwin nor real-data generalization is measured.',
        'Empirical position baselines see positions, tool kinematics and time; they do not receive scene collision geometry. Rigid/native shared-contact baselines address that difference.','', 'LIBRARY ACCOUNTING']
    libraries={};all_complete=True
    for family,expected in [('fem',35),('native',35),('rigid',5)]:
        rows=[read(x) for x in sorted((BASE/'library'/family).glob('*/complete.json'))]
        good=[r for r in rows if r['success']];bad=[r for r in rows if not r['success']]
        all_complete &=len(rows)==expected
        libraries[family]=dict(expected=expected,present=len(rows),success=len(good),failures=len(bad))
        lines.append(f'{family}: {len(rows)}/{expected} accounted; {len(good)} success; {len(bad)} failures')
        errors=collections.Counter(r.get('error','unknown') for r in bad)
        lines.append('  Failure reasons: '+str(dict(errors)))
        lines.append('  Successful scale/friction candidates: '+str([(r['job']['scale'],r['job']['friction']) for r in good]))
    lines+=['Native spring particles have no tetrahedral volume constraint; common positive-J rejection is a study-specific constraint, not evidence that the authors full method fails.','','CALIBRATION SELECTIONS']
    selections=[];bad_argmin=[];score_mismatches=[]
    library_arrays={}
    for family in libraries:
        library_arrays[family]=[]
        for j in sorted((BASE/'library'/family).glob('*/complete.json'),key=lambda q:tuple(map(int,q.parent.name.split('-')))):
            status=read(j);trajectory=j.parent/'trajectory.npz'
            if status['success']:
                with np.load(trajectory) as data:library_arrays[family].append(data['frames'].copy())
            else:library_arrays[family].append(None)
    for o in p['objects']:
        path=BASE/'inference'/o['id']/'selection.json'
        if not path.exists():all_complete=False;continue
        d=read(path)
        with np.load(path.parent/'observations.npz') as obs:
            observed=obs['observed'].copy();mask=obs['mask'].copy()
        for family,r in d.items():
            replay=np.array([float(np.mean((arr[mask]-observed[mask])**2)) if arr is not None else np.inf for arr in library_arrays[family]])
            if not np.allclose(replay,r['scores_m2'],rtol=1e-12,atol=1e-15):score_mismatches.append((o['id'],family))
            if r.get('success'):
                scores=np.asarray(r['scores_m2']);correct=int(np.argmin(scores))==r['index']
                if not correct:bad_argmin.append((o['id'],family))
                lines.append(f"{o['split']} {o['id']} {family}: scale={r['scale']:.8g}, friction={r['friction']}, calibrationRMSE={1000*r['calibration_rmse_m']:.4f}mm, scaleEdge={r['scale_at_grid_edge']}, frictionEdge={r['friction_at_grid_edge']}, finiteCandidates={int(np.isfinite(scores).sum())}/{len(scores)}, argminVerified={correct}")
                selections.append(dict(split=o['split'],object=o['id'],family=family,**r))
            else:lines.append(f"{o['id']} {family}: FAILED {r}")
    lines.append('Rigid scaleEdge is uninformative because rigid has a single unused scale.')
    for split in ['development','validation','reserved']:
        lines+=['',split.upper()+' BASELINE COMPARISONS']
        path=REPORT/f'{split}-results.json'
        if not path.exists():lines.append('Not complete yet');all_complete=False;continue
        d=read(path);rows=d['cases'];all_complete &=len(rows)==d['expected_cases']
        lines.append(f"Cases {len(rows)}/{d['expected_cases']}; jointly qualified references {sum(bool(r.get('joint_reference_qualified')) for r in rows)}")
        for family in ['fem','native','rigid','position_ridge','position_rigid','persistence','oracle']:
            valid=[r['results'][family] for r in rows if r.get('results',{}).get(family,{}).get('complete')]
            errs=[1000*r['coordinate_rmse_m'] for r in valid];shapes=[1000*r['shape_rmse_m'] for r in valid]
            lines.append(f'{family}: complete {len(valid)}/{len(rows)}, coordinate mean mm={np.mean(errs) if errs else None}, rigid-aligned shape mean mm={np.mean(shapes) if shapes else None}; means conditional if incomplete')
            for r in rows:
                v=r.get('results',{}).get(family,{})
                if not v.get('complete'):lines.append(f"  unavailable {r['object']}/{r['scenario']}: {v.get('failure',r.get('unavailable_reason','missing'))}")
        for family in ['native','rigid','oracle']:
            paired=[(r['results']['fem']['coordinate_rmse_m'],r['results'][family]['coordinate_rmse_m']) for r in rows if r.get('results',{}).get('fem',{}).get('complete') and r.get('results',{}).get(family,{}).get('complete')]
            if paired:
                paired=np.array(paired)*1000;gain=paired[:,1]-paired[:,0]
                lines.append(f'Paired FEM versus {family}: n={len(paired)}, FEMmean={paired[:,0].mean():.6f}mm baselineMean={paired[:,1].mean():.6f}mm, FEMwins={int((gain>1e-9).sum())}, losses={int((gain < -1e-9).sum())}, ties={int((abs(gain)<=1e-9).sum())}')
        for family in ['native','rigid']:
            chosen=[r for r in selections if r['split']==split and r['family']==family]
            lines.append(f"{family} calibration edge hits: scale {sum(r['scale_at_grid_edge'] for r in chosen)}/{len(chosen)}, friction {sum(r['friction_at_grid_edge'] for r in chosen)}/{len(chosen)}")
    lines+=['',f'ACCOUNTING COMPLETE: {all_complete}',f'Selection argmin mismatches: {bad_argmin}',f'Independent calibration score replay mismatches: {score_mismatches}',
            'Visual source audit: four panels use shared world limits and camera orbit; 100Hz saved samples at25fps are4xslow. Fixed curved case is a spherical obstacle, not bowl/container. Renderer missing/nonfinite frames and scene bounds should be explicitly checked in final visual QA.',
            'No forecast outcomes used to edit models, grids or selection. Audit generated from immutable result files; incomplete jobs remain pending rather than omitted.']
    (REPORT/'baseline-final-audit.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(accounting_complete=all_complete,source_mismatches=mismatch,kernel_identical=same,argmin_mismatches=bad_argmin,calibration_score_mismatches=score_mismatches,libraries=libraries)))
if __name__=='__main__':main()
