"""Additive primary-cohort scientific plot, guarded by completed fresh audits."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/coupled-friction-v13';PARENT=ROOT/'reports/coupled-surface-v13'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def inputs(report,parent):
    audit_path=report/'independent-audit.json';audit=read(audit_path)
    if audit.get('passed') is not True or audit.get('errors') or audit.get('pending'):raise RuntimeError('Completed passing independent audit required')
    for path,h in audit['evidence_sha256'].items():
        if sha(ROOT/path)!=h:raise RuntimeError('Stale audited evidence: '+path)
    freeze_path=report/'method-freeze.json';freeze=read(freeze_path)
    verifier=ROOT/'scripts/coupled_contact_v13/fine_grid_verify.py'
    if audit.get('verifier_sha256')!=sha(verifier):raise RuntimeError('Audit verifier source changed')
    for path,h in freeze['source_sha256'].items():
        if sha(ROOT/path)!=h:raise RuntimeError('Fine frozen implementation changed: '+path)
    if freeze['source_sha256'].get(str(verifier.relative_to(ROOT)))!=sha(verifier):raise RuntimeError('Verifier absent from current freeze')
    if freeze['protocol_sha256']!=sha(report/'protocol.json'):raise RuntimeError('Fine freeze protocol changed')
    summary_path=report/'summary.json';summary=read(summary_path)
    if summary.get('independent_audit',{}).get('sha256')!=sha(audit_path) or summary['independent_audit'].get('passed') is not True:raise RuntimeError('Summary lacks current passed audit')
    p=read(report/'protocol.json')
    if summary['protocol_sha256']!=sha(report/'protocol.json'):raise RuntimeError('Stale summary protocol')
    for path,h in summary['source_hashes'].items():
        if sha(ROOT/path)!=h:raise RuntimeError('Stale summary source: '+path)
    objects=[o['id'] for o in p['objects'] if o['split']=='reserved'];scenes=p['primary_forecast_scenarios']
    keys=[(o,s) for o in objects for s in scenes]
    if len(objects)!=8 or len(keys)!=24 or len(set(keys))!=24:raise RuntimeError('Fixed eight-object24-primary contract required')
    tables=[]
    for folder in (parent,report):
        path=folder/'reserved-results.json'
        if audit['evidence_sha256'].get(str(path))!=sha(path) and audit['evidence_sha256'].get(str(path.relative_to(ROOT)))!=sha(path):raise RuntimeError('Reserved report not bound to audit')
        rows=read(path)['cases'];lookup={(r['object'],r['scenario']):r for r in rows}
        expected={(o,s) for o in objects for s in p['forecast_scenarios']}
        if len(lookup)!=len(rows) or set(lookup)!=expected or any(r['split']!='reserved' for r in rows):raise RuntimeError('Incomplete/duplicate/undeclared reserved rows')
        tables.append(lookup)
    if summary['cohorts']['reserved']['primary']['expected']!=24 or summary['cohorts']['reserved']['primary']['present']!=24:raise RuntimeError('Incomplete fixed primary summary')
    files=[audit_path,freeze_path,verifier,summary_path,report/'protocol.json',parent/'reserved-results.json',report/'reserved-results.json']
    return keys,scenes,tables,{str(f):sha(f) for f in files}

def paired(keys,coarse,fine,metric,comparator):
    points=[];missing=[]
    for key in keys:
        a=(coarse if comparator=='coarse_fem' else fine)[key]['results'].get('fem' if comparator=='coarse_fem' else 'rigid',{})
        b=fine[key]['results'].get('fem',{})
        if not a.get('complete') or not b.get('complete'):
            missing.append({'object':key[0],'scenario':key[1],'comparator_complete':bool(a.get('complete')),'fine_fem_complete':bool(b.get('complete'))});continue
        x,y=float(a[metric])*1000,float(b[metric])*1000
        if not np.isfinite([x,y]).all() or min(x,y)<0:raise ValueError('Invalid complete metric')
        points.append({'object':key[0],'scenario':key[1],'x_mm':x,'y_mm':y,'joint_reference_qualified':bool(fine[key].get('joint_reference_qualified',False))})
    return points,missing

def generate(report=REPORT,parent=PARENT):
    keys,scenes,(coarse,fine),hashes=inputs(report,parent)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,9),layout='constrained')
    colors=dict(zip(scenes,['#0072B2','#D55E00','#009E73']));records={}
    for col,metric in enumerate(('coordinate_rmse_m','shape_rmse_m')):
        panels=[]
        for row,comp in enumerate(('coarse_fem','fine_rigid')):
            pts,missing=paired(keys,coarse,fine,metric,comp);panels.append((pts,missing));records[comp+'/'+metric]={'expected':24,'matched':len(pts),'missing_count':len(missing),'missing':missing,'points':pts}
        largest=max([v for pts,_ in panels for p in pts for v in (p['x_mm'],p['y_mm'])]+[.01])*1.08
        for row,comp in enumerate(('coarse_fem','fine_rigid')):
            ax=axes[row,col];pts,missing=panels[row]
            for scene in scenes:
                subset=[p for p in pts if p['scenario']==scene]
                for qualified in (True,False):
                    group=[p for p in subset if p['joint_reference_qualified']==qualified]
                    ax.scatter([p['x_mm'] for p in group],[p['y_mm'] for p in group],label=scene if qualified else None,facecolors=colors[scene] if qualified else 'none',s=38,alpha=.85,edgecolors=colors[scene],linewidths=.9)
            ax.plot([0,largest],[0,largest],color='.5',linestyle='--',linewidth=1)
            ax.set(xlim=(0,largest),ylim=(0,largest),aspect='equal',xlabel=('Coarse-grid FEM' if row==0 else 'Fine-grid rigid')+' error (mm)',ylabel='Fine-grid FEM error (mm)',title=('World-coordinate RMS' if col==0 else 'Rigid-aligned shape RMS')+f'\nMatched {len(pts)}/24; missing/failed {len(missing)}')
            ax.grid(alpha=.2);ax.text(.03,.97,'Below diagonal favors fine FEM\nHollow = reference unqualified',transform=ax.transAxes,va='top',fontsize=8)
    axes[0,0].legend(fontsize=8,loc='lower right')
    fig.suptitle('Secondary friction-grid study — fixed primary cohort only\n24 interactions from 8 correlated objects; all failures counted, no quality filtering',fontsize=12)
    stem=report/'fine-grid-prediction-comparison'
    fig.savefig(stem.with_suffix('.png'),dpi=180);fig.savefig(stem.with_suffix('.svg'));plt.close(fig)
    # Inputs must remain unchanged throughout plotting.
    for path,h in hashes.items():
        if sha(Path(path))!=h:raise RuntimeError('Plot inputs changed while rendering')
    provenance={'scope':'Secondary study, primary 24 only. Points are interactions, not independent objects; 8 objects. Both metrics shown; matched pairs only with all missing cases recorded. Sharp stress excluded by fixed prospective scope, never by quality. Native adaptation is not plotted here and remains in full tables.','inputs_sha256':hashes,'source_sha256':sha(Path(__file__)),'panels':records,'outputs_sha256':{str(stem.with_suffix(ext)):sha(stem.with_suffix(ext)) for ext in ('.png','.svg')},'visual_inspection':'pending'}
    stem.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');return provenance

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,default=REPORT);parser.add_argument('--parent',type=Path,default=PARENT);a=parser.parse_args();generate(a.report,a.parent)
if __name__=='__main__':main()
