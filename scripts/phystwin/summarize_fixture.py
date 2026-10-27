"""Report frozen spatial and quadrature gates without excluding failed refinements."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/phystwin-fixture-convergence-v04'
OUT=ROOT/'reports/phystwin-fixture-convergence-v04'


def compare(a,b,p,quadrature=False):
    d=a['sampled_u']-b['sampled_u']
    r=dict(rms_mm=float(np.sqrt(np.mean(d*d))*1000),max_mm=float(abs(d).max()*1000))
    for key in ['strain','traction']:
        r[key+'_relative']=float(np.linalg.norm(a[key]-b[key])/np.linalg.norm(b[key]))
    for key in ['elastic_J','spring_J']:
        r[key+'_relative']=float(abs(a[key]/b[key]-1))
    r['checks']={
        'displacement_rms':r['rms_mm']<=1000*p['quadrature_displacement_rms_tolerance_m' if quadrature else 'displacement_rms_tolerance_m'],
        'energy':max(r['elastic_J_relative'],r['spring_J_relative'])<=p['quadrature_energy_relative_tolerance' if quadrature else 'energy_relative_tolerance']}
    if not quadrature:
        r['checks'].update(displacement_max=r['max_mm']<=1000*p['displacement_max_tolerance_m'],strain=r['strain_relative']<=p['strain_relative_L2_tolerance'],traction=r['traction_relative']<=p['traction_relative_L2_tolerance'])
    for key,tol in [('force_balance_N','force_balance_tolerance_N'),('moment_balance_Nm','moment_balance_tolerance_Nm'),('residual_N','nonlinear_residual_tolerance_N')]:
        r[key]=float(max(a[key],b[key]));r['checks'][key]=r[key]<=p[tol]
    r['minimum_J']=float(min(a['minimum_J'],b['minimum_J']));r['checks']['positive_J']=r['minimum_J']>0
    r['pass']=all(r['checks'].values());return r


def main():
    p=json.loads((BASE/'protocol.json').read_text());rows=[];qualified={}
    for action in p['actions']:
        for tag in ['p1','p2','p2g']:
            files=sorted((BASE/'fields').glob(f'{tag}-n*-q3-{action}.npz'),key=lambda f:int(f.name.split('-')[1][1:]))
            for a,b in zip(files,files[1:]):
                rows.append(dict(action=action,kind='spatial',coarse=a.stem,fine=b.stem,**compare(np.load(a),np.load(b),p)))
        for b in sorted((BASE/'fields').glob(f'p2g-n*-q4-{action}.npz')):
            a=b.with_name(b.name.replace('-q4-','-q3-'))
            rows.append(dict(action=action,kind='quadrature',coarse=a.stem,fine=b.stem,**compare(np.load(a),np.load(b),p,True)))
        spatial=[r for r in rows if r['action']==action and r['kind']=='spatial' and r['fine'].startswith('p2g-')]
        last=spatial[-2:]
        finest=last[-1]['fine'] if last else ''
        quad=[r for r in rows if r['action']==action and r['kind']=='quadrature' and r['coarse']==finest]
        qualified[action]=len(last)==2 and all(r['pass'] for r in last) and bool(quad) and quad[0]['pass']
    result=dict(protocol=p,comparisons=rows,qualified=qualified,all_actions_qualified=all(qualified.values()),criterion='Last two graded spatial refinements and finest-mesh q3/q4; both elastic and spring energies checked.')
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['| Action | Comparison | RMS mm | Max mm | Strain % | Traction % | Elastic energy % | Spring energy % | Pass |','|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['action']} | {r['coarse']} → {r['fine']} | {r['rms_mm']:.5f} | {r['max_mm']:.5f} | {100*r['strain_relative']:.3f} | {100*r['traction_relative']:.3f} | {100*r['elastic_J_relative']:.3f} | {100*r['spring_J_relative']:.3f} | {r['pass']} |")
    (OUT/'convergence.md').write_text('\n'.join(lines)+'\n');print(json.dumps(qualified))

if __name__=='__main__':main()
