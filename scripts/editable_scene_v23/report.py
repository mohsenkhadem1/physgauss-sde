"""Summarize mechanics integration, explicitly separate sensitivity from uncertainty."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/editable-scene-v23';REPORT=ROOT/'reports/editable-scene-v23'

def main():
    rows=[];documents={}
    for kind in ('lift_release','drop','force_pulse'):
        for E in (6000,12000,24000):
            key=f'{kind}-{E}-dt5e-05';d=json.loads((BASE/key/'result.json').read_text());assert d['complete'] and d['min_J']>0
            documents[key]=d
            rows.append(dict(kind=kind,young_pa=E,**{k:d[k]for k in ('max_deformation_mm','max_penetration_mm','min_J','max_energy_work_drift_J')},force_impulse_error_Ns=d.get('force_impulse_error_Ns')))
    with np.load(BASE/'drop-12000-dt5e-05/states.npz')as a,np.load(BASE/'drop-12000-dt2.5e-05/states.npz')as b:
        assert np.allclose(a['times'],b['times']);delta=np.linalg.norm(a['positions']-b['positions'],axis=-1)
        sensitivity=dict(mean_mm=float(delta.mean()*1000),max_mm=float(delta.max()*1000),scope='Single timestep-halving comparison, not mesh convergence or real-world validation.')
    result=dict(complete=True,rollouts=10,rows=rows,timestep_sensitivity=sensitivity,
        scope='Editable mechanics integration pilot with specified parameters, analytic elastic law and known boundaries. No scene encoder trained. Hypothesis envelope is not calibrated uncertainty.',
        limitations=['Coarse mesh, finite-penalty floor contact, no friction','Camera projection only; no new appearance model','Not a new PhysTwin/SoftVTBench benchmark result'])
    (REPORT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    for col,kind in enumerate(('lift_release','drop','force_pulse')):
        for E in (6000,12000,24000):
            d=documents[f'{kind}-{E}-dt5e-05'];t=[r['time_s']for r in d['rows']]
            axis=0 if kind=='force_pulse' else 2;offset=d['rows'][0]['centroid_m'][axis] if kind=='force_pulse' else 0.
            axes[0,col].plot(t,[(r['centroid_m'][axis]-offset)*1000 for r in d['rows']],label=f'{E/1000:g} kPa')
            axes[1,col].plot(t,[r['deformation_rms_mm']for r in d['rows']])
        axes[0,col].set_title(kind.replace('_',' '));axes[0,col].legend();axes[1,col].set_xlabel('Time (s)')
    axes[0,2].set_ylabel('Centre x displacement (mm)');axes[0,0].set_ylabel('Centre height (mm)');axes[1,0].set_ylabel('Deformation after rigid alignment (mm)')
    fig.suptitle('Editable scene pilot: specified material hypotheses, not calibrated uncertainty')
    fig.savefig(REPORT/'pilot.png',dpi=150);fig.savefig(REPORT/'pilot.svg')
    p=REPORT/'pilot.svg';p.write_text('\n'.join(l.rstrip()for l in p.read_text().splitlines())+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
