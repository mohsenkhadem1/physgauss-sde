"""Publish the failed TRAIN timestep refinement without changing sealed results."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root,output):
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    analysis=json.loads((root/'analysis.json').read_text())
    copies={'analysis.json':root/'analysis.json','post-validation.json':root/'post-validation.json',
            'protocol.json':root/'protocol.json','execution-original-errors.json':root/'execution.json'}
    for name,source in copies.items():shutil.copyfile(source,output/name)
    plot_data={};fig,axs=plt.subplots(1,2,figsize=(10.8,4.1),layout='constrained')
    for rate,color in zip([480,960,1920],['#267b9b','#c75c27','#7d4e9e']):
        folder=root/f'hz{rate}'/'captures/private/t0001'
        audit=json.loads((folder/'validation.json').read_text())
        if not audit['passed']:raise ValueError('Unvalidated capture')
        with np.load(folder/'evaluation/trajectory.npz') as z:q,t=z['positions_m'].astype(float),z['times_s']
        with np.load(folder/'reference.npz') as z:rest=z['rest_points_m']
        rms=np.sqrt(np.mean((q-rest)**2,axis=(1,2)))*1e6
        plot_data[str(rate)]={'times_s':t.tolist(),'coordinate_rms_from_rest_um':rms.tolist()}
        axs[0].plot(t,rms,label=f'{rate:,} Hz',color=color,lw=2)
    axs[0].set(xlabel='Time (s)',ylabel='Coordinate RMS motion (µm)',title='Same TRAIN layout and force')
    axs[0].legend(frameon=False);axs[0].grid(alpha=.18)
    pairs=[r for r in analysis['refinement_pairs'] if r['action_id']=='evaluation']
    values=[r['coordinate_rmse_m']*1e6 for r in pairs]
    axs[1].bar(['480 → 960 Hz','960 → 1,920 Hz'],values,color=['#267b9b','#7d4e9e'],width=.6)
    for i,v in enumerate(values):axs[1].text(i,v+5,f'{v:.1f}',ha='center')
    axs[1].set(ylabel='Between-rate coordinate RMSE (µm)',title='Refinement difference increases',ylim=(0,max(values)*1.2))
    axs[1].grid(axis='y',alpha=.18);axs[1].set_axisbelow(True)
    for suffix in ['png','pdf']:fig.savefig(output/('timestep-refinement.'+suffix),dpi=170)
    plt.close(fig)
    (output/'plot-data.json').write_text(json.dumps(plot_data,indent=2)+'\n')
    (output/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Isaac numerical reliability diagnosis</title><style>body{max-width:1050px;margin:36px auto;padding:0 22px;font:17px/1.6 system-ui;color:#18252d}img{width:100%;height:auto}a{color:#126580}h1{line-height:1.2}p{max-width:90ch}.status{background:#fff0e5;padding:16px;border-left:4px solid #c75c27}table{border-collapse:collapse}td,th{padding:8px 20px;border-bottom:1px solid #ccd5db;text-align:right}th:first-child,td:first-child{text-align:left}</style>
<h1>Repeatable motion, failed timestep refinement</h1>
<p class="status">The tested Isaac setup has not passed numerical reliability checks. Further predictive fitting, bulk capture and active probing comparison wait for a controlled diagnosis.</p>
<p>The same fixed-patch TRAIN block, mesh, material, attachment geometry, camera and diagonal force were simulated at three physics rates. Full-trajectory coordinate RMS motion is 111.3,236.6 and148.4µm at480,960 and1,920Hz. Adjacent-rate differences are181.2 and275.4µm: comparable to the response itself and increasing with refinement. These three levels do not establish which response is accurate.</p>
<img src="timestep-refinement.png" alt="Motion traces at three rates differ; the finer adjacent-rate difference is larger">
<p>The fresh960Hz positions, velocities and force commands reproduce the previous960Hz run bit for bit. Zero-force motion remains about4.17nm. The commanded pulse impulse agrees with its analytic integral to1.38×10⁻¹¹N·s, saved clocks match the representable float32 timestep, and every rate passes the original sensor-state, mesh, fixture and force checks.</p>
<p>The original validator assumed a candidate named poke_0. These numerical captures name it zero_force. The original KeyError failures are retained; a source-pinned adapter changes exactly two reset-reference dictionary keys and reruns every check with its original tolerances. No trajectory or simulation result is repaired.</p>
<p>This is one-layout TRAIN diagnosis. It does not identify solver error, floating-point accumulation, finite deformation or attachment effects as the cause. The next declared tests vary solver iterations and translate the complete physical scene. Passing those checks would still leave mesh and attachment-model fidelity unresolved.</p>
<p><a href="analysis.json">Full numerical results and input hashes</a> · <a href="post-validation.json">Validation adapter provenance</a> · <a href="execution-original-errors.json">Preserved original validation errors</a> · <a href="protocol.json">Executed protocol</a> · <a href="plot-data.json">Plot data</a> · <a href="timestep-refinement.pdf">Figure PDF</a></p>
<p><a href="../../configs/gaussian_campaign_numerical_gate_v01.json">Numerical reliability gate</a> · <a href="../../docs/GAUSSIAN_RECOVERY_CAMPAIGN.md">Active campaign</a> · <a href="../gaussian-sealed-study-v01/index.html">Earlier sealed study</a></p></html>''')
    inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in copies.values()}
    inputs[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/'provenance.json').write_text(json.dumps({'source_and_result_sha256':inputs,'reserved_responses_opened':False},indent=2)+'\n')
    print(json.dumps({'output':str(output),'rate_difference_um':values}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='runs/gaussian-campaign-v01/numerical-timestep-v01');p.add_argument('--output',default='reports/gaussian-numerical-reliability-v01');main(**vars(p.parse_args()))
