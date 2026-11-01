"""Reporting-only renderer; explicit authorization plus committed-artifact gates."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main(default_manifest):
    parser=argparse.ArgumentParser();parser.add_argument('--authorized-outcome-access',action='store_true');args=parser.parse_args()
    if not args.authorized_outcome_access:raise RuntimeError('Root must authorize target access after all forecasts commit')
    manifest_path=ROOT/default_manifest;manifest=json.loads(manifest_path.read_text());hashes={str(manifest_path.relative_to(ROOT)):sha(manifest_path)}
    for path in manifest['required_before_target_access']:
        p=ROOT/path
        if not p.is_file():raise RuntimeError('Required committed artifact missing: '+path)
        hashes[path]=sha(p)
    if manifest.get('configuration_pending'):raise RuntimeError('Manifest incomplete; resolve paths before outcome authorization')
    from scripts.contact_benchmark_v12.data import load
    from scripts.external_benchmark.evaluate import expand_prediction
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    out=ROOT/manifest['output_directory'];out.mkdir(parents=True,exist_ok=True);records=[]
    for case in manifest['cases']:
        data=load(case['spec']);start=int(data['train_end']);end=len(data['times']);fps=20 if data['dataset']=='SoftVTBench'else 30
        sequences=[np.asarray(data['evaluation']['points'])];titles=['Actual simulation' if data['dataset']=='SoftVTBench'else'Observed reconstructed points'];sources={}
        for panel in case['panels']:
            p=ROOT/panel['path']
            with np.load(p,allow_pickle=False)as a:pred=a[panel['key']].copy()
            if pred.shape!=(end,len(data['nodes']),3):raise RuntimeError('Expected nodal predictor array')
            sequences.append(expand_prediction(data,pred));titles.append(panel['title']);sources[panel['path']]=sha(p)
        visible=np.asarray(data['evaluation']['mask'],bool)
        cloud=np.concatenate([x[start:end].reshape(-1,3)for x in sequences]);cloud=cloud[np.isfinite(cloud).all(1)]
        low,high=cloud.min(0),cloud.max(0);center=(low+high)/2;radius=max(float(np.max(high-low))*.55,.02);low=center-radius;high=center+radius
        fig=plt.figure(figsize=(5*len(sequences),5));axes=[fig.add_subplot(1,len(sequences),i+1,projection='3d')for i in range(len(sequences))];target=out/(case['name']+'.mp4');failed=[0]*len(sequences)
        with FFMpegWriter(fps=fps,codec='libx264',extra_args=['-pix_fmt','yuv420p']).saving(fig,str(target),100)as writer:
            for t in range(start,end):
                for i,(ax,sequence,title)in enumerate(zip(axes,sequences,titles)):
                    ax.clear();points=sequence[t];good=np.isfinite(points).all(1)
                    if i==0:good &= visible[t]
                    if (i==0 and good.any())or(i>0 and good.all()):ax.scatter(*points[good].T,s=2,c='#237f99',depthshade=True)
                    else:ax.text2D(.1,.5,'Unavailable / failed prediction',transform=ax.transAxes,color='red');failed[i]+=1
                    ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=title);ax.set_box_aspect((1,1,1));ax.view_init(23,-65+20*(t-start)/max(end-start-1,1))
                fig.suptitle(f'{case["name"]} | FORECAST SUFFIX ONLY | t={data["times"][t]:.2f}s | real-time playback | shared world axes')
                writer.grab_frame()
        plt.close(fig)
        if any(sha(ROOT/p)!=h for p,h in sources.items()):raise RuntimeError('Prediction changed during rendering')
        records.append(dict(name=case['name'],path=str(target.relative_to(ROOT)),sha256=sha(target),frames=end-start,fps=fps,frame_interval=[start,end],prediction_sha256=sources,unavailable_frames_by_panel=dict(zip(titles,failed)),axis_low=low.tolist(),axis_high=high.tolist(),source_data_sha256=sha(Path(data['source'])),visual_review_required=True))
    if any(sha(ROOT/p)!=h for p,h in hashes.items()):raise RuntimeError('Media gate evidence changed')
    hashes[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    (out/'provenance.json').write_text(json.dumps(dict(cases=records,input_sha256=hashes,scope=manifest['scope'],visual_review_required=True),indent=2)+'\n')
