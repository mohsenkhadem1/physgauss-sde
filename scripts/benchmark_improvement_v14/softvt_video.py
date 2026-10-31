"""Fixed-case suffix-only comparison; execute only after root releases outcome access."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--authorized-outcome-access',action='store_true');args=parser.parse_args()
    if not args.authorized_outcome_access:raise RuntimeError('Root outcome-access authorization required before evaluator loading')
    protocol_path=ROOT/'reports/benchmark-improvement-v14/protocol.json';protocol=json.loads(protocol_path.read_text())
    manifest=json.loads(args.manifest.read_text());cases=manifest['cases']
    fresh=protocol['fresh_softvt_cases'][0]
    expected=['object-soft-task0-demo2','object-soft-task0-demo5',fresh['name']]
    if [c['name'] for c in cases]!=expected:raise RuntimeError('Video cases must match prospective fixed order')
    from scripts.contact_benchmark_v12.data import load
    from scripts.external_benchmark.evaluate import expand_prediction
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    outdir=ROOT/'reports/benchmark-improvement-v14/videos';outdir.mkdir(parents=True,exist_ok=True)
    records=[]
    for case in cases:
        data=load(case['spec']);start=int(data['train_end']);end=len(data['times']);path=ROOT/case['prediction_path']
        with np.load(path,allow_pickle=False) as archive:
            original=expand_prediction(data,archive[case.get('original_key','contact_scene_state')])
            new=expand_prediction(data,archive[case.get('new_key','benchmark_v14')])
        truth=np.asarray(data['evaluation']['points']);sequences=[truth,original,new]
        finite=np.concatenate([x[start:end].reshape(-1,3) for x in sequences]);finite=finite[np.isfinite(finite).all(1)]
        low,high=finite.min(0),finite.max(0);center=(low+high)/2;radius=max(float(np.max(high-low))*.55,.02);low=center-radius;high=center+radius
        target=outdir/(case['name']+'.mp4');fig=plt.figure(figsize=(15,5));axes=[fig.add_subplot(1,3,i+1,projection='3d') for i in range(3)]
        writer=FFMpegWriter(fps=20,codec='libx264',extra_args=['-pix_fmt','yuv420p'])
        with writer.saving(fig,str(target),100):
            for t in range(start,end):
                for ax,sequence,title in zip(axes,sequences,['Actual simulation','Original contact_scene_state','Prefix-slip prediction']):
                    ax.clear();points=sequence[t];good=np.isfinite(points).all(1)
                    if good.all():ax.scatter(*points.T,s=3,c='#237f99',depthshade=True)
                    else:ax.text2D(.1,.5,'Prediction incomplete / failed',transform=ax.transAxes,color='red')
                    ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=title);ax.set_box_aspect((1,1,1));ax.view_init(23,-65+20*(t-start)/max(end-start-1,1))
                fig.suptitle(f'{case["name"]} | FORECAST SUFFIX ONLY | t={data["times"][t]:.2f}s | real-time playback | shared world axes')
                writer.grab_frame()
        plt.close(fig)
        records.append(dict(name=case['name'],path=str(target.relative_to(ROOT)),sha256=sha(target),frames=end-start,fps=20,frame_interval=[start,end],prediction_sha256=sha(path),source_data_sha256=sha(Path(data['source'])),axis_low=low.tolist(),axis_high=high.tolist(),visual_review_required=True))
    evidence={str(args.manifest):sha(args.manifest),str(protocol_path.relative_to(ROOT)):sha(protocol_path),str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    (outdir/'provenance.json').write_text(json.dumps(dict(cases=records,input_sha256=evidence,scope='Fixed cases, suffix-only forecast; shared axes/camera, no deformation exaggeration; visual QA pending'),indent=2)+'\n')
if __name__=='__main__':main()
