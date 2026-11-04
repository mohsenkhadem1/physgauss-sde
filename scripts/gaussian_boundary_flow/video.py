import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
import subprocess
from scripts.gaussian_boundary_flow.model import Block
from scripts.gaussian_boundary_flow.visual import render,cameras
from physgauss.camera import project


def main():
    torch.set_num_threads(2)
    root=Path('runs/gaussian-boundary-flow-v01');out=Path('reports/gaussian-boundary-flow-v01');out.mkdir(exist_ok=True,parents=True)
    data=torch.load(root/'evaluation.pt',weights_only=True)
    b=Block();kind='reverse';truth=data['trajectories'][kind];pred=data['prediction'][kind]
    err=(truth[:,17:]-pred[:,17:]).square().mean((1,2,3)).sqrt()
    order=err.argsort();ids=[int(order[len(order)//2]),int(order[-1])]
    # Deterministic median and worst; never hand-pick favorable examples.
    images=[]
    with torch.no_grad():
        for i in ids:
            images.append((render(b,truth[i:i+1],96)[0].numpy(),render(b,pred[i:i+1],96)[0].numpy()))
    # Add explicitly labeled projected mesh edges for legibility; RGB remains the actual renderer output.
    times=torch.linspace(0,1,41);eye,R=cameras(b.mesh,times)
    uv=[]
    for i in ids:
        uv.append(tuple(project(u[i:i+1]+b.mesh.rest,eye,R,96)[0][0].numpy()*250/96 for u in (truth,pred)))
    frames=[]
    for t in range(41):
        canvas=Image.new('RGB',(768,600),'#101827');draw=ImageDraw.Draw(canvas)
        draw.text((18,12),'Gaussian boundary inference | force reversal | true scale | mesh overlay',fill='white')
        for row,(gt,pr) in enumerate(images):
            y=45+row*270
            for col,(rgb,label) in enumerate(((gt,'Actual simulation'),(pr,'Predicted posterior mean'))):
                tile=Image.fromarray((np.clip(rgb[t],0,1)*255).astype('uint8')).resize((250,250))
                canvas.paste(tile,(15+col*270,y+18));draw.text((15+col*270,y),label,fill='white')
                pts=uv[row][col][t]
                for a,bnode in b.mesh.edges.tolist():
                    pa=pts[a]+np.array([15+col*270,y+18]);pb=pts[bnode]+np.array([15+col*270,y+18])
                    if np.all(np.isfinite([pa,pb])):draw.line([tuple(pa),tuple(pb)],fill='#cbd5e1',width=1)
            draw.text((565,y+45),'Median case' if row==0 else 'Worst case',fill='white')
            draw.text((565,y+65),f'{err[ids[row]]*1000:.2f} mm future RMSE',fill='white')
        draw.text((565,15),f't={t*.025:.3f}s',fill='white')
        draw.text((565,155),'Observed prefix' if t<=16 else 'Future: reversed force',fill='#fbbf24')
        frames.append(np.asarray(canvas))
    ffmpeg=Path('runs/media-tooling/ffmpeg-7.0.2-amd64-static/ffmpeg')
    proc=subprocess.Popen([str(ffmpeg),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','768x600','-r','20','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(out/'actual-vs-predicted.mp4')],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    for frame in frames: proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait()!=0: raise RuntimeError('Video encoding failed')
    Image.fromarray(frames[30]).save(out/'preview.png')
    (out/'video-provenance.json').write_text(json.dumps({'case_indices':ids,'selection':'median and maximum future coordinate RMSE, all 128 cases','program':kind,'deformation_magnification':1,'playback':'20 fps of .025 s steps: 0.5x physical speed','input':'projected 2D tracks, not RGB','renderer':'existing physgauss.render.GaussianRenderer'},indent=2))

if __name__=='__main__':main()
