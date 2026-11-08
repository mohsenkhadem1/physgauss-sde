"""Shared fixed 3DGS appearance on native MASIV trajectories, not fitted appearance."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'runs/gap-crossmethod-v02/masiv'
OUT = ROOT/'output/gap-crossmethod-v02/masiv'
OUT.mkdir(parents=True, exist_ok=True)
SIZE=384

def renderer():
    eye = np.array([1.10,.44,1.08])
    target = np.array([.51,.13,.5])
    z = target-eye; z/=np.linalg.norm(z)
    x = np.cross(z,[0,1,0]); x/=np.linalg.norm(x)
    y = np.cross(z,x)
    w2c = np.eye(4); w2c[:3,:3]=np.stack([x,y,z]); w2c[:3,3]=-w2c[:3,:3]@eye
    p=np.zeros((4,4)); f=1/np.tan(np.deg2rad(32)/2)
    p[0,0]=p[1,1]=f; p[2,2]=10/(10-.01); p[2,3]=-10*.01/(10-.01); p[3,2]=1
    tensor=lambda a: torch.as_tensor(a,dtype=torch.float32,device='cuda')
    cfg=GaussianRasterizationSettings(image_height=SIZE,image_width=SIZE,tanfovx=1/f,tanfovy=1/f,
        kernel_size=.1,subpixel_offset=torch.zeros(SIZE,SIZE,2,device='cuda'),bg=torch.ones(3,device='cuda'),
        scale_modifier=1.,viewmatrix=tensor(w2c.T.copy()),projmatrix=tensor((p@w2c).T.copy()),
        sh_degree=0,campos=tensor(eye),prefiltered=False,debug=False)
    return GaussianRasterizer(cfg)

def render_frames(xs,initial,render):
    n=len(initial)
    # Fixed textured colors tied to particle identity, independent of hidden physics.
    q=(initial-initial.min(0))/(initial.max(0)-initial.min(0))
    colors=np.stack([.10+.45*q[:,0],.25+.45*q[:,1],.5+.35*q[:,2]],axis=1)
    colors=torch.tensor(colors,dtype=torch.float32,device='cuda')
    rotations=torch.zeros(n,4,device='cuda'); rotations[:,0]=1
    kwargs=dict(colors_precomp=colors,opacities=torch.full((n,1),.85,device='cuda'),
                scales=torch.full((n,3),.009,device='cuda'),rotations=rotations)
    frames=[]
    with torch.no_grad():
        for a in xs:
            xyz=torch.tensor(a,device='cuda')
            rgb,_=render(means3D=xyz,means2D=torch.zeros_like(xyz),**kwargs)
            frames.append(rgb.permute(1,2,0).cpu().numpy())
    return np.stack(frames)

def main():
    render=renderer()
    rows=[]
    fontpath=Path('/home/bayespc2/.local/share/fonts/arial/Arial.ttf')
    if not fontpath.exists():
        fontpath=next(Path('/home/bayespc2/.local/share/fonts/arial').glob('*.TTF'))
    font=ImageFont.truetype(str(fontpath),22)
    arrays={}
    for shape in ['torus','ellipsoid']:
        traj={}
        imgs={}
        for scale in [1,2]:
            for force in [0,20]:
                name=f'{shape}-scale{scale}-force{force}'
                traj[scale,force]=np.load(DATA/(name+'.npz'))['x']
                imgs[scale,force]=render_frames(traj[scale,force],traj[scale,force][0],render)
        for force in [0,20]:
            a,b=imgs[1,force],imgs[2,force]
            mask=np.any(np.minimum(a,b)<.99,axis=-1)
            mse=float(np.mean((a-b)**2)); fg_mse=float(np.mean((a-b)[mask]**2))
            d=np.linalg.norm(traj[1,force]-traj[2,force],axis=-1)
            rows.append(dict(shape=shape,force_N=force,full_psnr_db=float(-10*np.log10(max(mse,1e-15))),
                foreground_psnr_db=float(-10*np.log10(max(fg_mse,1e-15))),
                trajectory_point_RMS_mm=float(np.sqrt(np.mean(d*d))*1000),
                final_point_RMS_mm=float(np.sqrt(np.mean(d[-1]**2))*1000),max_point_difference_mm=float(d.max()*1000)))
        arrays[shape]=(traj,imgs)
    (OUT/'metrics.json').write_text(json.dumps(dict(classification='native solver with shared fixed Gaussian appearance; not fitted MASIV render scores',rows=rows),indent=2)+'\n')
    for frame in range(25):
        canvas=Image.new('RGB',(4*SIZE,2*(SIZE+58)+70),'white'); draw=ImageDraw.Draw(canvas)
        draw.text((15,8),'MASIV native mechanics + shared fixed Gaussian appearance (controlled diagnostic)',fill='black',font=font)
        for r,shape in enumerate(['torus','ellipsoid']):
            traj,imgs=arrays[shape]
            for c,(scale,force) in enumerate([(1,0),(2,0),(1,20),(2,20)]):
                image=Image.fromarray(np.uint8(np.clip(imgs[scale,force][frame],0,1)*255))
                yy=50+r*(SIZE+58)
                canvas.paste(image,(c*SIZE,yy+28))
                draw.text((c*SIZE+8,yy),f'{shape}: '+('passive' if force==0 else '20 N pulse')+f', world {scale}',font=font,fill='black')
        canvas.save(OUT/f'frame-{frame:03d}.png')
    Image.open(OUT/'frame-024.png').save(OUT/'comparison.png')
    print(json.dumps(rows))

if __name__=='__main__': main()
