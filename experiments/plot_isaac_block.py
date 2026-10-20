"""Create a scientific overview and an animation of actual Isaac RGB frames."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def plot(root, output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    spec=json.loads((root/'truth/boundaries.json').read_text())
    raw=np.load(root/'truth/trajectory.npz');ref=np.load(root/'reference.npz')
    p=raw['positions_m'];rest=ref['rest_points_m'];faces=ref['surface_faces'];times=raw['times_s']
    centered=rest-rest.mean(0);elastic=[]
    for q in p:
        moving=q-q.mean(0);u,_,vt=np.linalg.svd(centered.T@moving)
        rotation=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
        elastic.append(np.sqrt(np.mean((moving-centered@rotation)**2)))
    peak=int(np.argmax(elastic))
    fig=plt.figure(figsize=(12,8),layout='constrained')
    grid=fig.add_gridspec(2,2)
    for cell,index,title in [(grid[0,0],0,'Isaac RGB: reference'),(grid[0,1],peak,'Isaac RGB: peak elastic deformation')]:
        ax=fig.add_subplot(cell);ax.imshow(Image.open(root/'observations'/f'rgb-{index:04d}.png'))
        ax.set_title(f'{title} (t={times[index]:.2f} s)');ax.axis('off')
    ax=fig.add_subplot(grid[1,0],projection='3d')
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    for e in edges:
        q=p[peak,e]*1000;ax.plot(q[:,0],q[:,1],q[:,2],color='#677e99',alpha=.3,lw=.45)
    ids=spec['fixture_nodes'];q=p[peak,ids]*1000
    ax.scatter(q[:,0],q[:,1],q[:,2],s=55,color='#d72f35',label='Fixed material point',depthshade=False)
    q=p[peak,spec['load_nodes']]*1000
    ax.scatter(q[:,0],q[:,1],q[:,2],s=35,color='#087b44',label='Force application nodes',depthshade=False)
    for i,xyz in enumerate(q):ax.text(*xyz,f' {i+1}',fontsize=9)
    ax.set(xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',title='Ground-truth boundaries (diagnostic view)')
    ax.set_box_aspect((3,1.5,1));ax.legend(loc='upper left',fontsize=8)
    ax=fig.add_subplot(grid[1,1]);force=raw['force_step_commands_N'];clock=raw['force_step_midpoints_s']
    for i in range(force.shape[1]):ax.plot(clock,np.linalg.norm(force[:,i],axis=-1),label=f'Handle {i+1}')
    ax.set(xlabel='Time [s]',ylabel='Commanded force [N]',title='Known load schedule and measured elastic motion')
    ax.legend(loc='upper left',fontsize=8);ax.grid(alpha=.2)
    other=ax.twinx();other.plot(times,np.asarray(elastic)*1000,'k--',lw=1.5)
    other.set_ylabel('Elastic coordinate RMS [mm] (dashed)')
    fig.suptitle('Boundary-learning simulator: random pin and random force pulses\nKnown material; forward simulation only',fontsize=14)
    fig.savefig(output/'overview.png',dpi=150);plt.close(fig)
    frames=[]
    for i,t in enumerate(times):
        frame=Image.open(root/'observations'/f'rgb-{i:04d}.png').convert('RGB')
        draw=ImageDraw.Draw(frame);draw.rectangle((8,8,405,33),fill='white')
        draw.text((15,15),f'Isaac Sim GPU FEM | seed {spec["seed"]} | t={t:.2f} s',fill='black')
        frames.append(frame)
    frames[0].save(output/'preview.gif',save_all=True,append_images=frames[1:],duration=round(1000*(times[1]-times[0])),loop=0)
    (output/'figure-provenance.json').write_text(json.dumps({'source':str(root),'peak_elastic_frame':peak,
        'peak_time_s':float(times[peak]),'rgb':'Actual Isaac renders without amplified deformation',
        'elastic_metric':'Coordinate RMS after best rigid alignment; excludes rigid rotation about pin'},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode');parser.add_argument('--output',required=True)
    args=parser.parse_args();plot(args.episode,args.output)
