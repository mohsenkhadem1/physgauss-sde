"""Camera-orbit comparisons of actual and predicted settled geometry."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
import numpy as np
import cv2
import imageio_ffmpeg
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,OUT
from scripts.phystwin.sequential_fixture import points
from scripts.position_ood.pilot import ACTIONS


def quads():
    shape=(9,3,3);faces=[]
    for axis in range(3):
        others=[j for j in range(3) if j!=axis]
        for side in [0,shape[axis]-1]:
            for i in range(shape[others[0]]-1):
                for j in range(shape[others[1]]-1):
                    row=[]
                    for a,b in [(0,0),(1,0),(1,1),(0,1)]:
                        ix=[0,0,0];ix[axis]=side;ix[others[0]]=i+a;ix[others[1]]=j+b;row.append(np.ravel_multi_index(ix,shape))
                    faces.append(row)
    return np.array(faces)


def action_label(name, amplitude):
    if name=='rest':return 'nominal tool poses'
    if name=='twist_x':return 'opposed x rotations: +/-0.15 rad'
    if name=='bend_z':return 'right-tool z rotation: +0.15 rad'
    if name=='stretch':return f'axial stretch: +/-{500*amplitude:g} mm'
    if name=='opposed_y':return f'opposed y motion: +/-{1000*amplitude:g} mm'
    if name=='diagonal':return f'diagonal: ({500*amplitude:g}, {-1000*amplitude:g}, {500*amplitude:g}) mm'
    return f'{name.replace("_", " ")}: {1000*amplitude:+g} mm'


def camera_basis(angle):
    az=angle;el=.35
    view=np.array([np.cos(el)*np.sin(az),-np.cos(el)*np.cos(az),np.sin(el)]);right=np.cross(view,[0,0,1.]);right/=np.linalg.norm(right);up=np.cross(right,view)
    return np.stack([right,up,view],axis=1)


def shared_scale(fields,angles,w=600,h=300):
    positions=np.concatenate([points()+field for field in fields],axis=0).reshape(-1,3)
    if not np.isfinite(positions).all():raise ValueError('Video geometry must be finite')
    centered=positions-np.array([.12,0.,-.005]);extent=np.zeros(2)
    for angle in angles:extent=np.maximum(extent,np.abs(centered@camera_basis(angle))[:,:2].max(axis=0))
    # Reserve the same header/footer margins in every panel, at every orbit angle.
    room=np.array([w/2-18,min(h*.53-50,h-40-h*.53)])
    return float(min(w/.36,h/.24,*list(room/np.maximum(extent,1e-12))))


def draw(image,positions,rect,angle,color,label,error=None,scale=None):
    x0,y0,w,h=rect;center=np.array([.12,0.,-.005])
    Q=camera_basis(angle);xyz=(positions-center)@Q;scale=min(w/.36,h/.24) if scale is None else scale;xy=np.c_[x0+w/2+scale*xyz[:,0],y0+h*.53-scale*xyz[:,1]].astype(np.int32);faces=quads();depth=xyz[faces,2].mean(1)
    for face in faces[np.argsort(depth)]:
        normal=np.cross(positions[face[1]]-positions[face[0]],positions[face[3]]-positions[face[0]]);normal/=max(np.linalg.norm(normal),1e-12);shade=.55+.45*abs(float(normal@np.array([.3,-.4,.866])));fill=tuple(int(c*shade) for c in color);cv2.fillConvexPoly(image,xy[face],fill,cv2.LINE_AA);cv2.polylines(image,[xy[face]],True,(80,85,90),1,cv2.LINE_AA)
    for p in xy:cv2.circle(image,tuple(p),2,(30,35,40),-1,cv2.LINE_AA)
    cv2.putText(image,label,(x0+18,y0+30),cv2.FONT_HERSHEY_SIMPLEX,.63,(30,35,40),1,cv2.LINE_AA)
    if error is not None:cv2.putText(image,f'Coordinate RMSE: {error:.3f} mm',(x0+18,y0+h-15),0,.52,(40,45,50),1,cv2.LINE_AA)
    bar_x=x0+w-110;bar_y=y0+h-15;cv2.line(image,(bar_x,bar_y),(bar_x+round(.05*scale),bar_y),(40,45,50),2,cv2.LINE_AA);cv2.putText(image,'50 mm',(bar_x,bar_y-9),0,.4,(40,45,50),1,cv2.LINE_AA)


def main(a):
    if 'test' in a.split.split('_'):
        from scripts.position_ood.seal import verify
        verify()
    actions=ACTIONS
    if a.split.startswith('transfer_'):
        from scripts.position_ood.transfer import TRANSFER_ACTIONS
        actions=TRANSFER_ACTIONS
    elif a.split=='force_law_test':actions=json.loads((BASE/'force-law-protocol.json').read_text())['actions']
    ref=np.load(BASE/f'fields/{a.split}-{a.object}.npz')['field'];ours=np.load(BASE/f'inverse/{a.split}-{a.object}/{a.method}-n{a.mesh}-p3.npz')['predicted'];pt_tag=a.phystwin_tag or f'y{a.initial:g}-dt1';pt=np.load(BASE/f'phystwin/{a.split}-{a.object}/{pt_tag}/predictions.npz')['predicted'];mpm=np.load(BASE/f'{a.mpm_tag}/{a.split}-{a.object}/predictions.npz')['predicted'];rows=[('Reference',ref,(105,185,120)),('Position-conditioned inverse',ours,(220,160,85)),('PhysTwin core adaptation',pt,(175,140,220)),('PhysGaussian MPM adaptation',mpm,(190,185,95))]
    dest=OUT/f'{a.split}-{a.object}-actual-vs-predicted.mp4';OUT.mkdir(exist_ok=True);W,H=1200,760;fps=24
    angles=np.linspace(-.7,.7,fps*2);scale=shared_scale([fields for _,fields,_ in rows],angles)
    process=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(fps),'-i','-','-an','-vcodec','libx264','-crf','20','-pix_fmt','yuv420p',str(dest)],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    try:
        for action in range(len(ref)):
            for frame in range(fps*2):
                image=np.full((H,W,3),248,np.uint8);angle=angles[frame]
                for k,(name,fields,color) in enumerate(rows):
                    error=None if k==0 else 1000*np.sqrt(np.mean((fields[action]-ref[action])**2));draw(image,points()+fields[action],((k%2)*600,65+(k//2)*310,600,300),angle,color,name,error,scale=scale)
                phase='OBSERVED PREFIX' if action<3 else 'HELD-OUT ACTION';name=action_label(*actions[action]);cv2.putText(image,f'{a.split.upper()} OBJECT {a.object} | {phase}: {name}',(20,32),0,.62,(25,30,35),1,cv2.LINE_AA)
                cv2.putText(image,'Actual deformation scale. Camera orbit around settled shapes; this is not a dynamic rollout.',(20,725),0,.57,(35,40,45),1,cv2.LINE_AA);process.stdin.write(image.tobytes())
                if action==6 and frame==fps:cv2.imwrite(str(dest.with_suffix('.png')),image)
    finally:process.stdin.close();code=process.wait()
    assert code==0
    sources=[BASE/f'fields/{a.split}-{a.object}.npz',BASE/f'inverse/{a.split}-{a.object}/{a.method}-n{a.mesh}-p3.npz',BASE/f'phystwin/{a.split}-{a.object}/{pt_tag}/predictions.npz',BASE/f'{a.mpm_tag}/{a.split}-{a.object}/predictions.npz']
    metadata=dict(split=a.split,object=a.object,frames=len(ref)*fps*2,fps=fps,resolution=[W,H],deformation_scale=1.,phystwin_tag=pt_tag,mpm_tag=a.mpm_tag,observed_actions=[0,1,2],held_actions=list(range(3,len(ref))),source_sha256={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},video_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),scope='Camera orbit around separately settled reference and predicted configurations. No dynamic trajectory or intervening shape is synthesized. Errors are coordinate RMSE at the declared 81 query points.')
    metadata.update(pixels_per_metre=scale,camera_center_m=[.12,0.,-.005],framing='One shared zoom across all methods, actions and orbit angles; no deformation amplification.',renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dest.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n');print(dest)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--object',type=int,default=1);p.add_argument('--initial',type=float,default=3.);p.add_argument('--phystwin-tag');p.add_argument('--method',default='extrapolated');p.add_argument('--mesh',type=int,default=3);p.add_argument('--mpm-tag',default='mpm-layer0.006');main(p.parse_args())
